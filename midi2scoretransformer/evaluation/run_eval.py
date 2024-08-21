"""Given a path to a model checkpoint and a dataset split ('test', 'train', 'validation', 'all'),
compute all metrics."""

"""Evaluate the model end-to-end on the full songs.
Note that predictions are cached, so if you want to re-run the evaluation from scratch,
use the --nocache flag.
"""
import argparse
import os
import sys
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import ASAPDataset
from utils import eval, infer, pad_batch
from models.roformer import Roformer
from tokenizer import MultistreamTokenizer

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--fast_eval", action="store_true")
    args = parser.parse_args()

    q = ASAPDataset("./data/", args.split)
    batch_size = 16
    overlap = 64
    paths = []
    for i in range(len(q.metadata)):
        sample = q.metadata.iloc[i]
        sample_path = sample["performance_MIDI_external"].replace(
            "{ASAP}", f"{q.data_dir}asap-dataset"
        )
        score_path = os.path.dirname(sample_path) + "/xml_score.musicxml"
        paths.append((sample_path, score_path))

    from lightning.pytorch import seed_everything
    seed_everything(42, workers=True)
    print("Load + Tokenize all songs")
    inputs = []
    lengths = []
    for midi, gt_mxl in tqdm(paths):
        x = MultistreamTokenizer.tokenize_midi(midi)
        inputs.append({k: v.unsqueeze(0).to(device) for k, v in x.items()})
        lengths.append(x["pitch"].shape[0])

    # Sort everything by length
    sorted_data = sorted(zip(lengths, inputs, paths), key=lambda x: x[0])
    lengths, inputs, paths = zip(*sorted_data)

    print("Running inference")
    model = Roformer.load_from_checkpoint(args.model)
    model.to(device)
    model.eval()

    # First run everything through the model (batched)
    y_full = None
    for i in tqdm(range(0, len(inputs), batch_size)):
        x = pad_batch(inputs[i : i + batch_size])
        y_hat = infer(x, model, overlap=overlap, chunk=512, kv_cache=True)
        if y_full is None:
            y_full = y_hat
        else:
            y_full = pad_batch([y_full, y_hat])

    print(f"Computing score similarities")
    sims = Parallel(n_jobs=16, verbose=10)(
        delayed(eval)({k: v[i, :l] for k, v in y_full.items()}, p[1])
        for i, (p, l) in enumerate(zip(paths, lengths))
    )

    sims = {k: [d[k] for d in sims if d[k]] for k in sims[0]}
    print("-----------------")
    print("Aggregate:", {k: len([s for s in sims[k] if s is not None]) for k in sims}, len(sims['mxl <-> gt_mxl']))
    sims_aggregate = {}
    for k, v in sims.items():
        # v is list of dicts
        aggregate = {k_: [d[k_] for d in v if d[k_] is not None] for k_ in v[0]}
        if any(key in k for key in ["TP", "FP", "FN", "TN"]):
            aggregate = {k_: sum(v_) for k_, v_ in aggregate.items()}
        else:
            aggregate = {k_: sum(v_) / (len(v_)+1e-9) for k_, v_ in aggregate.items()}
        sims_aggregate[k] = aggregate
        print(k, aggregate)
    sims["aggregate"] = sims_aggregate

    print(f"Ours", end=" ")
    for k in ["PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER", "MeanER"]:
        print(f"{round(sims_aggregate['muster'][k], 2):5.2f}", end=" ")
    for k in ["NoteDeletion", "NoteInsertion", "NoteDuration", "StaffAssignment", "StemDirection", "NoteSpelling"]:
        print(f"{round(100*sims_aggregate['mxl <-> gt_mxl'][k], 2):5.2f}", end=" ")

    print("\nTable 4:")
    print("SOTA  6.86 25.03  9.67   -     -     -")
    print(f"Ours", end=" ")
    for k in ["StaffAssignment", "StemDirection", "NoteSpelling", "GraceF1", "StaccatoF1", "TrillF1"]:
        print(f"{round(100*sims_aggregate['mxl <-> gt_mxl'][k], 2):5.2f}", end=" ")
    print(f"\n{args.model}")
