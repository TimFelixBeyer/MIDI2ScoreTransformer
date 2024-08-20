"""Utilities for inference, including batched and chunked inference and postprocessing."""
import warnings

import torch
from muster import muster
from score_transformer import score_similarity

from tokenizer import MultistreamTokenizer
from score_utils import postprocess_score


def eval(y_hat, gt_mxl_path: str) -> dict[str, dict[str, float]|None]:
    mxl = MultistreamTokenizer.detokenize_mxl(y_hat)
    mxl = postprocess_score(mxl, inPlace=True)

    # fmt: off
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = {
            "mxl <-> gt_mxl": score_similarity_normalized(mxl, gt_mxl_path, full=False),
            "muster": muster(mxl, gt_mxl_path),
        }
    return sim
    # fmt: on

def score_similarity_normalized(est, gt, full=False):
    if est is None or gt is None:
        return {
            "Clef": None,
            "KeySignature": None,
            "TimeSignature": None,
            "NoteDeletion": None,
            "NoteInsertion": None,
            "NoteSpelling": None,
            "NoteDuration": None,
            "StemDirection": None,
            "Beams": None,
            "Tie": None,
            "StaffAssignment": None,
            "Voice": None,
        }
    sim = score_similarity(est, gt, full=full)
    new_sim = {}
    for k, v in sim.items():
        if v is None:
            new_sim[k] = None
        elif k == "n_Note" or any(key in k for key in ["F1", "Rec", "Prec", "TP", "FP", "FN", "TN"]):
            new_sim[k] = v
        else:
            new_sim[k] = v / sim["n_Note"]
    return new_sim



def quantize_path(path, model, **kwargs):
    """Quantize a midi file at `path` using the model `model`.
    The resulting score should be saved with makeNotation=False.
    """
    x = MultistreamTokenizer.tokenize_midi(path)
    y_hat = infer(x, model, **kwargs)
    mxl = MultistreamTokenizer.detokenize_mxl(y_hat)
    mxl = postprocess_score(mxl)
    return mxl


def infer(x, model, overlap=64, chunk=512, verbose=True, kv_cache=True) -> dict[str, torch.Tensor]:
    single_example = x['pitch'].ndim == 2
    if single_example:
        x = {k: v.unsqueeze(0) for k, v in x.items()}
    x = {k: v.to(model.device) for k, v in x.items()}
    if chunk <= overlap:
        raise ValueError("`chunk` must be greater than `overlap`.")
    y_full = None
    for i in range(0, max(x['pitch'].shape[1] - overlap, 1), chunk - overlap):
        if verbose:
            print("Infer", i, "/", x['pitch'].shape[1], end='\r')
        x_chunk = {k: v[:, i:i + chunk] for k, v in x.items()}
        if i == 0 or overlap == 0:  # No context required
            y_hat = model.generate(x=x_chunk, top_k=1, max_length=chunk, kv_cache=kv_cache)
        else:
            # Keep the last 'overlap' notes of the previous chunk as context
            y_hat_prev = {k: v[:, -overlap:] if k != 'pad' else v[:, -overlap:, 0] for k, v in y_full.items()}
            with torch.autocast(device_type='cuda'):
                y_hat = model.generate(x=x_chunk, y=y_hat_prev, top_k=1, max_length=chunk, kv_cache=kv_cache)
            y_hat = {k: v[:, overlap:] for k, v in y_hat.items()}

        if y_full is None:
            y_full = y_hat
        else:
            for k in y_full:
                y_full[k] = torch.cat((y_full[k], y_hat[k]), dim=1)
    if single_example:
        y_full = {k: v[0].cpu() for k, v in y_full.items()}
    else:
        y_full = {k: v.cpu() for k, v in y_full.items()}
    return y_full


def pad_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a batch of irregular tensors to the same length, then concat."""
    max_len = max([x["pitch"].shape[1] for x in batch])
    for x in batch:
        for k, v in x.items():
            pad_length = max_len - v.shape[1]
            shape = list(v.shape)
            shape[1] = pad_length
            x[k] = torch.cat(
                (v, torch.zeros(shape, dtype=v.dtype, device=v.device)), dim=1
            )
    # Concat along first dim:
    out = {}
    for k in batch[0].keys():
        out[k] = torch.cat([x[k] for x in batch], dim=0)
    return out
