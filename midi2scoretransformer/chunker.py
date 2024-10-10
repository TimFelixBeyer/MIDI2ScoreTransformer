"""
This module contains code for chunking MIDI and MusicXML files into pseudo-measures using
beat-level annotations from the ASAP dataset to align the MIDI and MusicXML.

We use a greedy algorithm to align the MIDI and MusicXML files by moving notes that are
close to the beat boundaries to the next/previous measure if that improves the alignment.
The resulting chunks are saved as JSON files next to the performance-MIDI files.
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pretty_midi
from joblib import Parallel, delayed

from dataset import ASAPDataset
from tokenizer import MultistreamTokenizer


def make_measures(midi, midi_score, mxl, annots, swap=True):
    measures = {"midi": [], "mxl": []}
    # TODO: Maybe do beats instead of downbeats?
    swap_tol = 0.05
    measures = {"midi": [], "mxl": []}
    i, j = (0, 0)
    for sb_seconds in annots["midi_score_beats"]:
        # Convert the score beats from seconds to musical time using
        # the provided MIDI scores
        sb = midi_score.time_to_tick(sb_seconds) / midi_score.resolution
        measures["mxl"].append([])
        while j < len(mxl) and mxl[j].offset < sb:
            measures["mxl"][-1].append((j, mxl[j].pitch.midi))
            j += 1

    # Use the performance beats to split the MIDI into pseudo-measures of beat length
    for pb in annots["performance_beats"]:
        measures["midi"].append([])
        while i < len(midi) and midi[i].start < pb - swap_tol:
            measures["midi"][-1].append((i, midi[i].pitch))
            i += 1

    measures["midi"].append([])
    measures["mxl"].append([])
    while i < len(midi):
        measures["midi"][-1].append((i, midi[i].pitch))
        i += 1
    while j < len(mxl):
        measures["mxl"][-1].append((j, mxl[j].pitch.midi))
        j += 1

    if not swap:
        for i in range(len(measures["midi"])):
            measures["midi"][i] = [i_ for i_, p in measures["midi"][i]]
            measures["mxl"][i] = [i_ for i_, p in measures["mxl"][i]]
        measures["swapped"] = False
        return measures
    from collections import Counter

    n_swaps = 0
    swap_tol = 0.5
    for i in range(len(measures["midi"]) - 1):
        swapped = True
        while swapped:
            swapped = False
            # Figure out which MIDI pitches can be moved forward/backward by one beat
            # because that would yield better alignment. Only considers notes within
            # 0.5s of the beat boundary.
            c_midi = Counter([p for _, p in measures["midi"][i]])
            c_mxl = Counter([p for _, p in measures["mxl"][i]])

            c_midi_next = Counter([p for _, p in measures["midi"][i + 1]])
            c_mxl_next = Counter([p for _, p in measures["mxl"][i + 1]])
            too_much = c_midi - c_mxl
            lacking = c_mxl - c_midi
            too_much_next = c_midi_next - c_mxl_next
            lacking_next = c_mxl_next - c_midi_next
            can_be_moved_forward = too_much & lacking_next
            can_be_moved_backward = too_much_next & lacking
            for pitch in can_be_moved_forward:
                last_j, last_p = measures["midi"][i][-1]
                # Only swap notes if they are within 0.5s of the beat boundary
                if (
                    last_p == pitch
                    and annots["performance_beats"][i] - midi[last_j].start < swap_tol
                ):
                    measures["midi"][i + 1].insert(0, measures["midi"][i].pop())
                    n_swaps += 1
                    swapped = True
                    break

            for pitch in can_be_moved_backward.keys():
                first_j, first_p = measures["midi"][i + 1][0]
                # ---only swap notes if they are within 0.5s of the beat boundary
                if (
                    first_p == pitch
                    and midi[first_j].start - annots["performance_beats"][i] < swap_tol
                ):
                    measures["midi"][i].append(measures["midi"][i + 1].pop(0))
                    n_swaps += 1
                    swapped = True
                    break

    # print("Swaps", n_swaps, n_swaps/len(midi))
    # If we had to swap a lot of notes, we probably messed up and should just return
    # the raw alignment...
    if n_swaps / len(midi) > 0.1:
        return make_measures(midi, midi_score, mxl, annots, swap=False)
    for i in range(len(measures["midi"])):
        measures["midi"][i] = sorted([j for j, p in measures["midi"][i]])
        measures["mxl"][i] = sorted([j for j, p in measures["mxl"][i]])
    assert max([max(m + [0]) for m in measures["midi"]]) == len(midi) - 1
    assert sum(len(m) for m in measures["midi"]) == len(midi)
    assert max([max(m + [0]) for m in measures["mxl"]]) == len(mxl) - 1
    assert sum(len(m) for m in measures["mxl"]) == len(mxl)
    measures["swapped"] = True
    return measures


def handle_file(midi_path, mxl_path, save_path):
    annots = annotations[midi_path.replace("./data/asap-dataset/", "")]
    if not annots["score_and_performance_aligned"]:
        return
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, s = MultistreamTokenizer.mxl_to_list(mxl_path)
        measures = make_measures(
            MultistreamTokenizer.midi_to_list(midi_path),
            pretty_midi.PrettyMIDI(
                mxl_path.replace("xml_score.musicxml", "midi_score.mid")
            ),
            m,
            annots,
        )

    if os.path.exists(save_path):
        try:
            prev = json.load(open(save_path))
        except json.decoder.JSONDecodeError:
            print(save_path)
            raise
        # Print differences
        if prev["midi"] != measures["midi"]:
            print("midi", prev["midi"], measures["midi"])
        if prev["mxl"] != measures["mxl"]:
            print("mxl", prev["mxl"], measures["mxl"])

    json.dump(measures, open(save_path, "w"))


if __name__ == "__main__":
    annotations = json.load(open("data/asap-dataset/asap_annotations.json"))
    skip = set(["data/asap-dataset/Glinka/The_Lark"])
    paths = []
    for root, dirs, files in os.walk("data/asap-dataset/"):
        for file in files:
            if file.endswith(".musicxml") and root not in skip:
                mxl_path = os.path.join(root, file)
                break
        else:
            continue
        for file in files:
            if file.endswith(".mid") and not file.startswith("midi_score"):
                midi_path = os.path.join(root, file)
                save_path = os.path.join(root, file.replace(".mid", "_chunks.json"))
                paths.append((midi_path, mxl_path, save_path))

    q = ASAPDataset("./data/", "all")
    midi_paths = [
        q.metadata.iloc[idx]["performance_MIDI_external"].replace(
            "{ASAP}", f"{q.data_dir}asap-dataset"
        )
        for idx in range(0, len(q))
    ]
    mxl_paths = [
        os.path.join(os.path.dirname(m), "xml_score.musicxml") for m in midi_paths
    ]
    save_paths = [m.replace(".mid", "_chunks.json") for m in midi_paths]
    paths = list(zip(midi_paths, mxl_paths, save_paths))
    Parallel(n_jobs=min(16, len(paths)), verbose=10)(
        delayed(handle_file)(midi_path, mxl_path, save_path)
        for midi_path, mxl_path, save_path in paths
    )
