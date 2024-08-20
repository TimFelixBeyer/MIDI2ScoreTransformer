"""
This file contains a torch.utils.data.Dataset wrapper for the
ASAP dataset, which is a collection of MIDI files and corresponding MusicXML files.

The first run is significantly slower as metadata and caches are built.
Subsequent runs are much faster.
"""
from functools import lru_cache
import hashlib
import json
import os
import random
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import torch
from music21 import key, pitch
from torch.utils.data import Dataset

from constants import SKIP, TEST_PIECE_IDS, TO_IGNORE_INDICES
from utils import cat_dict, cut_pad
from tokenizer import MultistreamTokenizer


class ASAPDataset(Dataset):
    """Implements a torch-compatible interface to the ASAP Dataset"""
    def __init__(
        self,
        data_dir: str = "./data/",
        split: str = "train",
        seq_length: Optional[int] = None,
        cache: bool = True,
        padding: str = 'per-beat',
        augmentations: Dict[str, Union[float, Dict[str, float]]] = {},
        return_continous: bool = False,
        return_paths: bool = False,
        id: str="diffusion_2024_04_18",
    ):
        """
        Parameters
        ----------
        data_dir : str (default='./data/')
            Path to the data directory
        split : str (default='train')
            Which split to use. One of ["all", "train", "validation", "test"]
        seq_length : Optional[int]
            If not None, will cut/pad to this length
        cache : bool (default=True)
            Whether to cache the parsed MIDI/MXL files
        padding : str (default='per-beat')
            How to pad the data. One of ["per-beat", "end", None]
        augmentations : Dict[str, Union[float, Dict[str, float]]]
            Augmentations to apply to the data. If a key is not given, the augmentation
            is ignored.
            Possible keys are ["transpose", "tempo_jitter", "onset_jitter", "random_crop", "random_shift"]
            - transpose: int
                Whether to transpose the data by a random amount up to the given value.
            - random_crop: Union[bool, int]
                Whether to crop the data to a random length between 16
            - tempo_jitter: Tuple[float, float]
                Whether to jitter the tempo by a random amount between the given values.
            - onset_jitter: float
                Whether to jitter the onset by a random amount according to the given value.
                Multiplicative (the intra-onset intervals are scaled by N(1,onset_jitter^2)).
        return_continous : bool (default=False)
            Whether to return the data as continous values, or as a dictionary of bucketed tensors.
        id : str (default="diffusion_2023_10_13")
            A unique identifier for the dataset. This is used to ensure that the cache is not
            reused between different datasets.
        """
        # Get metadata
        self.data_dir = data_dir
        self.split = split
        self.seq_length = seq_length
        self.cache = cache
        assert padding in ('per-beat', 'end', None)
        self.padding = padding
        self.augmentations = augmentations
        self.return_continous = return_continous
        self.return_paths = return_paths
        self.id = id
        self.metadata = self._load_metadata(data_dir, split)

    def _load_metadata(self, data_dir: str, split: str) -> pd.DataFrame:
        data_real = pd.read_csv(data_dir + "/ACPAS-dataset/metadata_R.csv")
        data_synthetic = pd.read_csv(data_dir + "/ACPAS-dataset/metadata_S.csv")
        asap_annotations = json.load(
            open(data_dir + "/asap-dataset/asap_annotations.json")
        )
        UNALIGNED = set(
            "{ASAP}/" + k
            for k, v in asap_annotations.items()
            if not v["score_and_performance_aligned"]
        )
        # Filter
        data = pd.concat([data_real, data_synthetic])
        # Initial filtering
        data = data[(data["source"] == "ASAP") & data["aligned"]]
        data = data[~data["performance_MIDI_external"].isin(SKIP)]
        data = data[~data["performance_MIDI_external"].isin(UNALIGNED)]
        data = data.drop_duplicates(subset=["performance_MIDI_external"])
        # Filter by annotations
        data.reset_index(inplace=True)
        data.drop(TO_IGNORE_INDICES, inplace=True)

        # Select first piece from each composer for testing (may have multiple performances)
        # test_ids = data.groupby("composer").first()["piece_id"].values
        test_idx = data["piece_id"].isin(TEST_PIECE_IDS)

        if split == "all":
            return data
        elif split == "test":
            return data[test_idx]
        elif split == "validation":
            return data[(data["piece_id"] % 10 == 0) & (~data["piece_id"].isin(TEST_PIECE_IDS))]
        elif split == "train":
            d = data[(data["piece_id"] % 10 != 0) & (~data["piece_id"].isin(TEST_PIECE_IDS))]
            try:
                self.lengths = []
                for idx in range(len(d)):
                    sample = d.iloc[idx]
                    sample_path = sample["performance_MIDI_external"].replace(
                        "{ASAP}", f"{self.data_dir}asap-dataset"
                    )
                    # fmt: off
                    pkl_file = os.path.join(self.data_dir, "cache", f"{sha256(sample_path + self.id)}_.pkl")
                    # fmt: on
                    input_stream, output_stream = torch.load(pkl_file)
                    self.lengths.append(len(input_stream['onset']))
                self.lengths = torch.FloatTensor(self.lengths)
            # When creating the cache for the first time, we don't have the lengths yet,
            # so we will just sample uniformly.
            except FileNotFoundError as e:
                self.lengths = torch.ones(len(d))
            return d
        else:
            raise ValueError(f"Invalid split: {split}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if self.split == "train":
            idx: int = torch.multinomial(self.lengths, 1, replacement=True).item()
        sample = self.metadata.iloc[idx]
        sample_path = sample["performance_MIDI_external"].replace(
            "{ASAP}", f"{self.data_dir}asap-dataset"
        )
        sample_dir = os.path.dirname(sample_path)

        pkl_file = os.path.join(self.data_dir, "cache", f"{sha256(sample_path + self.id)}.pkl")

        if (not self.cache) or (not os.path.exists(pkl_file)):
            score_path = sample_dir + "/xml_score.musicxml"
            input_stream = MultistreamTokenizer.parse_midi(sample_path)
            output_stream = MultistreamTokenizer.parse_mxl(score_path)
            torch.save((input_stream, output_stream), pkl_file)

        input_stream, output_stream = torch.load(pkl_file)

        if self.augmentations.get("transpose", False):
            shift = random.randint(-6, 6)
            input_stream["pitch"], output_stream["pitch"], output_stream["accidental"], output_stream["keysignature"] = self._transpose(
                shift,
                midi_stream=input_stream["pitch"],
                mxl_stream=output_stream["pitch"],
                accidental_stream=output_stream["accidental"],
                keysignature_stream=output_stream["keysignature"]
            )

        if (v := self.augmentations.get("tempo_jitter", False)):
            jitter_onset = random.uniform(*v)
            jitter_duration = jitter_onset + random.uniform(-0.05, 0.05)
            input_stream["onset"] = input_stream["onset"] * jitter_onset
            input_stream["duration"] = input_stream["duration"] * jitter_duration
        if (v := self.augmentations.get("onset_jitter", False)):
            jitter = 1 + torch.randn(input_stream["onset"].shape) * v
            # adjust intervals between onsets
            inter_note_intervals = torch.diff(input_stream["onset"], prepend=torch.tensor([0]), dim=0)
            input_stream["onset"] = torch.cumsum(inter_note_intervals * jitter, dim=0)
        if (v := self.augmentations.get("velocity_jitter", False)):
            input_stream["velocity"] += torch.round(torch.randn(input_stream["velocity"].shape) * v).long()
            input_stream["velocity"] = torch.clamp(input_stream["velocity"], 1, 127)

        if self.return_continous:
            return input_stream, output_stream

        input_stream = MultistreamTokenizer.bucket_midi(input_stream)
        output_stream = MultistreamTokenizer.bucket_mxl(output_stream)

        if self.seq_length is not None:
            seq_length = self.seq_length
        else:
            # need buffer due to padding with 'per-beat' option
            seq_length = max(len(input_stream['onset']), len(output_stream['offset'])) + 256

        chunk_annots = json.load(open(sample_path.replace(".mid", "_chunks.json")))

        if (v := self.augmentations.get("random_crop", False)):
            min_beats = 16
            if v is True:
                n_0 = random.randint(0, max(len(chunk_annots["midi"]) - min_beats, 0))
            elif isinstance(v, int):
                average = sum([len(x) for x in chunk_annots["midi"]])/len(chunk_annots["midi"])
                n_0 = random.choice(range(0, max(len(chunk_annots["midi"]) - min_beats, 1), max(1, int(v/average))))
            else:
                raise ValueError("Invalid random_crop value")
        else:
            n_0 = 0

        def process_chunk(stream, chunk, padding, length):
            if padding == "per-beat":
                return {k: cut_pad(v[chunk], length, 0) for k, v in stream.items()}
            return {k: v[chunk] for k, v in stream.items()}

        new_input_stream = None  # just a sentry
        for midi_chunk, mxl_chunk in zip(
            chunk_annots["midi"][n_0:], chunk_annots["mxl"][n_0:]
        ):
            length = max(len(midi_chunk), len(mxl_chunk))
            if new_input_stream is not None and len(new_input_stream["onset"]) + length > seq_length + self.augmentations.get("random_shift", 0):
                break
            in_chunk = process_chunk(input_stream, midi_chunk, self.padding, length)
            out_chunk = process_chunk(output_stream, mxl_chunk, self.padding, length)
            if new_input_stream is None:
                new_input_stream = in_chunk
                new_output_stream = out_chunk
            else:
                new_input_stream = cat_dict(new_input_stream, in_chunk)
                new_output_stream = cat_dict(new_output_stream, out_chunk)
        if (v := self.augmentations.get("random_shift", False)):
            shift = random.randint(0, v - 1)
            for k, v in new_input_stream.items():
                new_input_stream[k] = v[shift:]
            for k, v in new_output_stream.items():
                new_output_stream[k] = v[shift:]
        if self.padding is not None:
            # Cut/Pad to exact seq-length
            for k, v in new_input_stream.items():
                input_stream[k] = cut_pad(v, seq_length, 0)
            for k, v in new_output_stream.items():
                output_stream[k] = cut_pad(v, seq_length, 0)
        if self.return_paths:
            return input_stream, output_stream, sample_path, sample_dir + "/xml_score.musicxml"
        return input_stream, output_stream

    # @lru_cache(None)
    @staticmethod
    def _accidental_map(p, a, i):
        def alter_map(accidental):
            alter_to_value = {None: 5, -2.0: 0, -1.0: 1, 0.0: 2, 1.0: 3, 2.0: 4}
            alter = accidental.alter if isinstance(accidental, pitch.Accidental) else accidental
            # 6 if not known
            return alter_to_value.get(alter, 6)

        if i is None:
            return a
        accidental_mapping = {0: 2, 1: 1, 2: 0, 3: -1, 4: -2}
        alter = accidental_mapping.get(a, 0)
        p_obj = pitch.Pitch()
        p_obj.midi = p + alter
        if a in accidental_mapping:
            p_obj.accidental = -accidental_mapping[a]
        p_obj.spellingIsInferred = False
        tp = p_obj.transpose(i)
        accepted_pitches = {
            'C', 'B#', 'D--', 'C#', 'B##', 'D-', 'D', 'C##', 'E--', 'D#', 'E-', 'F--',
            'E', 'D##', 'F-', 'F', 'E#', 'G--', 'F#', 'E##', 'G-', 'G', 'F##', 'A--',
            'G#', 'A-', 'A', 'G##', 'B--', 'A#', 'B-', 'C--', 'B', 'A##', 'C-'
        }
        if tp.name not in accepted_pitches:
            return None
        return alter_map(tp.accidental)

    @lru_cache(None)
    @staticmethod
    def _ks_map(ks: int, i: str):
        if i is None:
            return ks
        if ks == 15:
            return None
        k_obj = key.KeySignature(ks - 7)
        ns = k_obj.transpose(i).sharps
        if not -7 <= ns <= 7:
            return None
        return ns + 7

    @staticmethod
    def _transpose(shift, midi_stream, mxl_stream=None, accidental_stream=None, keysignature_stream=None):
        """Transpose pitches by a random amount between -6 and 6. If accidental_stream and
        keysignature_stream are provided, they will be adjusted following the procedure
        in https://arxiv.org/pdf/2107.14009.pdf

        In more detail, pitches are simply shifted by the desired amount.
        Then, all musical intervals with that shift are tried.
        If the transposed accidentals or key signatures are invalid, they are set to ignore_index.
        Among the valid transpositions, the one with the lowest number of accidentals is selected.

        Parameters
        ----------
        shift : int
            The amount of transposition
        midi_stream : torch.Tensor
            The MIDI pitch stream
        mxl_stream : torch.Tensor|None
            The MusicXML pitch stream, if provided.
        accidental_stream : torch.Tensor|None
            The accidental stream, if provided.
        keysignature_stream : torch.Tensor|None
            The key signature stream, if provided.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            The transposed MIDI pitch stream, the transposed MusicXML pitch stream
            the transposed accidental stream, and the transposed key signature stream.
        """
        assert (accidental_stream is None) == (keysignature_stream is None), "Either both or none of the accidentals and key signatures should be provided"
        if accidental_stream is not None:
            assert mxl_stream is not None, "Need mxl pitch stream alongside accidentals"

        def shift_pitch(stream, shift):
            stream = stream + shift
            stream[stream > 127] -= 12
            stream[stream < 0] += 12
            return stream

        midi_stream = shift_pitch(midi_stream, shift)
        results = [midi_stream]
        if mxl_stream is not None and accidental_stream is not None and keysignature_stream is not None:
            if shift == 0 and random.random() < 0.5:
                # Always include the original version some of the time.
                pass
            else:
                INTERVALS = {
                    -6: ["d5", "A4"],
                    -5: ["P5", "d6", "AA4"],
                    -4: ["m6", "A5"],
                    -3: ["M6", "d7", "AA5"],
                    -2: ["m7", "A6"],
                    -1: ["d1", "M7", "AA6"],
                    0: [None, "P1", "d2", "A7"],
                    1: ["m2", "A1"],
                    2: ["M2", "d3", "AA1"],
                    3: ["m3", "A2"],
                    4: ["M3", "d4", "AA2"],
                    5: ["P4", "A3"],
                    6: ["d5", "A4"],
                }
                intervals = INTERVALS[shift]
                m = mxl_stream.numpy()
                a = accidental_stream.numpy()
                ks = keysignature_stream.unique()
                best_error = float('inf')
                errors = dict()
                for interv in intervals:
                    accidental_cand = torch.zeros_like(accidental_stream)
                    keysignature_cand: torch.Tensor = keysignature_stream.clone()
                    for k in ks:
                        val = ASAPDataset._ks_map(int(k), interv)
                        if val is None:  # invalid key signature
                            accidental_cand.fill_(6)
                            keysignature_cand.fill_(15)
                            break
                        keysignature_cand[keysignature_cand == k] = val
                    else:  # valid keysignatures, so we look for accidentals
                        for i in range(len(mxl_stream)):
                            val = ASAPDataset._accidental_map(m[i], a[i], interv)
                            if val is None:  # invalid accidental
                                accidental_cand.fill_(6)
                                keysignature_cand.fill_(15)
                                break
                            accidental_cand[i] = val * 1.0
                    error = (accidental_cand[accidental_cand != 5] - 2).abs().sum()
                    errors[interv] = error
                    # print(f"Error: {error} for {interv}", accidental_cand)
                    if error < best_error:
                        best_error = error
                        accidental_stream = accidental_cand
                        keysignature_stream = keysignature_cand
        if mxl_stream is not None:
            mxl_stream = shift_pitch(mxl_stream, shift)
            results.append(mxl_stream)
        if accidental_stream is not None:
            results.append(accidental_stream)
            results.append(keysignature_stream)
        return results


def sha256(string: str) -> str:
    h = hashlib.new("sha256")
    h.update(string.encode())
    return h.hexdigest()

if __name__ == "__main__":
    print("Initializing QuantizationDataset")
    print("ASAP")
    for split in ("all", "train", "validation", "test"):
        q = ASAPDataset("./data/", split, seq_length=None, padding=None, cache=True, return_continous=False)
        print(split, len(q))
