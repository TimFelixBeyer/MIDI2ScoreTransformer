"""Utilities for inference, including batched and chunked inference and postprocessing."""
import warnings

import torch
from muster import muster
from score_transformer import score_similarity

from tokenizer import MultistreamTokenizer
from score_utils import postprocess_score

device = "cuda" if torch.cuda.is_available() else "cpu"


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
            with torch.autocast(device_type=device):
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


def cat_dict(
    a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], dim=0
) -> dict[str, torch.Tensor]:
    assert set(a.keys()) == set(b.keys())
    return {k: torch.cat([a[k], b[k]], dim=dim) for k in a.keys()}


def cut_pad(
    tensor: torch.Tensor, max_len: int, offset: int, pad_value: int = 0
) -> torch.Tensor:
    """
    Cut a tensor's first dimension to a maximum length and pad the tensor's first
    dimension to a minimum length.

    Args:
        tensor (Tensor): tensor to be cut or padded
        max_len (int): maximum length of the tensor's first dimension
        offset (int): offset to cut the tensor if too long
        pad_value (int): value used for padding, default is 0

    Returns:
        Tensor: tensor cut or padded along its first dimension to shape (max_len,)
        or (max_len, n_cols) if input is 2D
    """
    if tensor.dim() == 1:
        n = tensor.size(0)
        if n > max_len:
            tensor = tensor[offset : offset + max_len]
        elif n < max_len:
            pad_size = max_len - n
            pad = torch.full((pad_size,), pad_value, dtype=tensor.dtype)
            tensor = torch.cat((tensor, pad), dim=0)
    elif tensor.dim() == 2:
        n, n_cols = tensor.size()
        if n > max_len:
            tensor = tensor[offset : offset + max_len]
        elif n < max_len:
            pad_size = max_len - n
            pad = torch.full((pad_size, n_cols), pad_value, dtype=tensor.dtype)
            tensor = torch.cat((tensor, pad), dim=0)
    else:
        raise ValueError("Input tensor must be 1D or 2D.")

    return tensor