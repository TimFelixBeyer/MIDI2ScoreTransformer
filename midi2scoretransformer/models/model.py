"""Base model class for the PM2S Transformer model."""
from typing import Dict
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class BaseModel(pl.LightningModule):
    def __init__(
        self, enc_configuration=None, dec_configuration=None, hyperparameters=None
    ):
        super().__init__()
        self.enc_config = enc_configuration
        self.dec_config = dec_configuration
        self.hyperparameters = hyperparameters
        self.save_hyperparameters()

    def forward(
        self,
        input_streams: torch.FloatTensor = None,
        output_streams: torch.FloatTensor = None,
    ) -> Dict[str, torch.Tensor]:
        encodings = self.forward_enc(
            input_streams, attention_mask=input_streams["pad"]
        )
        # encoder-decoder
        return self.forward_dec(
            input_streams=output_streams,
            encoder_hidden_states=encodings,
            encoder_attention_mask=input_streams["pad"],
        )

    @torch.no_grad()
    def generate(self, x, y=None, max_length=512, temperature=1.0, top_k=1):
        """Generate a sequence of tokens from the model.
        If y with T timesteps is provided, only max_length - T tokens will be generated.
        The first T tokens will be y_hist.
        """
        B, T, _ = x["pitch"].shape
        device = x["pitch"].device
        conf = self.dec_config
        # Model is used to the first tokens being all 0's & it will be overwritten anyways
        # fmt: off
        y_start_token = {
            "offset": torch.zeros((B, 1, conf.out_offset_vocab_size), device=device),
            "downbeat": torch.zeros((B, 1, conf.out_downbeat_vocab_size), device=device),
            "duration": torch.zeros((B, 1, conf.out_duration_vocab_size), device=device),
            "pitch": torch.zeros((B, 1, conf.out_pitch_vocab_size), device=device),
            "velocity": torch.zeros((B, 1, conf.out_velocity_vocab_size), device=device),
            "grace": torch.zeros((B, 1, conf.out_grace_vocab_size), device=device),
            "trill": torch.zeros((B, 1, conf.out_trill_vocab_size), device=device),
            "staccato": torch.zeros((B, 1, conf.out_staccato_vocab_size), device=device),
            "voice": torch.zeros((B, 1, conf.out_voice_vocab_size), device=device),
            "stem": torch.zeros((B, 1, conf.out_stem_vocab_size), device=device),
            "hand": torch.zeros((B, 1, conf.out_hand_vocab_size), device=device),
            "pad": torch.zeros((B, 1), device=device).long(),
        }
        # fmt: on
        if y is None:
            y = y_start_token
        else:
            y = {k: torch.cat([y_start_token[k], y[k]], dim=1) for k in y.keys()}
        T_cond = y["pad"].shape[1]

        encoder_hidden_states = self.forward_enc(
            x, attention_mask=x["pad"]
        )  # (B, T, D)
        encoder_attention_mask = x["pad"]

        for _ in range(max_length + 1 - T_cond):
            shifted_y = {k: torch.roll(v, -1, 1) for k, v in y.items()}
            y_pred = self.forward_dec(
                input_streams=shifted_y,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            for k in y.keys():
                # forward the model to get the logits for the index in the sequence
                logits = y_pred[k]
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature
                # ensure that we sample a downbeat wherever the offset decreases, since that guarantees a measure change!
                if k == "downbeat" and y_pred["offset"].shape[1] > 1:
                    is_downbeat = y_pred["offset"][:, -1].argmax(-1) < y_pred["offset"][:, -2].argmax(-1)
                    logits[is_downbeat, 0] = -float("Inf")

                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                # apply softmax to convert logits to (normalized) probabilities
                probs = (
                    F.softmax(logits, dim=-1)
                    if k != "pad"
                    else torch.cat([1 - F.sigmoid(logits), F.sigmoid(logits)], dim=-1)
                )
                # Greedy decoding (equivalent to argmax for topk = 1)
                next_token = torch.multinomial(probs, num_samples=1)  # 633 tok/s
                # sample from the distribution
                # next_token = torch.searchsorted(torch.cumsum(probs, dim=-1), torch.rand((B, 1)).to(probs.device)) # 660 tok/s
                if k == "pad":  # special case
                    y[k] = torch.cat([y[k], next_token], dim=1)
                else:
                    # Token back to one-hot
                    next_token = F.one_hot(
                        next_token, num_classes=y_pred[k].shape[-1]
                    )
                    y[k] = torch.cat([y[k], next_token], dim=1)

            # set other tokens zero where mask
            mask = y["pad"][:, -1] == 0
            for k in y.keys():
                if k != "pad":
                    y[k][mask, -1] = 0

        # Remove the <start> token
        for k in y.keys():
            y[k] = y[k][:, 1:]
        y["pad"] = y["pad"].unsqueeze(-1).float()
        return y