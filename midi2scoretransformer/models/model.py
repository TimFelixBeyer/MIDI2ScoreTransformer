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
    def generate(self, x, y=None, max_length=512, temperature=1.0, top_k=1, kv_cache=False) -> dict[str, torch.Tensor]:
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
            "accidental": torch.zeros((B, 1, conf.out_accidental_vocab_size), device=device),
            "keysignature": torch.zeros((B, 1, conf.out_keysignature_vocab_size), device=device),
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
        if "encoder" in self.hyperparameters["components"]:
            encoder_hidden_states = self.forward_enc(
                x, attention_mask=x["pad"]
            )  # (B, T, D)
            encoder_attention_mask = x["pad"]
        else:
            encoder_hidden_states = None
            encoder_attention_mask = None
        if y is None:
            y = y_start_token
            past_key_values = None
        else:
            y = {k: torch.cat([y_start_token[k], y[k]], dim=1) for k in y.keys()}
            # Have to populate KV-cache
            past_key_values = self.forward_dec(
                input_streams={k: torch.roll(v[:, :-1], -1, 1) for k, v in y.items()},
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=None,
                use_cache=True
            )[1]
        for _ in range(max_length + 1 - y["pad"].shape[1]):
            if kv_cache:
                y_pred, past_key_values = self.forward_dec(
                    input_streams={k: v[:, -1:] for k, v in y.items()},
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            else:
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
                if k == "downbeat" and y["offset"].shape[1] > 1:
                    is_downbeat = y_pred["offset"][:, -1].argmax(-1) < y["offset"][:, -2].argmax(-1)
                    logits[is_downbeat, 0] = -float("Inf")

                if k == "accidental":
                    # ensure that we only sample valid accidentals
                    {
                        0: 'double-flat',
                        1: 'flat',
                        2: 'natural',
                        3: 'sharp',
                        4: 'double-sharp',
                    }
                    never_allowed = [0, 4, 6]
                    impossible_accidentals = {
                        0:  [1, 4],
                        1:  [0, 2, 5],
                        2:  [1, 3],
                        3:  [2, 4, 5],
                        4:  [0, 3],
                        5:  [1, 4],
                        6:  [0, 2, 5],
                        7:  [1, 3],
                        8:  [0, 2, 4, 5],
                        9:  [1, 3],
                        10: [2, 4, 5],
                        11: [0, 3]
                    }
                    for i in range(logits.shape[0]):
                        predicted_pitch = y["pitch"][i, -1].argmax()
                        options = impossible_accidentals[predicted_pitch.item() % 12] + never_allowed
                        logits[i, options] = float("-inf")
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
                if k == "pad":  # special case + NO ARGMAX sampling
                    next_token = probs.argmax(-1, keepdim=True)
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