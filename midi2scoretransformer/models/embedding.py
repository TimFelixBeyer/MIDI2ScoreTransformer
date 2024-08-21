"""Embedding modules for MIDI and MXL compound token data.
Each token stream is embedded into fixed-size embeddings, which are then summed.
"""
import math

import torch
import torch.nn as nn


class MIDIEmbeddings(nn.Module):
    """Construct embeddings given 5 one-hot input token streams."""
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.ParameterDict({
            "onset": nn.Linear(config.in_onset_vocab_size, config.embedding_size, bias=config.bias),
            "duration": nn.Linear(config.in_duration_vocab_size, config.embedding_size, bias=config.bias),
            "pitch": nn.Linear(config.in_pitch_vocab_size, config.embedding_size, bias=config.bias),
            "velocity": nn.Linear(config.in_velocity_vocab_size, config.embedding_size, bias=config.bias),
            "unconditional": nn.Linear(1, config.embedding_size, bias=False),
        })
        self.layer_norm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps, bias=config.bias)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, input_streams):
        """Embeds MIDI input token streams into a fixed-size embedding.

        Parameters
        ----------
        input_streams : Dict[str, torch.Tensor]
            List of tensors of shape (n_notes, N)

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_notes, config.embedding_size)
        """
        if self.config.is_autoregressive:
            shifted_input_streams = {k: torch.roll(v, 1, 1) for k, v in input_streams.items()}
            for k in shifted_input_streams.keys():
                shifted_input_streams[k][:, 0] = 0
            input_streams = shifted_input_streams
        input_embeds = {
            k: v(input_streams[k]) for k, v in self.embeddings.items() if k in input_streams
        }
        embeddings = sum(input_embeds.values())
        embeddings = self.layer_norm(embeddings)  # Layernorm tested helpful #1188
        embeddings = self.dropout(embeddings)
        return embeddings


class MIDIUnembeddings(nn.Module):
    """Project embeddings to compound tokens for MIDI data.
    Not required for the final model, only here for completeness.
    """

    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.ParameterDict({
            "onset": nn.Linear(config.embedding_size, config.in_onset_vocab_size),
            "duration": nn.Linear(config.embedding_size, config.in_duration_vocab_size),
            "pitch": nn.Linear(config.embedding_size, config.in_pitch_vocab_size),
            "velocity": nn.Linear(config.embedding_size, config.in_velocity_vocab_size),
        })
        self.mask_embeddings = nn.Linear(config.embedding_size, 1)
        self.layer_norm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, embeddings):
        """Projects fixed-size embeddings into four parallel token streams representing
        MIDI data.

        Parameters
        ----------
        embeddings : List[tensor]
            List of tensors of shape (n_notes, config.embedding_dims)

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_notes, config.embedding_size)
        """
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        output_embeds = {k: v(embeddings) for k, v in self.embeddings.items()}
        output_embeds["pad"] = self.mask_embeddings(embeddings)
        return output_embeds


class MXLEmbeddings(nn.Module):
    """Construct the embeddings from MusicXML token streams."""

    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.ParameterDict({
            "offset": nn.Linear(config.out_offset_vocab_size, config.embedding_size, bias=config.bias),
            "downbeat": nn.Linear(config.out_downbeat_vocab_size, config.embedding_size, bias=config.bias),
            "duration": nn.Linear(config.out_duration_vocab_size, config.embedding_size, bias=config.bias),
            "pitch": nn.Linear(config.out_pitch_vocab_size, config.embedding_size, bias=config.bias),
            "accidental": nn.Linear(config.out_accidental_vocab_size, config.embedding_size, bias=config.bias),
            "keysignature": nn.Linear(config.out_keysignature_vocab_size, config.embedding_size, bias=config.bias),
            # "velocity": nn.Linear(config.out_velocity_vocab_size, config.embedding_size, bias=config.bias),
            "grace": nn.Linear(config.out_grace_vocab_size, config.embedding_size, bias=config.bias),
            "trill": nn.Linear(config.out_trill_vocab_size, config.embedding_size, bias=config.bias),
            "staccato": nn.Linear(config.out_staccato_vocab_size, config.embedding_size, bias=config.bias),
            "voice": nn.Linear(config.out_voice_vocab_size, config.embedding_size, bias=config.bias),
            "stem": nn.Linear(config.out_stem_vocab_size, config.embedding_size, bias=config.bias),
            "hand": nn.Linear(config.out_hand_vocab_size, config.embedding_size, bias=config.bias),
        })
        self.mask_embeddings = nn.Linear(1, config.embedding_size, bias=config.bias)
        self.layer_norm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps, bias=config.bias)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, input_streams):
        """Embeds MXL input token streams into a fixed-size embedding.

        Parameters
        ----------
        input_streams : _type_
            List of (typically one-hot) tensors of shape (n_notes, N, C)

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_notes, config.embedding_size)
        """
        # Shift everything by 1 if the model is autoregressive
        if self.config.is_autoregressive:
            shifted_input_streams = {k: torch.roll(v, 1, 1) for k, v in input_streams.items()}
            for k in shifted_input_streams.keys():
                shifted_input_streams[k][:, 0] = 0
            input_streams = shifted_input_streams
        output_embeds = {
            k: self.embeddings[k](v) for k, v in input_streams.items() if k in self.embeddings
        }
        if "pad" in input_streams:
            output_embeds["pad"] = self.mask_embeddings(input_streams["pad"].float().unsqueeze(2))
        embeddings = sum(output_embeds.values())
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MXLUnembeddings(nn.Module):
    """Project embeddings to compound tokens."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.ParameterDict({
            "offset": nn.Linear(config.embedding_size, config.out_offset_vocab_size, bias=config.bias),
            "downbeat": nn.Linear(config.embedding_size, config.out_downbeat_vocab_size, bias=config.bias),
            "duration": nn.Linear(config.embedding_size, config.out_duration_vocab_size, bias=config.bias),
            "pitch": nn.Linear(config.embedding_size, config.out_pitch_vocab_size, bias=config.bias),
            "accidental": nn.Linear(config.embedding_size, config.out_accidental_vocab_size, bias=config.bias),
            "keysignature": nn.Linear(config.embedding_size, config.out_keysignature_vocab_size, bias=config.bias),
            "velocity": nn.Linear(config.embedding_size, config.out_velocity_vocab_size, bias=config.bias),
            "grace": nn.Linear(config.embedding_size, config.out_grace_vocab_size, bias=config.bias),
            "trill": nn.Linear(config.embedding_size, config.out_trill_vocab_size, bias=config.bias),
            "staccato": nn.Linear(config.embedding_size, config.out_staccato_vocab_size, bias=config.bias),
            "voice": nn.Linear(config.embedding_size, config.out_voice_vocab_size, bias=config.bias),
            "stem": nn.Linear(config.embedding_size, config.out_stem_vocab_size, bias=config.bias),
            "hand": nn.Linear(config.embedding_size, config.out_hand_vocab_size, bias=config.bias),
        })
        self.mask_embeddings = nn.Linear(config.embedding_size, 1, bias=config.bias)

        self.layer_norm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps, bias=config.bias)

    def forward(self, embeddings):
        """Projects fixed-size embedding into multiple data streams representing
        MusicXML data.

        Parameters
        ----------
        embedding : torch.tensor
            List of tensors of shape (B, n_notes, D)

        Returns
        -------
        torch.Tensor
            the fixed-size embedding of the input
        """
        embeddings = self.layer_norm(embeddings)
        output_embeds = {k: v(embeddings) for k, v in self.embeddings.items()}
        output_embeds["pad"] = self.mask_embeddings(embeddings)

        return output_embeds
