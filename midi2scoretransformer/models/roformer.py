"""This module contains our RoFormer model implementation. It is mostly a copy of the RoFormer implementation
from HuggingFace Transformers, but contains some modifications:
    - custom embeddings/projections
    - flash attention
    - SwiGLU activation function
    - QKV layer fusion where possible
    - RoPE in cross-attention
    - pre-norm
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RoFormerModel as RoFormerModelBase
from transformers.modeling_outputs import \
    BaseModelOutputWithPastAndCrossAttentions
from transformers.models.roformer.modeling_roformer import (RoFormerAttention,
                                                            RoFormerEncoder,
                                                            RoFormerLayer)
from models.embedding import MIDIEmbeddings, MXLEmbeddings, MXLUnembeddings
from models.model import BaseModel


class CustomRoFormerSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.bias)
            self.key_value = nn.Linear(config.hidden_size, self.all_head_size * 2, bias=config.bias)
        else:
            self.query_key_value = nn.Linear(config.hidden_size, self.all_head_size * 3, bias=config.bias)

        self.dropout = config.attention_probs_dropout_prob
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=config.bias)

        self.is_decoder = config.is_decoder
        self.rotary_value = config.rotary_value

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        hidden_states = self.norm(hidden_states)
        if not self.is_cross_attention:
            q, k, v = self.query_key_value(hidden_states).chunk(3, dim=-1)
            query_layer = self.transpose_for_scores(q)
            key_layer = self.transpose_for_scores(k)
            value_layer = self.transpose_for_scores(v)
        else:
            q = self.query(hidden_states)
            query_layer = self.transpose_for_scores(q)
            if past_key_value is not None:
                # reuse k, v, cross_attentions
                key_layer = past_key_value[0]
                value_layer = past_key_value[1]
            else:
                k, v = self.key_value(encoder_hidden_states).chunk(2, dim=-1)
                key_layer = self.transpose_for_scores(k)
                value_layer = self.transpose_for_scores(v)
            attention_mask = encoder_attention_mask

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if sinusoidal_pos is not None:
            if past_key_value is not None and self.is_cross_attention:
                # the past_key_values have already been rotated
                query_layer, _ = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, query_layer, query_layer
                )
            else:
                if self.rotary_value:
                    query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_layer, key_layer, value_layer
                    )
                else:
                    query_layer, key_layer = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_layer, key_layer
                    )
        if not self.is_cross_attention and past_key_value is not None:
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        if head_mask is None:
            if self.is_decoder and not self.is_cross_attention:
                context_layer = F.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    is_causal=True if past_key_value is None else False,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout if self.training else 0
                )
            else:
                context_layer = F.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout if self.training else 0
                )

        # We sometimes ran into nan's with flash attention, thus a fallback here.
        if head_mask is not None or context_layer.isnan().any():
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in RoFormerModel forward() function)
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = F.dropout(attention_probs, p=self.dropout if self.training else 0)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        q_t = query_layer.size(2)
        query_layer = query_layer * cos_pos[:, :, :q_t] + rotate_half_query_layer * sin_pos[:, :, :q_t]
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        k_t = key_layer.size(2)
        key_layer = key_layer * cos_pos[:, :, :k_t] + rotate_half_key_layer * sin_pos[:, :, :k_t]
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
                value_layer
            )
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer


class CustomRoFormerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor


class CustomRoFormerAttention(RoFormerAttention):
    """Patch in pre-norm layers."""
    def __init__(self, config, is_cross_attention=False):
        super().__init__(config)
        self.self = CustomRoFormerSelfAttention(config, is_cross_attention=is_cross_attention)
        self.output = CustomRoFormerSelfOutput(config)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class CustomRoFormerIntermediate(nn.Module):
    """Add norm to input of intermediate layer."""
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=config.bias)
        self.dense = nn.Linear(config.hidden_size, 2*config.intermediate_size, bias=config.bias)
        self.intermediate_act_fn = SwiGLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class CustomRoFormerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor


class CustomRoFormerLayer(RoFormerLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = CustomRoFormerAttention(config)  # Use your custom attention here
        if self.add_cross_attention:
            self.crossattention = CustomRoFormerAttention(config, is_cross_attention=True)
        self.intermediate = CustomRoFormerIntermediate(config)
        self.output = CustomRoFormerOutput(config)


class CustomRoFormerEncoder(RoFormerEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([CustomRoFormerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # [sequence_length, embed_size_per_head] -> [batch_size, num_heads, sequence_length, embed_size_per_head]
        # We have to ensure that positional embeddings work for both encoder and decoder states
        B, T = hidden_states.shape[:-1]
        T = max(encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0, T)
        sinusoidal_pos = self.embed_positions((B, T,), past_key_values_length)[None, None, :, :]

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    sinusoidal_pos,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    sinusoidal_pos,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class RoFormerModel(RoFormerModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = CustomRoFormerEncoder(config)
        del self.embeddings


class Roformer(BaseModel):
    def __init__(self, enc_configuration=None, dec_configuration=None, hyperparameters=None):
        super().__init__(enc_configuration, dec_configuration, hyperparameters)
        self.encoder = RoFormerModel(enc_configuration)
        self.decoder = RoFormerModel(dec_configuration)
        self.embeddings_enc = MIDIEmbeddings(enc_configuration)
        self.embeddings_dec = MXLEmbeddings(dec_configuration)
        self.unembeddings_dec = MXLUnembeddings(dec_configuration)

        self.norm = nn.LayerNorm(enc_configuration.hidden_size, eps=enc_configuration.layer_norm_eps, bias=enc_configuration.bias)

    def forward_enc(
        self,
        input_streams: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        output_attentions = self.enc_config.output_attentions
        output_hidden_states = self.enc_config.output_hidden_states
        return_dict = self.enc_config.use_return_dict

        sample_stream = input_streams[list(input_streams.keys())[0]]
        B, T = sample_stream.size()[:-1]
        device = sample_stream.device

        if attention_mask is None:
            attention_mask = torch.ones(((B, T)), device=device)
        extended_attention_mask: torch.Tensor = (
            self.encoder.get_extended_attention_mask(attention_mask, (B, T))
        )
        encoder_extended_attention_mask = None

        embedding_output = self.embeddings_enc(input_streams)

        if hasattr(self, "embeddings_project"):
            embedding_output = self.encoder.embeddings_project(embedding_output)

        encoder_outputs = self.encoder.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_attention_mask=encoder_extended_attention_mask,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return self.norm(encoder_outputs.last_hidden_state)

    def forward_dec(
        self,
        input_streams: Dict[str, torch.Tensor],
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[tuple[torch.Tensor]] = None,
        use_cache=False,
    ) -> Dict[str, torch.Tensor]:

        output_attentions = self.dec_config.output_attentions
        output_hidden_states = self.dec_config.output_hidden_states
        return_dict = self.dec_config.use_return_dict

        B, T = input_streams["offset"].size()[:2]
        if past_key_values is not None:
            T += past_key_values[0][0].size(2)
        device = input_streams["offset"].device

        # Make/convert attention masks
        attention_mask = torch.ones(((B, T)), device=device)
        extended_attention_mask: torch.Tensor = (
            self.decoder.get_extended_attention_mask(attention_mask, (B, T))
        )
        if not self.dec_config.is_autoregressive:
            extended_attention_mask = torch.zeros_like(extended_attention_mask)

        assert self.dec_config.is_decoder
        if encoder_hidden_states is not None:
            encoder_hidden_shape = encoder_hidden_states.size()[:2]
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.decoder.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None
        if use_cache:
            extended_attention_mask = extended_attention_mask[..., -input_streams['pad'].size(1):, :]
        # pass to model
        embedding_output = self.embeddings_dec({k: v[:, :T] for k, v in input_streams.items()})

        decoder_outputs = self.decoder.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        out_proj = self.unembeddings_dec(decoder_outputs.last_hidden_state)
        if use_cache:
            return out_proj, decoder_outputs.past_key_values
        else:
            return out_proj
