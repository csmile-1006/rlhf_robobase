# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention:
        See TransformerDecoderLayer + TransformerEncoderLayer
    * extra LN at the end of encoder is removed: See Transformer
    * decoder returns a stack of activations from all decoding layers:
        See TransformerDecoder
"""

from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.nn.modules.transformer import _get_clones
from torch import nn, Tensor
import torch.nn.functional as F
from robobase.models.core import get_activation_fn_from_str

from robobase.utils import pref_accuracy
from robobase.models.fusion import FusionModule
from robobase.models.multi_view_transformer import get_sinusoid_encoding_table


class CausalTransformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        norm_first=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, norm_first
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        mask,
        pos_embed,
    ):
        assert len(src.shape) == 3
        # flatten NxHWxC to HWxNxC
        bs, hw, c = src.shape
        src = src.permute(1, 0, 2)
        pos_embed = pos_embed.repeat(1, bs, 1)

        output = self.decoder(
            src,
            tgt_mask=mask,
            pos=pos_embed,
        )
        return output.transpose(0, 1)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                pos=pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        norm_first: bool = False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn_from_str(activation)()
        self.norm_first = norm_first

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                tgt_mask,
                tgt_key_padding_mask,
                pos,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, pos))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        pos,
    ) -> Tensor:
        q = k = self.with_pos_embed(x, pos)
        # NOTE: Order is different in original implementation x, x, x
        x = self.self_attn(
            q,
            k,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=True,
        )[0]
        return self.dropout1(x)

    # feedforward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class MultiViewTransformerDecoderPT(FusionModule):
    """
    Multi-View Transformer Decoder for ACT model.

    Args:
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        nheads (int): Number of attention heads.
        dim_feedforward (int): Dimension of feedforward network.
        enc_layers (int): Number of encoder layers.
        dec_layers (int): Number of decoder layers.
        pre_norm (bool): Use pre-normalization.
        state_dim (int): Dimension of state.
        action_dim (int): Dimension of action.
        num_queries (int): Number of queries. Equivalent to length of action sequence.
        kl_weight (int): Weight for KL divergence.
        use_lang_cond (bool): Use film for language conditioning

    """

    def __init__(
        self,
        input_shape: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        nheads: int = 8,
        dim_feedforward: int = 3200,
        enc_layers: int = 4,
        dec_layers: int = 1,
        pre_norm: bool = False,
        state_dim: int = 8,
        action_dim: int = 8,
        seq_len: int = 50,
        use_lang_cond: bool = False,
        position_embedding: str = "sine",
        causal_attn: bool = True,
        num_labels: int = 1,
    ):
        super().__init__(input_shape=input_shape)
        self.dec_layers = dec_layers
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.use_lang_cond = use_lang_cond
        self.num_labels = num_labels

        # encoder extra parameters
        self.encoder_rgb_feat_proj = nn.Linear(
            np.prod(input_shape), hidden_dim
        )  # project rgb to embedding
        self.encoder_action_proj = nn.Linear(
            action_dim, hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(
            self.state_dim, hidden_dim
        )  # project qpos to embedding

        if position_embedding == "sine":
            self.register_buffer(
                "pos_table",
                get_sinusoid_encoding_table(self.seq_len, hidden_dim * 3),
            )  # [CLS], qpos, a_seq
        elif position_embedding == "learned":
            self.pos_table = nn.Parameter(torch.randn(self.seq_len, hidden_dim))
        self.position_embedding = position_embedding

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied
            # to the left in the input sequence
            # torch.nn.Transformer uses additive mask
            # as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf
            # and others (including diag) should be 0.
            sz = seq_len
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self.register_buffer("mask", mask)
        else:
            self.mask = None
            self.memory_mask = None

        self.transformer = CausalTransformer(
            d_model=hidden_dim * 3,
            nhead=nheads,
            num_decoder_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=pre_norm,
            return_intermediate_dec=False,
        )
        self.reward_head = nn.Linear(hidden_dim * 3, self.num_labels)
        self.criterion = nn.CrossEntropyLoss()

    @property
    def output_shape(self) -> Tuple[int]:
        return ((self.seq_len, 1),)

    def forward(
        self,
        rgb_feat: torch.Tensor,
        qpos: torch.Tensor,
        actions: torch.Tensor = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the MultiViewTransformerEncoderDecoderACT model.

        Args:
            rgb_feat (torch.Tensor): Tensor containing image features.
            qpos (torch.Tensor): Tensor containing proprioception features.
            actions (torch.Tensor, optional): Tensor containing action sequences.
            attn_mask (torch.Tensor, optional): Tensor containing attention mask.

        Returns:
            reward_hat (torch.Tensor): reward prediction.
        """

        actions = actions[:, : self.seq_len]

        rgb_feat_embed = self.encoder_rgb_feat_proj(rgb_feat)  # (bs, seq, hidden_dim)
        action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
        qpos_embed = self.encoder_joint_proj(qpos)  # (bs, seq, hidden_dim)

        input = torch.cat([rgb_feat_embed, action_embed, qpos_embed], axis=-1)

        # pos_embed = self.position_embedding(rgb_feat_embed)
        if self.position_embedding == "sine":
            # pos_embed = self.position_embedding(rgb_feat_embed)
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)
        elif self.position_embedding == "learnable":
            pos_embed = self.pos_table

        input = torch.cat([rgb_feat_embed, action_embed, qpos_embed], axis=-1)

        # Apply transformer block
        # Change to get the last output after passing through all decoder layer.
        # Fix the bug https://github.com/tonyzhaozh/act/issues/25#issue-2258740521
        hs = self.transformer(
            input, self.mask if attn_mask is None else attn_mask, pos_embed
        )[:, -1]
        reward_hat = self.reward_head(hs)

        return reward_hat

    def calculate_loss(
        self,
        input_feats: Tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, dict]]:
        """
        Calculate the loss for the MultiViewTransformerEncoderDecoderACT model.

        Args:
            input_feats (Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]):
                    Tuple containing action predictions, padding predictions,
                    and a list of latent variables [mu, logvar].
            labels (torch.Tensor): Tensor containing ground truth preference labels.
            is_pad (torch.Tensor): Tensor indicating padding positions.

        Returns:
            Optional[Tuple[torch.Tensor, dict]]:
                    Tuple containing the loss tensor and a dictionary of loss
                    components.
        """
        reward_hat_1, reward_hat_2 = input_feats
        logits = torch.stack([reward_hat_1, reward_hat_2], dim=-1)

        loss_dict = dict()
        reward_loss = 0.0
        for idx, (logit, label) in enumerate(zip(logits.unbind(1), labels.unbind(1))):
            reward_loss = F.cross_entropy(logit, label.long())
            loss_dict[f"pref_acc_label_{idx}"] = pref_accuracy(logit, label)

        loss_dict["loss"] = reward_loss

        return (loss_dict["loss"], loss_dict)
