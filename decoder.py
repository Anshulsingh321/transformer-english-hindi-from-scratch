import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.encoder import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # 1 Masked self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # 2 Encoder–decoder attention
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)

        # 3 Feed-forward network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 1 Masked self-attention
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 2 Encoder–decoder attention
        attn_output, _ = self.enc_dec_attention(
            x, enc_output, enc_output, src_mask
        )
        x = self.norm2(x + self.dropout2(attn_output))

        # 3 Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x
