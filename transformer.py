import torch
import torch.nn as nn

from model.embeddings import TokenEmbedding, PositionalEncoding
from model.encoder import EncoderLayer
from model.decoder import DecoderLayer
from utils.masking import create_padding_mask, create_look_ahead_mask


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout
        )

        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # 1 Create masks
        src_mask = create_padding_mask(src)
        tgt_padding_mask = create_padding_mask(tgt)

        seq_len = tgt.size(1)
        look_ahead_mask = create_look_ahead_mask(seq_len).to(tgt.device)

        # Combine padding + look-ahead masks
        tgt_mask = tgt_padding_mask & look_ahead_mask.unsqueeze(0).unsqueeze(1)

        # 2 Encoder
        enc_output = self.encoder(src, src_mask)

        # 3 Decoder
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        # 4 Output projection
        logits = self.output_linear(dec_output)

        return logits
