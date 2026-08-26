import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from modules.embeddings import TransformerEmbedding


class Seq2SeqTransformer(nn.Module):
    """
    End-to-End Modern Seq2Seq Transformer.
    Integrates Byte-Level Token Embeddings, Pre-RMSNorm, SwiGLU FFNs,
    RoPE/GQA Attention, and Auto-Masking pipelines.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_embed = TransformerEmbedding(
            src_vocab_size, d_model=d_model, dropout=dropout
        )
        self.tgt_embed = TransformerEmbedding(
            tgt_vocab_size, d_model=d_model, dropout=dropout
        )

        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), device=tgt.device)
        ).bool()
        return tgt_pad_mask & causal_mask

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.src_embed(src), mask=src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        return self.decoder(
            self.tgt_embed(tgt), memory, src_mask=src_mask, tgt_mask=tgt_mask
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        memory = self.encode(src, src_mask)
        dec_out, _ = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(dec_out)
