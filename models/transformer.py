import torch
import torch.nn as nn
from modules.embeddings import TransformerEmbedding
from models.encoder import Encoder
from models.decoder import Decoder


class Seq2SeqTransformer(nn.Module):
    """
    Complete End-to-End Sequence-to-Sequence Transformer architecture
    from 'Attention Is All You Need' (Vaswani et al., 2017).
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
        max_len: int = 5000,
        share_tgt_embeddings: bool = True,
    ):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        # 1. Source and Target Embedding Modules
        self.src_embed = TransformerEmbedding(
            vocab_size=src_vocab_size, d_model=d_model, max_len=max_len, dropout=dropout
        )
        self.tgt_embed = TransformerEmbedding(
            vocab_size=tgt_vocab_size, d_model=d_model, max_len=max_len, dropout=dropout
        )

        # 2. Encoder and Decoder Stacks
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

        # 3. Final Linear Projection Head to Target Vocabulary
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Optional: Weight tying between target embedding matrix and linear generator
        if share_tgt_embeddings:
            self.generator.weight = self.tgt_embed.lut.weight

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier/Glorot uniform parameter initialization across linear layers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Creates padding mask for source sequence.

        Args:
            src: Tensor of token IDs, shape (batch_size, src_len)
        Returns:
            src_mask: Boolean tensor, shape (batch_size, 1, 1, src_len)
        """
        # True for valid tokens, False for <pad> tokens
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Combines target padding mask with causal look-ahead lower-triangular mask.

        Args:
            tgt: Tensor of token IDs, shape (batch_size, tgt_len)
        Returns:
            tgt_mask: Boolean tensor, shape (batch_size, 1, tgt_len, tgt_len)
        """
        batch_size, tgt_len = tgt.shape

        # 1. Target Padding Mask: Shape (batch_size, 1, 1, tgt_len)
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)

        # 2. Causal Subsequence Mask: Shape (1, 1, tgt_len, tgt_len)
        # Prevents position i from attending to positions > i
        tgt_causal_mask = (
            torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device))
            .bool()
            .unsqueeze(0)
            .unsqueeze(1)
        )

        # Combine via logical AND
        tgt_mask = tgt_pad_mask & tgt_causal_mask
        return tgt_mask

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Passes source sequence through embedding and encoder stack."""
        src_embedded = self.src_embed(src)
        return self.encoder(src_embedded, mask=src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Passes target sequence through embedding and decoder stack."""
        tgt_embedded = self.tgt_embed(tgt)
        return self.decoder(
            tgt_embedded, enc_output=enc_output, src_mask=src_mask, tgt_mask=tgt_mask
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Full Forward Pass for Training.

        Args:
            src: Source sequence token IDs, shape (batch_size, src_len)
            tgt: Target sequence token IDs (shifted right), shape (batch_size, tgt_len)

        Returns:
            logits: Logits over target vocabulary, shape (batch_size, tgt_len, tgt_vocab_size)
        """
        # Generate dynamic masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # Encode & Decode
        enc_output = self.encode(src, src_mask)
        dec_output, _ = self.decode(tgt, enc_output, src_mask, tgt_mask)

        # Project output representations to target vocabulary dimension
        logits = self.generator(dec_output)
        return logits


if __name__ == "__main__":
    # Test Parameters
    batch_size = 2
    src_len, tgt_len = 10, 8
    src_vocab_size, tgt_vocab_size = 1000, 1200
    src_pad_idx, tgt_pad_idx = 0, 0

    # Mock Inputs with padding tokens (represented by 0)
    src_tokens = torch.tensor(
        [[12, 45, 98, 23, 88, 1, 0, 0, 0, 0], [5, 67, 891, 23, 4, 12, 90, 3, 1, 0]]
    )
    tgt_tokens = torch.tensor(
        [[1, 55, 30, 99, 12, 2, 0, 0], [1, 9, 88, 432, 10, 12, 4, 2]]
    )

    # Model Instantiation
    model = Seq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
    )

    # Execution Forward Pass
    logits = model(src_tokens, tgt_tokens)

    # Verification assertions
    assert logits.shape == (
        batch_size,
        tgt_len,
        tgt_vocab_size,
    ), f"Expected logits shape {(batch_size, tgt_len, tgt_vocab_size)}, got {logits.shape}"

    print(" Complete Seq2SeqTransformer constructed and executed successfully!")
    print(f" Output Logits Tensor Shape: {logits.shape}")
