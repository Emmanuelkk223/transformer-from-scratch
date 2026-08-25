import torch
import torch.nn as nn
from modules.embeddings import TransformerEmbedding
from modules.feed_forward import PositionwiseFeedForward


class LayerNorm(nn.Module):
    """
    Custom Layer Normalization with learnable scale (gamma) and shift (beta) parameters.
    """

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


if __name__ == "__main__":
    batch_size, seq_len, vocab_size, d_model = 2, 10, 1000, 512
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Test Embeddings
    embed = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)
    x = embed(tokens)
    assert x.shape == (batch_size, seq_len, d_model)

    # Test Feed-Forward
    ffn = PositionwiseFeedForward(d_model=d_model)
    ffn_out = ffn(x)
    assert ffn_out.shape == (batch_size, seq_len, d_model)

    # Test LayerNorm
    ln = LayerNorm(features=d_model)
    norm_out = ln(ffn_out)
    assert norm_out.shape == (batch_size, seq_len, d_model)

    print(" All foundation modules (Embeddings, FFN, LayerNorm) passed assertions!")
