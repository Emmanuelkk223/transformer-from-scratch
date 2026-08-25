import pytest
import torch
from modules.attention import MultiHeadAttention, ScaledDotProductAttention
from modules.embeddings import TransformerEmbedding
from models.transformer import Seq2SeqTransformer


def test_scaled_dot_product_attention():
    batch_size, num_heads, seq_len, d_k = 2, 8, 10, 64
    q = torch.randn(batch_size, num_heads, seq_len, d_k)
    k = torch.randn(batch_size, num_heads, seq_len, d_k)
    v = torch.randn(batch_size, num_heads, seq_len, d_k)

    attn = ScaledDotProductAttention()
    out, weights = attn(q, k, v)

    assert out.shape == (batch_size, num_heads, seq_len, d_k)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_multi_head_attention_masking():
    batch_size, seq_len, d_model, num_heads = 2, 8, 512, 8
    x = torch.randn(batch_size, seq_len, d_model)

    # Causal lower-triangular mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).unsqueeze(1)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    out, weights = mha(q=x, k=x, v=x, mask=mask)

    assert out.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_positional_embeddings():
    vocab_size, d_model, seq_len = 1000, 512, 16
    tokens = torch.randint(0, vocab_size, (2, seq_len))

    emb = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model)
    out = emb(tokens)

    assert out.shape == (2, seq_len, d_model)


def test_transformer_full_forward():
    src = torch.randint(1, 500, (2, 10))
    tgt = torch.randint(1, 500, (2, 8))

    model = Seq2SeqTransformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        src_pad_idx=0,
        tgt_pad_idx=0,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
    )

    logits = model(src, tgt)
    assert logits.shape == (2, 8, 1000)
