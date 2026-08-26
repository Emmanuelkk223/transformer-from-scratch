import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.transformer import Seq2SeqTransformer

os.makedirs("checkpoints", exist_ok=True)

# 1. Clean Parallel Dataset
sentences = [
    ("sorry", "je suis désolé ."),
    ("i am sorry", "je suis désolé ."),
    ("emmanuel is amazing", "emmanuel est incroyable ."),
    ("the cat jumped over the wall .", "le chat a sauté par-dessus le mur ."),
    (
        "python is an amazing programming language .",
        "python est un langage de programmation incroyable .",
    ),
    (
        "artificial intelligence is changing the world .",
        "l'intelligence artificielle change le monde .",
    ),
    (
        "machine learning is transforming software engineering .",
        "l'apprentissage automatique transforme l'ingénierie logicielle .",
    ),
    ("hello my friend .", "bonjour mon ami ."),
    ("good morning everyone .", "bonjour à tous ."),
    (
        "the quick brown fox jumps over the lazy dog .",
        "le rapide renard brun saute par-dessus le chien paresseux .",
    ),
] * 30

src_texts = [p[0] for p in sentences]
tgt_texts = [p[1] for p in sentences]


# 2. Build Lowercased WordPiece Tokenizer (Zero [UNK] pollution)
def build_tokenizer(texts, path):
    tok = Tokenizer(WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    )
    tok.train_from_iterator(texts, trainer)
    tok.save(path)
    return tok


src_tok = build_tokenizer(src_texts, "checkpoints/src_tok.json")
tgt_tok = build_tokenizer(tgt_texts, "checkpoints/tgt_tok.json")


# 3. Dataset Pipeline
class NMTDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_tok, tgt_tok, max_len=32):
        self.samples = []
        pad_id, sos_id, eos_id = (
            src_tok.token_to_id("[PAD]"),
            src_tok.token_to_id("[SOS]"),
            src_tok.token_to_id("[EOS]"),
        )

        for s, t in zip(src_texts, tgt_texts):
            s_ids = [sos_id] + src_tok.encode(s).ids + [eos_id]
            t_ids = [sos_id] + tgt_tok.encode(t).ids + [eos_id]

            s_ids += [pad_id] * (max_len - len(s_ids))
            t_ids += [pad_id] * (max_len - len(t_ids))

            self.samples.append(
                (
                    torch.tensor(s_ids[:max_len]),
                    torch.tensor(t_ids[:max_len][:-1]),
                    torch.tensor(t_ids[:max_len][1:]),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "src": self.samples[idx][0],
            "tgt": self.samples[idx][1],
            "tgt_y": self.samples[idx][2],
        }


loader = DataLoader(
    NMTDataset(src_texts, tgt_texts, src_tok, tgt_tok), batch_size=16, shuffle=True
)

# 4. Train Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Seq2SeqTransformer(
    src_vocab_size=src_tok.get_vocab_size(),
    tgt_vocab_size=tgt_tok.get_vocab_size(),
    src_pad_idx=src_tok.token_to_id("[PAD]"),
    tgt_pad_idx=tgt_tok.token_to_id("[PAD]"),
    d_model=256,
    num_heads=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    d_ff=512,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.token_to_id("[PAD]"))

model.train()
for epoch in range(1, 61):
    total_loss = 0
    for b in loader:
        src, tgt, tgt_y = (
            b["src"].to(device),
            b["tgt"].to(device),
            b["tgt_y"].to(device),
        )
        optimizer.zero_grad()
        logits = model(src, tgt)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:02d}/60 | Loss: {total_loss/len(loader):.4f}")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": {
            "d_model": 256,
            "num_heads": 4,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "d_ff": 512,
        },
    },
    "checkpoints/best_model.pt",
)
print("✅ Fixed Model Checkpoint Saved!")
