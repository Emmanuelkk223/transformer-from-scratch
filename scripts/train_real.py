import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.transformer import Seq2SeqTransformer

os.makedirs("checkpoints", exist_ok=True)

# 1. Fetch Everyday Conversational Dataset (Tatoeba EN-FR)
print("📥 Fetching Everyday Conversational Dataset (Tatoeba EN-FR)...")
try:
    raw_ds = load_dataset("tatoeba", lang1="en", lang2="fr", split="train[:15000]")
    src_sentences = [
        item["translation"]["en"]
        for item in raw_ds
        if item["translation"]["en"] and item["translation"]["fr"]
    ]
    tgt_sentences = [
        item["translation"]["fr"]
        for item in raw_ds
        if item["translation"]["en"] and item["translation"]["fr"]
    ]
except Exception:
    # Reliable Fallback dataset if Tatoeba URI shifts
    raw_ds = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train[:10000]")
    src_sentences = [item["translation"]["en"] for item in raw_ds]
    tgt_sentences = [item["translation"]["fr"] for item in raw_ds]


# 2. Build BPE Tokenizer tuned for 4000 vocab (keeps subwords meaningful)
def train_bpe(texts, save_path, vocab_size=4000):
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
    )
    tok.train_from_iterator(texts, trainer)
    tok.save(save_path)
    return tok


print("🔤 Training Subword Tokenizers...")
src_tok = train_bpe(src_texts, "checkpoints/src_tok.json")
tgt_tok = train_bpe(tgt_texts, "checkpoints/tgt_tok.json")


# 3. Dataset Pipeline
class NMTDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_tok, tgt_tok, max_len=32):
        self.samples = []
        pad_id = src_tok.token_to_id("[PAD]")
        sos_id = src_tok.token_to_id("[SOS]")
        eos_id = src_tok.token_to_id("[EOS]")

        for s, t in zip(src_texts, tgt_texts):
            s_ids = [sos_id] + src_tok.encode(s).ids[: max_len - 2] + [eos_id]
            t_ids = [sos_id] + tgt_tok.encode(t).ids[: max_len - 2] + [eos_id]

            s_pad = s_ids + [pad_id] * (max_len - len(s_ids))
            t_pad = t_ids + [pad_id] * (max_len - len(t_ids))

            self.samples.append(
                (
                    torch.tensor(s_pad[:max_len]),
                    torch.tensor(t_pad[:max_len][:-1]),
                    torch.tensor(t_pad[:max_len][1:]),
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


train_loader = DataLoader(
    NMTDataset(src_sentences, tgt_sentences, src_tok, tgt_tok),
    batch_size=64,
    shuffle=True,
)

# 4. Model & Optimizer
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

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(
    ignore_index=tgt_tok.token_to_id("[PAD]"), label_smoothing=0.1
)

print(f"🚀 Training Transformer on {device} (25 Epochs)...")
model.train()
for epoch in range(1, 26):
    total_loss = 0
    for batch in train_loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_y = batch["tgt_y"].to(device)

        optimizer.zero_grad()
        logits = model(src, tgt)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/25 | Loss: {total_loss / len(train_loader):.4f}")

torch.save({"model_state_dict": model.state_dict()}, "checkpoints/best_model.pt")
print("✅ Trained Model & Tokenizers Saved!")
