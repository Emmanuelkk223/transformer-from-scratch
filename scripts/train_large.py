import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.transformer import Seq2SeqTransformer
from utils.scheduler import NoamLR
from utils.decoding import greedy_decode

os.makedirs("checkpoints", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Fetch Parallel Dataset
print("📥 Fetching English-French Dataset...")
try:
    ds = load_dataset("tatoeba", lang1="en", lang2="fr", split="train[:50000]")
    train_src = [
        item["translation"]["en"]
        for item in ds
        if item["translation"]["en"] and item["translation"]["fr"]
    ]
    train_tgt = [
        item["translation"]["fr"]
        for item in ds
        if item["translation"]["en"] and item["translation"]["fr"]
    ]
except Exception:
    ds = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train[:50000]")
    train_src = [item["translation"]["en"] for item in ds]
    train_tgt = [item["translation"]["fr"] for item in ds]

print(f"Dataset Loaded: {len(train_src):,} sentence pairs.")


# 2. Robust Lowercased WordPiece Tokenizers (Zero prefix-space mismatches)
def build_wordpiece(texts, save_path, vocab_size=12000):
    tok = Tokenizer(WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
    )
    tok.train_from_iterator(texts, trainer)
    tok.save(save_path)
    return tok


print("🔤 Training WordPiece Vocabularies...")
src_tok = build_wordpiece(train_src, "checkpoints/src_tok.json", vocab_size=12000)
tgt_tok = build_wordpiece(train_tgt, "checkpoints/tgt_tok.json", vocab_size=12000)

pad_id = src_tok.token_to_id("[PAD]")
sos_id = src_tok.token_to_id("[SOS]")
eos_id = src_tok.token_to_id("[EOS]")


# 3. Dataset Pipeline
class NMTDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_tok, tgt_tok, max_len=40):
        self.samples = []
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
    NMTDataset(train_src, train_tgt, src_tok, tgt_tok),
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True if device.type == "cuda" else False,
)

# 4. Instantiate Fixed Transformer Architecture
model = Seq2SeqTransformer(
    src_vocab_size=src_tok.get_vocab_size(),
    tgt_vocab_size=tgt_tok.get_vocab_size(),
    src_pad_idx=pad_id,
    tgt_pad_idx=pad_id,
    d_model=256,
    num_heads=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    d_ff=1024,
    dropout=0.1,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
scheduler = NoamLR(optimizer, d_model=256, warmup_steps=4000)
criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

print(
    f"🚀 Training Transformer Model ({sum(p.numel() for p in model.parameters()):,} parameters) on {device}..."
)

# Sample validation check
eval_sentences = ["Tom is happy.", "The cat is black.", "I love programming."]

epochs = 25
best_loss = float("inf")

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    start_t = time.time()

    for batch in train_loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_y = batch["tgt_y"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(src, tgt)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    elapsed = time.time() - start_t
    current_lr = scheduler.get_lr()[0]
    print(
        f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s"
    )

    # In-line Validation Check Every 5 Epochs
    if epoch % 5 == 0 or epoch == epochs:
        model.eval()
        print("\n--- 🔍 VALIDATION TRANSLATION CHECK ---")
        for test_sentence in eval_sentences:
            s_ids = [sos_id] + src_tok.encode(test_sentence).ids + [eos_id]
            s_tensor = torch.tensor(s_ids, dtype=torch.long).unsqueeze(0).to(device)
            s_mask = model.make_src_mask(s_tensor)
            with torch.no_grad():
                res = greedy_decode(
                    model, s_tensor, s_mask, max_len=30, sos_idx=sos_id, eos_idx=eos_id
                )
            print(f"EN: '{test_sentence}' ➔ FR: '{tgt_tok.decode(res.cpu().tolist())}'")
        print("---------------------------------------\n")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "d_model": 256,
                    "num_heads": 8,
                    "num_encoder_layers": 4,
                    "num_decoder_layers": 4,
                    "d_ff": 1024,
                },
            },
            "checkpoints/best_model.pt",
        )

print("✅ Training Complete!")
