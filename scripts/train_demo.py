import os
import time
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset, DataLoader

from models.transformer import Seq2SeqTransformer
from utils.scheduler import NoamLR

# 1. Parallel English-French Corpus
parallel_corpus = [
    ("The cat jumped over the wall.", "Le chat a sauté par-dessus le mur."),
    (
        "Artificial intelligence is changing the world.",
        "L'intelligence artificielle change le monde.",
    ),
    (
        "Machine learning is transforming software engineering.",
        "L'apprentissage automatique transforme l'ingénierie logicielle.",
    ),
    ("The black cat sat on the mat.", "Le chat noir s'est assis sur le tapis."),
    ("Hello my friend.", "Bonjour mon ami."),
    ("Good morning everyone.", "Bonjour à tous."),
    (
        "Python is an amazing programming language.",
        "Python est un langage de programmation incroyable.",
    ),
    (
        "We built a seq2seq transformer from scratch.",
        "Nous avons construit un transformateur seq2seq à partir de zéro.",
    ),
    (
        "Deep learning models require data.",
        "Les modèles d'apprentissage profond nécessitent des données.",
    ),
    (
        "The quick brown fox jumps over the lazy dog.",
        "Le rapide renard brun saute par-dessus le chien paresseux.",
    ),
] * 20  # Duplicate pairs to form a training batch

os.makedirs("checkpoints", exist_ok=True)


# 2. Build BPE Subword Tokenizers
def train_tokenizer(texts, save_path):
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"])
    tok.train_from_iterator(texts, trainer)
    tok.save(save_path)
    return tok


src_texts = [pair[0] for pair in parallel_corpus]
tgt_texts = [pair[1] for pair in parallel_corpus]

src_tok = train_tokenizer(src_texts, "checkpoints/src_tok.json")
tgt_tok = train_tokenizer(tgt_texts, "checkpoints/tgt_tok.json")


# 3. PyTorch Parallel Dataset
class ParallelDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_tok, tgt_tok, max_len=32):
        self.samples = []
        for s, t in zip(src_texts, tgt_texts):
            s_ids = (
                [src_tok.token_to_id("[SOS]")]
                + src_tok.encode(s).ids
                + [src_tok.token_to_id("[EOS]")]
            )
            t_ids = (
                [tgt_tok.token_to_id("[SOS]")]
                + tgt_tok.encode(t).ids
                + [tgt_tok.token_to_id("[EOS]")]
            )

            # Pad sequences
            s_ids += [src_tok.token_to_id("[PAD]")] * (max_len - len(s_ids))
            t_ids += [tgt_tok.token_to_id("[PAD]")] * (max_len - len(t_ids))

            self.samples.append(
                (
                    torch.tensor(s_ids[:max_len]),
                    torch.tensor(t_ids[:max_len][:-1]),  # Target Input
                    torch.tensor(t_ids[:max_len][1:]),  # Target Label
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt_in, tgt_out = self.samples[idx]
        return {"src": src, "tgt": tgt_in, "tgt_y": tgt_out}


dataset = ParallelDataset(src_texts, tgt_texts, src_tok, tgt_tok)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 4. Initialize Transformer & Optimization Loop
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

print("Starting Mini-NMT Model Training...")
model.train()
for epoch in range(1, 51):
    total_loss = 0
    for batch in dataloader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_y = batch["tgt_y"].to(device)

        optimizer.zero_grad()
        logits = model(src, tgt)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/50 | Loss: {total_loss/len(dataloader):.4f}")

# Save Checkpoint
torch.save({"model_state_dict": model.state_dict()}, "checkpoints/best_model.pt")
print("✅ Demo Model Trained & Checkpoint Saved to checkpoints/best_model.pt!")
