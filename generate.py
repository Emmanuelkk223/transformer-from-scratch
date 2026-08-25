import os
import torch
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from models.transformer import Seq2SeqTransformer

# 1. Create dummy directory
os.makedirs("checkpoints", exist_ok=True)


# 2. Build dummy tokenizers
def make_dummy_tok(path):
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"])
    tok.train_from_iterator(["The black cat sat on the mat."], trainer)
    tok.save(path)


make_dummy_tok("checkpoints/src_tok.json")
make_dummy_tok("checkpoints/tgt_tok.json")

# 3. Save a dummy model checkpoint
model = Seq2SeqTransformer(
    src_vocab_size=100, tgt_vocab_size=100, src_pad_idx=0, tgt_pad_idx=0
)
torch.save({"model_state_dict": model.state_dict()}, "checkpoints/best_model.pt")
print("Mock checkpoint created in checkpoints/!")
