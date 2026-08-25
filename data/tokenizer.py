from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def build_and_train_tokenizer(
    corpus_files: list[str], vocab_size: int = 32000, save_path: str = "tokenizer.json"
) -> Tokenizer:
    """
    Trains a Byte-Pair Encoding (BPE) subword tokenizer on raw text corpora.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
    )

    tokenizer.train(files=corpus_files, trainer=trainer)

    # Save trained tokenizer config and vocabulary
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer successfully trained and saved to {save_path}")
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    """Loads a pre-trained tokenizer from disk."""
    return Tokenizer.from_file(path)
