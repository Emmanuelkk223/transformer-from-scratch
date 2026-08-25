import torch
from torch.utils.data import Dataset, DataLoader
from data.tokenizer import Tokenizer
from data.tokenizer import build_and_train_tokenizer


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for Transformer Sequence-to-Sequence training.
    """

    def __init__(
        self,
        src_texts: list[str],
        tgt_texts: list[str],
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        max_len: int = 128,
    ):
        assert len(src_texts) == len(
            tgt_texts
        ), "Source and Target sentence counts must match."
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

        # Extract special token IDs
        self.src_pad_id = src_tokenizer.token_to_id("[PAD]")
        self.tgt_pad_id = tgt_tokenizer.token_to_id("[PAD]")
        self.sos_id = tgt_tokenizer.token_to_id("[SOS]")
        self.eos_id = tgt_tokenizer.token_to_id("[EOS]")

    def __len__(self) -> int:
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # Tokenize sentences
        src_ids = self.src_tokenizer.encode(src_text).ids
        tgt_ids = self.tgt_tokenizer.encode(tgt_text).ids

        # Truncate to max length minus special tokens buffer
        src_ids = src_ids[: self.max_len - 2]
        tgt_ids = tgt_ids[: self.max_len - 1]

        # Construct Source sequence: [SOS] + src_ids + [EOS]
        src_tokens = torch.tensor(
            [self.src_tokenizer.token_to_id("[SOS]")]
            + src_ids
            + [self.src_tokenizer.token_to_id("[EOS]")],
            dtype=torch.long,
        )

        # Construct Target Input: [SOS] + tgt_ids
        tgt_input = torch.tensor([self.sos_id] + tgt_ids, dtype=torch.long)

        # Construct Target Label: tgt_ids + [EOS]
        tgt_label = torch.tensor(tgt_ids + [self.eos_id], dtype=torch.long)

        return {
            "src": src_tokens,
            "tgt": tgt_input,
            "tgt_y": tgt_label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def collate_fn(
    batch: list[dict[str, torch.Tensor]], src_pad_id: int = 0, tgt_pad_id: int = 0
) -> dict[str, torch.Tensor]:
    """
    Dynamic padding collate function to pad batches to maximum sentence length per batch.
    """
    src_list = [item["src"] for item in batch]
    tgt_list = [item["tgt"] for item in batch]
    tgt_y_list = [item["tgt_y"] for item in batch]

    # Dynamically pad sequences to batch max length
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_list, batch_first=True, padding_value=src_pad_id
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_list, batch_first=True, padding_value=tgt_pad_id
    )
    tgt_y_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_y_list, batch_first=True, padding_value=tgt_pad_id
    )

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "tgt_y": tgt_y_padded,
    }


if __name__ == "__main__":
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    # 1. Mock Sentences (English -> German toy pair)
    src_sentences = [
        "The cat sat on the mat.",
        "Artificial intelligence is changing the world.",
    ]
    tgt_sentences = [
        "Die Katze saß auf der Matte.",
        "Künstliche Intelligenz verändert die Welt.",
    ]

    # 2. Write temp corpora file for tokenizer training
    with open("temp_src.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(src_sentences))
    with open("temp_tgt.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(tgt_sentences))

    # 3. Train Tokenizers
    src_tokenizer = build_and_train_tokenizer(
        ["temp_src.txt"], vocab_size=1000, save_path="checkpoints/src_tok.json"
    )
    tgt_tokenizer = build_and_train_tokenizer(
        ["temp_tgt.txt"], vocab_size=1000, save_path="checkpoints/tgt_tok.json"
    )

    # 4. Instantiate Dataset & DataLoader
    dataset = TranslationDataset(
        src_texts=src_sentences,
        tgt_texts=tgt_sentences,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_len=30,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=lambda b: collate_fn(b, src_pad_id=0, tgt_pad_id=0),
    )

    # 5. Verify Batch Structure
    for batch in loader:
        print("✅ Batch retrieved successfully!")
        print(f"Source Tensor Shape: {batch['src'].shape}")
        print(f"Target Input Tensor Shape: {batch['tgt'].shape}")
        print(f"Target Label Tensor Shape: {batch['tgt_y'].shape}")
        break
