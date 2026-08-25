import argparse
import torch
from tokenizers import Tokenizer
from models.transformer import Seq2SeqTransformer
from utils.decoding import beam_search_decode, greedy_decode
from utils.visualization import plot_attention_maps


def evaluate_interactive():
    parser = argparse.ArgumentParser(description="Transformer Inference CLI")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint .pt"
    )
    parser.add_argument(
        "--src_tok", type=str, required=True, help="Path to source tokenizer JSON"
    )
    parser.add_argument(
        "--tgt_tok", type=str, required=True, help="Path to target tokenizer JSON"
    )
    parser.add_argument("--beam_size", type=int, default=5, help="Beam search width")
    parser.add_argument(
        "--visualize", action="store_true", help="Plot attention heatmap"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Tokenizers
    src_tokenizer = Tokenizer.from_file(args.src_tok)
    tgt_tokenizer = Tokenizer.from_file(args.tgt_tok)

    sos_idx = tgt_tokenizer.token_to_id("[SOS]")
    eos_idx = tgt_tokenizer.token_to_id("[EOS]")
    src_pad_idx = src_tokenizer.token_to_id("[PAD]")
    tgt_pad_idx = tgt_tokenizer.token_to_id("[PAD]")

    # Load Model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = Seq2SeqTransformer(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully! Type 'exit' or 'quit' to stop.\n")

    while True:
        src_text = input("Input Text > ").strip()
        if src_text.lower() in ["exit", "quit"]:
            break

        # Tokenize & Tensorize
        src_ids = (
            [src_tokenizer.token_to_id("[SOS]")]
            + src_tokenizer.encode(src_text).ids
            + [src_tokenizer.token_to_id("[EOS]")]
        )
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = model.make_src_mask(src_tensor)

        # Generate Sequence
        if args.beam_size > 1:
            output_tokens = beam_search_decode(
                model,
                src_tensor,
                src_mask,
                max_len=128,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                beam_size=args.beam_size,
            )
        else:
            output_tokens = greedy_decode(
                model,
                src_tensor,
                src_mask,
                max_len=128,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
            )

        output_text = tgt_tokenizer.decode(output_tokens.cpu().tolist())
        print(f"Translation > {output_text}\n")


if __name__ == "__main__":
    evaluate_interactive()
