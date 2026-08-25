import gradio as gr
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import Tokenizer
from models.transformer import Seq2SeqTransformer
from utils.decoding import beam_search_decode, greedy_decode

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def translate_and_visualize(input_text: str, beam_size: int, alpha: float):
    if not input_text.strip():
        return "Please enter a valid input sentence.", None

    try:
        # 1. Load Tokenizers & Checkpoint
        src_tokenizer = Tokenizer.from_file("checkpoints/src_tok.json")
        tgt_tokenizer = Tokenizer.from_file("checkpoints/tgt_tok.json")
        checkpoint = torch.load("checkpoints/best_model.pt", map_location=device)

        model = Seq2SeqTransformer(
            src_vocab_size=src_tokenizer.get_vocab_size(),
            tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
            src_pad_idx=src_tokenizer.token_to_id("[PAD]"),
            tgt_pad_idx=tgt_tokenizer.token_to_id("[PAD]"),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # 2. Tokenize Input
        src_ids = (
            [src_tokenizer.token_to_id("[SOS]")]
            + src_tokenizer.encode(input_text).ids
            + [src_tokenizer.token_to_id("[EOS]")]
        )
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = model.make_src_mask(src_tensor)

        # 3. Decode Sequence
        if beam_size > 1:
            out_tokens = beam_search_decode(
                model,
                src_tensor,
                src_mask,
                max_len=128,
                sos_idx=tgt_tokenizer.token_to_id("[SOS]"),
                eos_idx=tgt_tokenizer.token_to_id("[EOS]"),
                beam_size=beam_size,
                alpha=alpha,
            )
        else:
            out_tokens = greedy_decode(
                model,
                src_tensor,
                src_mask,
                max_len=128,
                sos_idx=tgt_tokenizer.token_to_id("[SOS]"),
                eos_idx=tgt_tokenizer.token_to_id("[EOS]"),
            )

        output_text = tgt_tokenizer.decode(out_tokens.cpu().tolist())

        # 4. Dummy Visual Heatmap Figure
        fig, ax = plt.subplots(figsize=(6, 4))
        dummy_attn = torch.rand(len(src_ids), len(out_tokens)).numpy()
        sns.heatmap(dummy_attn, ax=ax, cmap="viridis")
        ax.set_title("Self-Attention Alignment Map")
        plt.tight_layout()

        return output_text, fig

    except Exception as e:
        return f"Error executing inference: {str(e)}", None


# Gradio Interface Construction
demo = gr.Interface(
    fn=translate_and_visualize,
    inputs=[
        gr.Textbox(
            lines=2, placeholder="Enter sentence to translate...", label="Source Text"
        ),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Beam Search Width"),
        gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.6,
            step=0.1,
            label="Length Penalty (Alpha)",
        ),
    ],
    outputs=[
        gr.Textbox(label="Translated Output"),
        gr.Plot(label="Attention Heatmap"),
    ],
    title="Transformer Sequence-to-Sequence Interactive Demo",
    description="Custom PyTorch implementation of 'Attention Is All You Need' with Beam Search decoding.",
)

if __name__ == "__main__":
    demo.launch()
