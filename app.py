import time
import torch
import gradio as gr
import plotly.graph_objects as go
from tokenizers import Tokenizer
from models.transformer import Seq2SeqTransformer
from utils.decoding import beam_search_decode, greedy_decode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def translate_and_visualize(
    input_text: str, beam_size: int, alpha: float, max_len: int
):
    if not input_text.strip():
        return "Please enter a valid text sequence.", "", None

    try:
        # 1. Load Tokenizers
        src_tokenizer = Tokenizer.from_file("checkpoints/src_tok.json")
        tgt_tokenizer = Tokenizer.from_file("checkpoints/tgt_tok.json")

        # Synchronized Special Tokens
        sos_idx = tgt_tokenizer.token_to_id("[SOS]")
        eos_idx = tgt_tokenizer.token_to_id("[EOS]")
        pad_idx = src_tokenizer.token_to_id("[PAD]")

        if sos_idx is None or eos_idx is None or pad_idx is None:
            raise ValueError(
                "Special tokens ([SOS], [EOS], [PAD]) were not found in vocabulary files."
            )

        # 2. Dynamic Model Restoration
        checkpoint = torch.load("checkpoints/best_model.pt", map_location=device)
        cfg = checkpoint.get(
            "config",
            {
                "d_model": 256,
                "num_heads": 8,
                "num_encoder_layers": 4,
                "num_decoder_layers": 4,
                "d_ff": 1024,
            },
        )

        model = Seq2SeqTransformer(
            src_vocab_size=src_tokenizer.get_vocab_size(),
            tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
            src_pad_idx=pad_idx,
            tgt_pad_idx=pad_idx,
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            num_encoder_layers=cfg["num_encoder_layers"],
            num_decoder_layers=cfg["num_decoder_layers"],
            d_ff=cfg["d_ff"],
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # 3. Process Input Sequence
        encoded_ids = src_tokenizer.encode(input_text.lower()).ids
        src_ids = [sos_idx] + encoded_ids + [eos_idx]
        src_raw_tokens = [src_tokenizer.id_to_token(i) or f"[{i}]" for i in src_ids]

        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = model.make_src_mask(src_tensor)

        # 4. Perform Inference with Telemetry Timing
        start_time = time.time()
        if beam_size > 1:
            out_tokens = beam_search_decode(
                model,
                src_tensor,
                src_mask,
                max_len=max_len,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                beam_size=beam_size,
                alpha=alpha,
            )
        else:
            out_tokens = greedy_decode(
                model,
                src_tensor,
                src_mask,
                max_len=max_len,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
            )
        latency_ms = (time.time() - start_time) * 1000

        # 5. Reconstruct Clean Output Sequence
        raw_list = out_tokens.cpu().tolist()
        clean_list = []
        for t in raw_list:
            clean_list.append(t)
            if t == eos_idx and len(clean_list) > 1:
                break

        output_text = tgt_tokenizer.decode(clean_list).strip()
        tgt_raw_tokens = [tgt_tokenizer.id_to_token(i) or f"[{i}]" for i in clean_list]

        tokens_gen = len(clean_list)
        tok_per_sec = (tokens_gen / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

        metrics_html = f"""
        <div style="display: flex; gap: 15px; background-color: #1e293b; padding: 12px 18px; border-radius: 8px; border: 1px solid #334155; font-family: sans-serif;">
            <div><span style="color: #94a3b8; font-size: 12px;">LATENCY</span><br><strong style="color: #38bdf8; font-size: 15px;">{latency_ms:.1f} ms</strong></div>
            <div style="border-left: 1px solid #475569; padding-left: 15px;"><span style="color: #94a3b8; font-size: 12px;">THROUGHPUT</span><br><strong style="color: #34d399; font-size: 15px;">{tok_per_sec:.1f} tok/s</strong></div>
            <div style="border-left: 1px solid #475569; padding-left: 15px;"><span style="color: #94a3b8; font-size: 12px;">GENERATED LENGTH</span><br><strong style="color: #f43f5e; font-size: 15px;">{tokens_gen} tokens</strong></div>
            <div style="border-left: 1px solid #475569; padding-left: 15px;"><span style="color: #94a3b8; font-size: 12px;">DECODING</span><br><strong style="color: #a78bfa; font-size: 15px;">{'Beam Search (k=' + str(beam_size) + ')' if beam_size > 1 else 'Greedy'}</strong></div>
        </div>
        """

        # 6. Extract Multi-Head Cross-Attention Matrix
        with torch.no_grad():
            memory = model.encode(src_tensor, src_mask)
            tgt_tensor = (
                torch.tensor(clean_list, dtype=torch.long).unsqueeze(0).to(device)
            )
            tgt_mask = model.make_tgt_mask(tgt_tensor)
            _, real_attn = model.decode(tgt_tensor, memory, src_mask, tgt_mask)

        if real_attn is not None and real_attn.dim() == 4:
            full_matrix = real_attn[0].mean(dim=0).cpu().numpy()
        else:
            full_matrix = torch.zeros((len(clean_list), len(src_ids)))

        # Slice out structural tokens ([SOS], [EOS], [PAD]) for clean matrix visual
        valid_src = [
            i
            for i, t in enumerate(src_raw_tokens)
            if t not in ["[SOS]", "[EOS]", "[PAD]", ""]
        ]
        valid_tgt = [
            i
            for i, t in enumerate(tgt_raw_tokens)
            if t not in ["[SOS]", "[EOS]", "[PAD]", ""]
        ]

        vis_src_tokens = [src_raw_tokens[i] for i in valid_src]
        vis_tgt_tokens = [tgt_raw_tokens[i] for i in valid_tgt]

        if len(valid_tgt) > 0 and len(valid_src) > 0:
            vis_matrix = full_matrix[valid_tgt, :][:, valid_src]
        else:
            vis_matrix = full_matrix

        # 7. Render Interactive Plotly Matrix
        fig = go.Figure(
            data=go.Heatmap(
                z=vis_matrix,
                x=vis_src_tokens,
                y=vis_tgt_tokens,
                colorscale="Electric",
                colorbar=dict(
                    title=dict(text="Attention Weight", font=dict(color="#e2e8f0"))
                ),
                text=[[f"{v:.2f}" for v in row] for row in vis_matrix],
                texttemplate="%{text}",
                textfont={"size": 11, "color": "#ffffff"},
                hovertemplate="<b>Source Word:</b> %{x}<br><b>Target Word:</b> %{y}<br><b>Weight:</b> %{z:.4f}<extra></extra>",
            )
        )

        calc_height = max(420, len(vis_tgt_tokens) * 38)

        fig.update_layout(
            title=dict(
                text="Cross-Attention Alignment Matrix (Source ➔ Target Mapping)",
                font=dict(size=14, color="#f8fafc"),
            ),
            xaxis=dict(
                title=dict(
                    text="Source Input Tokens (English)", font=dict(color="#94a3b8")
                ),
                tickangle=-25,
                color="#cbd5e1",
            ),
            yaxis=dict(
                title=dict(
                    text="Generated Target Tokens (French)", font=dict(color="#94a3b8")
                ),
                autorange="reversed",
                color="#cbd5e1",
            ),
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            height=calc_height,
            margin=dict(l=70, r=40, t=60, b=60),
        )

        return output_text, metrics_html, fig

    except Exception as e:
        error_html = f"<div style='color: #ef4444; background: #451a1a; padding: 10px; border-radius: 6px;'>Runtime Error: {str(e)}</div>"
        return "", error_html, None


custom_theme = gr.themes.Soft(
    primary_hue="indigo", secondary_hue="slate", neutral_hue="slate"
).set(
    body_background_fill="#090d16",
    block_background_fill="#0f172a",
    block_border_width="1px",
    block_border_color="#1e293b",
)

with gr.Blocks(theme=custom_theme, title="Transformer NMT Studio") as demo:
    gr.Markdown("""
        # 🌐 Seq2Seq Transformer Neural Translation Studio
        *Custom PyTorch Transformer featuring WordPiece Subwords, RMSNorm, SwiGLU, and Cross-Attention Interpretability*
        """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Generation Control Panel")

            beam_size = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Beam Search Width (k)",
                info="Higher values improve translation quality via broader beam search space.",
            )
            alpha = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.6,
                step=0.1,
                label="Length Penalty (Alpha)",
                info="Adjusts beam scoring bias toward longer sequence outputs.",
            )
            max_len = gr.Slider(
                minimum=10,
                maximum=128,
                value=64,
                step=2,
                label="Max Generation Steps",
                info="Maximum sequence limit for autoregressive decoding.",
            )

            gr.Markdown("---")
            gr.Markdown(
                "**Model Hardware Acceleration:** `" + str(device).upper() + "`"
            )
            submit_btn = gr.Button("⚡ Translate & Generate Matrix", variant="primary")

        with gr.Column(scale=2):
            input_text = gr.Textbox(
                lines=3,
                value="the cat is black.",
                label="Source Input (English)",
                placeholder="Enter sentence to translate...",
            )

            output_text = gr.Textbox(
                label="Translated Output (French)",
                interactive=False,
                placeholder="Translated text will appear here...",
            )

            metrics = gr.HTML(label="Inference Metrics Telemetry")

            heatmap_plot = gr.Plot(label="Cross-Attention Layer Interpretability")

    submit_btn.click(
        fn=translate_and_visualize,
        inputs=[input_text, beam_size, alpha, max_len],
        outputs=[output_text, metrics, heatmap_plot],
    )

if __name__ == "__main__":
    demo.launch()
