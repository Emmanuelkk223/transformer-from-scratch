import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_attention_maps(
    attention_weights: torch.Tensor,
    src_tokens: list[str],
    tgt_tokens: list[str],
    save_path: str = "attention_map.png",
):
    """
    Plots multi-head attention weight heatmaps across all heads.

    Args:
        attention_weights: Tensor of shape (num_heads, tgt_len, src_len)
        src_tokens: List of source sentence token strings
        tgt_tokens: List of target sentence token strings
    """
    num_heads = attention_weights.size(0)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for h in range(num_heads):
        ax = axes[h]
        attn_map = attention_weights[h].detach().cpu().numpy()

        sns.heatmap(
            attn_map,
            ax=ax,
            cmap="viridis",
            xticklabels=src_tokens,
            yticklabels=tgt_tokens,
            cbar=False,
        )
        ax.set_title(f"Head {h + 1}")
        ax.set_xlabel("Source Tokens")
        ax.set_ylabel("Target Tokens")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" Attention heatmap saved to {save_path}")
