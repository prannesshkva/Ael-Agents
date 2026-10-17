import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, Any, Optional, List

class CortexVisualizer:

    @staticmethod
    def plot_cortex_diagnostics(
        state: Dict[str, Any],
        param_rows: List[Dict[str, float]],
        title: str = "11-Domain Trainable Jacobian-RSA Brain Architecture (DMN -> SN -> CEN)",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        domain_names = [r["domain"] for r in param_rows]
        short_labels = [f"w{i+1}:{n[:7]}" for i, n in enumerate(domain_names)]

        rdm = state.get("rdm_matrix")
        gram = state.get("jacobian_gram_matrix")
        wm_domains = set(state.get("working_memory_domains", []))

        fig, axes = plt.subplots(1, 3, figsize=(21, 6.2))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        ax1 = axes[0]
        if rdm is not None:
            rdm_np = rdm.detach().cpu().numpy() if isinstance(rdm, torch.Tensor) else np.array(rdm)
            im1 = ax1.imshow(rdm_np, cmap="viridis")
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="RDM Dissimilarity (1 - Cosine)")
            ax1.set_xticks(range(len(short_labels)))
            ax1.set_yticks(range(len(short_labels)))
            ax1.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
            ax1.set_yticklabels(short_labels, fontsize=8)
            ax1.set_title("1. DMN: 11x11 External RSA Matrix (RDM)", fontsize=11, fontweight="bold")

        ax2 = axes[1]
        if gram is not None:
            gram_np = gram.detach().cpu().numpy() if isinstance(gram, torch.Tensor) else np.array(gram)
            im2 = ax2.imshow(gram_np, cmap="magma")
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label=r"$G_{i,j} = (\partial y/\partial w_i)^\top (\partial y/\partial w_j)$")
            ax2.set_xticks(range(len(short_labels)))
            ax2.set_yticks(range(len(short_labels)))
            ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
            ax2.set_yticklabels(short_labels, fontsize=8)
            ax2.set_title(r"2. Salience Network: 11x11 Jacobian Lens ($J^\top J$)", fontsize=11, fontweight="bold")

        ax3 = axes[2]
        w_vals = [r["value"] for r in param_rows]
        bar_colors = ["#e74c3c" if r["domain"] in wm_domains else "#3498db" for r in param_rows]
        bars = ax3.bar(range(len(short_labels)), w_vals, color=bar_colors, edgecolor="black", alpha=0.9)
        ax3.set_xticks(range(len(short_labels)))
        ax3.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("Learned Parameter Value ($w_i$)", fontsize=10)
        ax3.set_title("3. CEN: 11 Trainable Parameters (Red = Active Working Memory)", fontsize=11, fontweight="bold")
        ax3.grid(axis="y", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, w_vals):
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                fontsize=7.5,
                fontweight="bold"
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches="tight")
        return fig
