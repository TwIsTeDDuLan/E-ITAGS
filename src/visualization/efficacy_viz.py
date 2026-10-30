"""Visualization of Continuous Trait-Efficacy Maps, GP Posterior Uncertainty, and Active Learning Curves."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from ..models.task import Task
from ..models.coalition import Coalition
from ..learning.efficacy_map import GroundTruthEfficacyOracle
from ..learning.active_learner import LearnedEfficacyGP


class EfficacyVisualizer:
    """Renders research-grade visualizations for Trait-Efficacy modeling and active learning."""

    STRATEGY_COLORS = {
        "exact": "#10B981",          # Emerald Green (Best)
        "box": "#F59E0B",            # Amber
        "unconstrained": "#EF4444",    # Red
        "random": "#64748B",         # Slate Gray (Baseline)
    }

    STRATEGY_LABELS = {
        "exact": "Exact Realizable (Direct Ω)",
        "box": "Box-Constrained + Projection",
        "unconstrained": "Unconstrained + Projection",
        "random": "Random Sampling (Baseline)",
    }

    def render_learning_curves(
        self,
        strategy_rmses: Dict[str, List[float]],
        task_name: str = "T4: Collaborative Pallet Sort",
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot convergence curves comparing prediction RMSE vs. Query Budget across strategies."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for strat_key, rmses in strategy_rmses.items():
            color = self.STRATEGY_COLORS.get(strat_key, "#3B82F6")
            label = self.STRATEGY_LABELS.get(strat_key, strat_key.title())
            linestyle = "-" if strat_key == "exact" else ("--" if strat_key in ["box", "unconstrained"] else ":")
            linewidth = 2.6 if strat_key == "exact" else 2.0
            marker = "o" if strat_key == "exact" else ("s" if strat_key == "box" else "^")

            iterations = np.arange(len(rmses))
            ax.plot(
                iterations, rmses,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=6,
                alpha=0.9,
            )

        ax.set_title(f"Realizability-Aware Active Learning Efficiency: {task_name}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Number of Active Queries (Simulated Executions)", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel("Trait-Efficacy Prediction Error (RMSE)", fontsize=11, fontweight="bold", labelpad=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9.5, loc="upper right")

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved active learning curves to: {save_path}")

        if show:
            plt.show()

        return fig

    def render_gp_posterior(
        self,
        task: Task,
        candidates: List[Coalition],
        gp_model: LearnedEfficacyGP,
        oracle: GroundTruthEfficacyOracle,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Render GP predicted efficacy mean and +/-2sigma confidence interval against ground truth."""
        fig, ax = plt.subplots(figsize=(12, 6.5))

        coalition_names = [f"C{idx+1}: {c}" for idx, c in enumerate(candidates)]
        indices = np.arange(len(candidates))

        means = []
        stds = []
        trues = []

        for c in candidates:
            m, s = gp_model.predict_with_uncertainty(c.to_vector())
            t = oracle.predict_vector(c.to_vector(), task)
            means.append(m)
            stds.append(s)
            trues.append(t)

        means = np.array(means)
        stds = np.array(stds)
        trues = np.array(trues)

        # Bar chart of GP Posterior Mean with error bars (95% CI: 2 * std)
        bar_width = 0.55
        ax.bar(
            indices, means, yerr=2 * stds, width=bar_width,
            color="#34D399", edgecolor="#065F46", linewidth=1.2,
            alpha=0.75, capsize=5, ecolor="#047857",
            label="GP Predicted Efficacy (μ ± 2σ)",
            zorder=3,
        )

        # Scatter plot of Ground Truth Oracle
        ax.scatter(
            indices, trues, color="#DC2626", s=90, marker="D",
            edgecolors="#1E293B", linewidths=1.5, zorder=5,
            label="True Ground Truth Efficacy (Oracle)",
        )

        ax.set_xticks(indices)
        ax.set_xticklabels(coalition_names, rotation=35, ha="right", fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05, 1.25)
        ax.set_ylabel("Task Efficacy ξ(Q) ∈ [0, 1]", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Gaussian Process Posterior vs. Ground Truth: Task {task.id} ('{task.name}')",
            fontsize=13, fontweight="bold", pad=12
        )
        ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
        ax.legend(loc="upper left", fontsize=10)

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved GP posterior plot to: {save_path}")

        if show:
            plt.show()

        return fig

    def render_realizability_manifold_surface(
        self,
        task: Task,
        oracle: GroundTruthEfficacyOracle,
        realizable_coalitions: List[Coalition],
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Render 2D continuous performance contour with discrete physical realizable points overlaid."""
        fig, ax = plt.subplots(figsize=(10, 7))

        # Continuous grid: Payload [0, 2000] vs Camera [0.0, 1.0]
        payloads = np.linspace(0, 2200, 100)
        cameras = np.linspace(0.0, 1.0, 100)
        P, C = np.meshgrid(payloads, cameras)

        Z = np.zeros_like(P)
        for i in range(len(cameras)):
            for j in range(len(payloads)):
                # Evaluate continuous dummy trait vector with fixed LiDAR=1, Lift=1
                vec = np.array([P[i, j], 1.0, C[i, j], 1.0, 0.0, 2.0])
                Z[i, j] = oracle.predict_vector(vec, task)

        # Plot contour
        cs = ax.contourf(P, C, Z, levels=20, cmap="viridis", alpha=0.85)
        cbar = plt.colorbar(cs, ax=ax)
        cbar.set_label("Underlying Task Efficacy ξ(Q)", fontsize=10, fontweight="bold")

        # Plot discrete realizable points Ω
        r_payloads = [c.aggregate_traits.payload_kg for c in realizable_coalitions]
        r_cameras = [c.aggregate_traits.has_camera_rgb for c in realizable_coalitions]
        ax.scatter(
            r_payloads, r_cameras,
            color="#EF4444", s=130, marker="o", edgecolors="white", linewidths=1.8,
            label="Directly Realizable Coalitions Ω (Formable)",
            zorder=6,
        )

        # Annotate a few coalitions
        for c in realizable_coalitions[:5]:
            ax.annotate(
                str(c),
                (c.aggregate_traits.payload_kg, c.aggregate_traits.has_camera_rgb),
                xytext=(8, -8), textcoords="offset points",
                fontsize=8, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0F172A", alpha=0.75, edgecolor="none")
            )

        ax.set_title(f"Continuous Trait-Efficacy Landscape vs. Discrete Realizable Set Ω\n(Task {task.id}: '{task.name}')", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Collective Payload (kg)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Collective RGB Camera Feed Quality", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.4, color="white")

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved realizability manifold surface to: {save_path}")

        if show:
            plt.show()

        return fig
