"""Publication-Quality Benchmark Visualizations for E-ITAGS.

Generates the figures replicating Section 6 of the research paper:
- Figure 5 & 6: Efficacy vs. Makespan optimization & Runtime decomposition
- Figure 7: Validation of theoretical bounds on optimality gaps
- Figure 9 & 11: Active learning cumulative regret & Best Uncovered Reward (BUR)
- Figure 10: Single-step runtime scalability across fleet sizes
- Figure 12: Downstream closed-loop task allocation efficacy distributions
- Consolidated Paper Summary Figure (Multi-panel publication figure)
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..experiments.benchmark_suite import (
    ComparativeITAGSResult,
    OptimalityGapResult,
    ActiveLearningBenchmarkResult,
    ClosedLoopBenchmarkResult,
    ScalabilityBenchmarkResult,
    FailureBoundaryResult,
)


class BenchmarkVisualizer:
    """Renderer for publication-grade benchmark figures."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Apply standard clean styling
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "grid.color": "#e0e0e0",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
        })

    def plot_efficacy_vs_makespan(
        self,
        results: List[ComparativeITAGSResult],
        save_path: Optional[str] = None,
    ) -> str:
        """Figure 5: E-ITAGS vs. Standard ITAGS: Efficacy, Makespan, and Runtime.
        
        Shows that E-ITAGS consistently improves efficacy while adhering to time budgets.
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "paper_fig5_efficacy_vs_makespan.png")

        def _format_label(name: str) -> str:
            if "Scenario" in name:
                parts = name.split(":")
                if len(parts) == 2:
                    return f"{parts[0].replace('Scenario ', 'S')}:\n{parts[1].strip()}"
            return name.replace("Warehouse-Inst-", "Inst ")

        instances = [_format_label(r.instance_name) for r in results]
        x = np.arange(len(instances))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

        # Panel 1: Allocation Efficacy
        itags_eff = [r.itags_efficacy for r in results]
        eitags_eff = [r.eitags_efficacy for r in results]

        rects1 = ax1.bar(x - width/2, itags_eff, width, label="Standard ITAGS (Makespan-only)", color="#7f8c8d", alpha=0.9)
        rects2 = ax1.bar(x + width/2, eitags_eff, width, label="E-ITAGS (Efficacy-aware)", color="#27ae60", alpha=0.95)

        ax1.set_title("Total Mission Allocation Efficacy", fontsize=12, fontweight="bold", pad=10)
        ax1.set_xlabel("Problem Instance", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Total Efficacy (Score / 5.0)", fontsize=10, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(instances, fontsize=9)
        ax1.set_ylim(0, 5.5)
        ax1.grid(True, axis="y")
        ax1.legend(loc="lower right", framealpha=0.9)

        # Annotate efficacy gain percentages
        for i in range(len(instances)):
            gain = ((eitags_eff[i] - itags_eff[i]) / max(1e-3, itags_eff[i])) * 100.0
            ax1.annotate(
                f"+{gain:.1f}%",
                xy=(x[i] + width/2, eitags_eff[i]),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1e8449"
            )

        # Panel 2: Mission Makespan (Adherence to budget)
        itags_ms = [r.itags_makespan for r in results]
        eitags_ms = [r.eitags_makespan for r in results]

        ax2.bar(x - width/2, itags_ms, width, label="Standard ITAGS", color="#2980b9", alpha=0.9)
        ax2.bar(x + width/2, eitags_ms, width, label="E-ITAGS (Under Budget)", color="#8e44ad", alpha=0.9)

        ax2.set_title("Mission Makespan Comparison", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Problem Instance", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Makespan (Seconds)", fontsize=10, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(instances, fontsize=9)
        ax2.grid(True, axis="y")
        ax2.legend(loc="upper left", framealpha=0.9)

        plt.suptitle("E-ITAGS Efficacy Optimization under Schedule Time Budgets (Replicating Paper Fig. 5)",
                     fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_runtime_breakdown(
        self,
        results: List[ComparativeITAGSResult],
        save_path: Optional[str] = None,
    ) -> str:
        """Figure 6: Computational Runtime Decomposition (Search, Scheduling, Motion Planning)."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, "paper_fig6_runtime_breakdown.png")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

        itags_search = [r.itags_search_time * 1000 for r in results]
        eitags_search = [r.eitags_search_time * 1000 for r in results]

        itags_sched = [r.itags_sched_time * 1000 for r in results]
        eitags_sched = [r.eitags_sched_time * 1000 for r in results]

        itags_motion = [r.itags_motion_time * 1000 for r in results]
        eitags_motion = [r.eitags_motion_time * 1000 for r in results]

        # Panel 1: Task Allocation Search Time
        bp1 = ax1.boxplot([itags_search, eitags_search], tick_labels=["ITAGS", "E-ITAGS"], patch_artist=True)
        bp1['boxes'][0].set_facecolor('#bdc3c7')
        bp1['boxes'][1].set_facecolor('#e67e22')
        ax1.set_title("Task Allocation Search Time", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Time (ms)", fontsize=10, fontweight="bold")
        ax1.grid(True, axis="y")

        # Panel 2: Scheduling Time
        bp2 = ax2.boxplot([itags_sched, eitags_sched], tick_labels=["ITAGS", "E-ITAGS"], patch_artist=True)
        bp2['boxes'][0].set_facecolor('#bdc3c7')
        bp2['boxes'][1].set_facecolor('#3498db')
        ax2.set_title("Scheduling Time", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Time (ms)", fontsize=10, fontweight="bold")
        ax2.grid(True, axis="y")

        # Panel 3: Motion Planning Time
        bp3 = ax3.boxplot([itags_motion, eitags_motion], tick_labels=["ITAGS", "E-ITAGS"], patch_artist=True)
        bp3['boxes'][0].set_facecolor('#bdc3c7')
        bp3['boxes'][1].set_facecolor('#2ecc71')
        ax3.set_title("Motion Planning Time", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Time (ms)", fontsize=10, fontweight="bold")
        ax3.grid(True, axis="y")

        plt.suptitle("Computational Runtime Breakdown across Modules (Replicating Paper Fig. 6)",
                     fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_suboptimality_bounds(
        self,
        res: OptimalityGapResult,
        save_path: Optional[str] = None,
    ) -> str:
        """Figure 7: Empirical validation of theoretical suboptimality bound on optimality gap."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, "paper_fig7_suboptimality_bounds.png")

        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

        alphas = np.array(res.alpha_values)
        actual = np.array(res.actual_gaps)
        theor = np.array(res.theoretical_bounds)

        # Plot theoretical bound curve
        ax.plot(alphas, theor, color="#e74c3c", linewidth=2.5, linestyle="--",
                label="Theoretical Upper Bound (Eq. 5)", zorder=3)

        # Plot empirical actual optimality gap
        ax.plot(alphas, actual, color="#2980b9", linewidth=2.5, marker="o", markersize=6,
                label="Empirical Optimality Gap", zorder=4)

        # Shade region between actual and theoretical bound
        ax.fill_between(alphas, actual, theor, color="#e74c3c", alpha=0.15,
                        label="Guaranteed Safe Feasibility Region")

        ax.set_title("Empirical Validation of Theoretical Bounds on Optimality Gaps\n(Replicating Paper Fig. 7)",
                     fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(r"Trade-off Parameter $\alpha$ (Weight on NAC)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalized Allocation Optimality Gap", fontsize=11, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=10)

        # Annotations
        ax.annotate(r"$\alpha = 0 \to$ Pure Makespan Minimization", xy=(0.05, actual[0]),
                    xytext=(0.10, 0.25), arrowprops=dict(arrowstyle="->", color="#333333"),
                    fontsize=9, fontweight="bold")
        ax.annotate(r"$\alpha \to 1 \to$ Maximum Mission Efficacy", xy=(alphas[-1], actual[-1]),
                    xytext=(0.55, 0.08), arrowprops=dict(arrowstyle="->", color="#27ae60"),
                    fontsize=9, fontweight="bold", color="#1e8449")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_cumulative_regret(
        self,
        res: ActiveLearningBenchmarkResult,
        save_path: Optional[str] = None,
    ) -> str:
        """Figure 9: Cumulative Regret trajectories across the 4 active learning strategies."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, "paper_fig9_cumulative_regrets.png")

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

        color_map = {
            "exact": "#27ae60",          # Emerald Green (Best)
            "box": "#2980b9",            # Sapphire Blue
            "unconstrained": "#e74c3c",  # Red (Worst)
            "random": "#95a5a6",         # Grey (Baseline)
        }
        label_map = {
            "exact": "Exact Realizable (Ω)",
            "box": "Box-Constrained (Proj)",
            "unconstrained": "Unconstrained (Proj)",
            "random": "Random Baseline (Ω)",
        }

        iters = res.iterations
        for strat in res.strategies:
            regrets = res.cumulative_regrets[strat]  # (n_trials, n_iters)
            mean_reg = np.mean(regrets, axis=0)
            std_reg = np.std(regrets, axis=0)
            c = color_map.get(strat, "#333333")
            lbl = label_map.get(strat, strat.capitalize())

            ax.plot(iters, mean_reg, color=c, linewidth=2.5, label=lbl, zorder=3)
            ax.fill_between(iters, np.maximum(0, mean_reg - std_reg), mean_reg + std_reg,
                            color=c, alpha=0.18, zorder=2)

        ax.set_title("Active Learning Cumulative Regret vs. Number of Queries\n(Replicating Paper Fig. 9)",
                     fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Active Learning Iterations / Query Budget", fontsize=11, fontweight="bold")
        ax.set_ylabel("Cumulative Regret", fontsize=11, fontweight="bold")
        ax.set_xlim(1, len(iters))
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_single_step_runtime(
        self,
        res: ActiveLearningBenchmarkResult,
        save_path: Optional[str] = None,
    ) -> str:
        """Figure 10: Single-step runtime scalability across fleet sizes."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, "paper_fig10_runtime_scalability.png")

        fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)

        color_map = {
            "exact": "#27ae60",
            "box": "#2980b9",
            "unconstrained": "#e74c3c",
            "random": "#95a5a6",
        }
        label_map = {
            "exact": "Exact Realizable (Ω)",
            "box": "Box-Constrained (Proj)",
            "unconstrained": "Unconstrained (Proj)",
            "random": "Random Baseline",
        }

        sizes = res.team_sizes
        for strat in res.strategies:
            times = [t * 1000.0 for t in res.runtime_per_step[strat]]
            c = color_map.get(strat, "#333333")
            lbl = label_map.get(strat, strat.capitalize())
            ax.plot(sizes, times, marker="o", linewidth=2.2, color=c, label=lbl)

        ax.set_title("Active Learning Single-Step Query Runtime vs. Fleet Size\n(Replicating Paper Fig. 10)",
                     fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Number of Robots in Warehouse Fleet", fontsize=11, fontweight="bold")
        ax.set_ylabel("Single-Step Query Runtime (ms)", fontsize=11, fontweight="bold")
        ax.set_xticks(sizes)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_best_uncovered_reward(
        self,
        res: ActiveLearningBenchmarkResult,
        save_path: Optional[str] = None,
    ) -> str:
        """Figure 11: Best Uncovered Reward (BUR) progression over query iterations."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, "paper_fig11_best_uncovered_reward.png")

        fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)

        color_map = {
            "exact": "#27ae60",
            "box": "#2980b9",
            "unconstrained": "#e74c3c",
            "random": "#95a5a6",
        }
        label_map = {
            "exact": "Exact Realizable (Ω)",
            "box": "Box-Constrained (Proj)",
            "unconstrained": "Unconstrained (Proj)",
            "random": "Random Baseline",
        }

        iters = res.iterations
        for strat in res.strategies:
            burs = res.best_uncovered_rewards[strat]
            mean_bur = np.mean(burs, axis=0)
            std_bur = np.std(burs, axis=0)
            c = color_map.get(strat, "#333333")
            lbl = label_map.get(strat, strat.capitalize())

            ax.plot(iters, mean_bur, color=c, linewidth=2.5, label=lbl)
            ax.fill_between(iters, np.maximum(0, mean_bur - std_bur), np.minimum(1.0, mean_bur + std_bur),
                            color=c, alpha=0.15)

        ax.set_title("Best Uncovered Reward (BUR) Progression (Replicating Paper Fig. 11)",
                     fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Query Iteration", fontsize=11, fontweight="bold")
        ax.set_ylabel("Maximum Efficacy Discovered", fontsize=11, fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="lower right", framealpha=0.95, fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_closed_loop_efficacy(
        self,
        res: ClosedLoopBenchmarkResult,
        save_path: Optional[str] = None,
    ) -> str:
        """Figure 12: Downstream Task Allocation Efficacy under Learned GP Models."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, "paper_fig12_closed_loop_efficacy.png")

        fig, axes = plt.subplots(1, len(res.weights), figsize=(14, 4.8), dpi=150, sharey=True)
        if len(res.weights) == 1:
            axes = [axes]

        methods = res.strategies  # ["exact", "box", "unconstrained", "oracle"]
        x = np.arange(len(methods))
        width = 0.35

        for idx, w in enumerate(res.weights):
            ax = axes[idx]
            pred_means = [np.mean(res.predicted_efficacies[m][w]) for m in methods]
            true_means = [np.mean(res.true_efficacies[m][w]) for m in methods]

            ax.bar(x - width/2, pred_means, width, label="Predicted (GP Mean)", color="#e74c3c", alpha=0.85)
            ax.bar(x + width/2, true_means, width, label="Ground Truth Efficacy", color="#2980b9", alpha=0.85)

            ax.set_title(rf"Trade-off Weight $\alpha = {w}$", fontsize=11, fontweight="bold")
            ax.set_xlabel("Sampling Method", fontsize=10, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([m.capitalize() for m in methods], fontsize=9)
            ax.set_ylim(0, 5.5)
            ax.grid(True, axis="y")
            if idx == 0:
                ax.set_ylabel("Total Mission Efficacy", fontsize=10, fontweight="bold")
                ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

        plt.suptitle("Total Mission Allocation Efficacy under Actively Learned Models (Replicating Paper Fig. 12)",
                     fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_comprehensive_paper_figure(
        self,
        exp1_res: List[ComparativeITAGSResult],
        exp2_res: OptimalityGapResult,
        exp3_res: ActiveLearningBenchmarkResult,
        exp4_res: ClosedLoopBenchmarkResult,
        save_path: Optional[str] = None,
    ) -> str:
        """Consolidated Multi-Panel Publication-Quality Figure."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, "e_itags_comprehensive_paper_figure.png")

        fig = plt.figure(figsize=(18, 12), dpi=200)
        gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.25)

        # Panel A: Efficacy & Makespan (Exp 1)
        ax_a = fig.add_subplot(gs[0, 0])
        inst = [f"Inst {i+1}" for i in range(len(exp1_res))]
        x = np.arange(len(inst))
        w = 0.35
        ax_a.bar(x - w/2, [r.itags_efficacy for r in exp1_res], w, label="ITAGS (baseline)", color="#7f8c8d")
        ax_a.bar(x + w/2, [r.eitags_efficacy for r in exp1_res], w, label="E-ITAGS (ours)", color="#27ae60")
        ax_a.set_title("(a) Mission Efficacy under Time Budget", fontweight="bold", fontsize=11)
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(inst, fontsize=8)
        ax_a.set_ylabel("Total Efficacy", fontweight="bold", fontsize=10)
        ax_a.grid(True, axis="y")
        ax_a.legend(fontsize=8, loc="lower right")

        # Panel B: Suboptimality Bounds (Exp 2)
        ax_b = fig.add_subplot(gs[0, 1])
        alphas = np.array(exp2_res.alpha_values)
        ax_b.plot(alphas, exp2_res.theoretical_bounds, "r--", linewidth=2.2, label="Theoretical Bound (Eq. 5)")
        ax_b.plot(alphas, exp2_res.actual_gaps, "b-o", markersize=4, linewidth=2.0, label="Empirical Gap")
        ax_b.fill_between(alphas, exp2_res.actual_gaps, exp2_res.theoretical_bounds, color="r", alpha=0.15)
        ax_b.set_title(r"(b) Optimality Gap vs. Trade-off $\alpha$", fontweight="bold", fontsize=11)
        ax_b.set_xlabel(r"Weight $\alpha$", fontweight="bold", fontsize=10)
        ax_b.set_ylabel("Normalized Gap", fontweight="bold", fontsize=10)
        ax_b.grid(True)
        ax_b.legend(fontsize=8, loc="upper left")

        # Panel C: Cumulative Regret (Exp 3)
        ax_c = fig.add_subplot(gs[0, 2])
        color_map = {"exact": "#27ae60", "box": "#2980b9", "unconstrained": "#e74c3c", "random": "#95a5a6"}
        for s in exp3_res.strategies:
            mean_r = np.mean(exp3_res.cumulative_regrets[s], axis=0)
            ax_c.plot(exp3_res.iterations, mean_r, color=color_map.get(s, "#333"), linewidth=2.0, label=s.capitalize())
        ax_c.set_title("(c) Active Learning Cumulative Regret", fontweight="bold", fontsize=11)
        ax_c.set_xlabel("Query Iteration", fontweight="bold", fontsize=10)
        ax_c.set_ylabel("Cumulative Regret", fontweight="bold", fontsize=10)
        ax_c.grid(True)
        ax_c.legend(fontsize=8, loc="upper left")

        # Panel D: Single-Step Runtime Scaling (Exp 3)
        ax_d = fig.add_subplot(gs[1, 0])
        for s in exp3_res.strategies:
            times_ms = [t * 1000.0 for t in exp3_res.runtime_per_step[s]]
            ax_d.plot(exp3_res.team_sizes, times_ms, marker="s", color=color_map.get(s, "#333"), linewidth=1.8, label=s.capitalize())
        ax_d.set_title("(d) Query Runtime Scaling vs. Fleet Size", fontweight="bold", fontsize=11)
        ax_d.set_xlabel("Fleet Size (Robots)", fontweight="bold", fontsize=10)
        ax_d.set_ylabel("Step Runtime (ms)", fontweight="bold", fontsize=10)
        ax_d.set_xticks(exp3_res.team_sizes)
        ax_d.grid(True)
        ax_d.legend(fontsize=8, loc="upper left")

        # Panel E: Best Uncovered Reward (Exp 3)
        ax_e = fig.add_subplot(gs[1, 1])
        for s in exp3_res.strategies:
            mean_bur = np.mean(exp3_res.best_uncovered_rewards[s], axis=0)
            ax_e.plot(exp3_res.iterations, mean_bur, color=color_map.get(s, "#333"), linewidth=2.0, label=s.capitalize())
        ax_e.set_title("(e) Best Uncovered Reward (BUR)", fontweight="bold", fontsize=11)
        ax_e.set_xlabel("Query Iteration", fontweight="bold", fontsize=10)
        ax_e.set_ylabel("Discovered Efficacy", fontweight="bold", fontsize=10)
        ax_e.set_ylim(0, 1.05)
        ax_e.grid(True)
        ax_e.legend(fontsize=8, loc="lower right")

        # Panel F: Downstream Closed-Loop Task Allocation (Exp 4)
        ax_f = fig.add_subplot(gs[1, 2])
        w_eval = 0.5
        methods = exp4_res.strategies
        xm = np.arange(len(methods))
        pred_m = [np.mean(exp4_res.predicted_efficacies[m][w_eval]) for m in methods]
        true_m = [np.mean(exp4_res.true_efficacies[m][w_eval]) for m in methods]
        ax_f.bar(xm - w/2, pred_m, w, label="Predicted Efficacy", color="#e74c3c", alpha=0.85)
        ax_f.bar(xm + w/2, true_m, w, label="True Ground-Truth", color="#2980b9", alpha=0.85)
        ax_f.set_title(r"(f) Downstream Mission Efficacy ($\alpha = 0.5$)", fontweight="bold", fontsize=11)
        ax_f.set_xticks(xm)
        ax_f.set_xticklabels([m.capitalize() for m in methods], fontsize=8)
        ax_f.set_ylabel("Total Efficacy", fontweight="bold", fontsize=10)
        ax_f.grid(True, axis="y")
        ax_f.legend(fontsize=8, loc="lower right")

        plt.suptitle("E-ITAGS Full System Benchmark & Empirical Validation Summary",
                     fontsize=15, fontweight="bold", y=0.98)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_failure_boundaries(
        self,
        res: FailureBoundaryResult,
        save_path: Optional[str] = None,
    ) -> str:
        """Publication Figure: E-ITAGS Failure Boundary and Stress Test Analysis.
        
        Demonstrates the operational, physical, and epistemic boundaries where E-ITAGS fails:
        1. Temporal budget compression cliff
        2. Task overload and capacity exhaustion
        3. Physical fleet depletion and zone infeasibility
        4. Active learning epistemic noise breakdown
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "paper_fig_failure_boundaries.png")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11), dpi=150)

        # Panel 1: Temporal Budget Compression Cliff
        x1 = res.budget_factors
        ax1.plot(x1, [sr * 100.0 for sr in res.budget_success_rates], marker="o", color="#27ae60", linewidth=2.2, label="Mission Success Rate (%)")
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x1, res.budget_efficacies, marker="s", color="#2980b9", linestyle="--", linewidth=1.8, label="Delivered Efficacy")
        ax1.axvline(x=res.critical_budget_factor, color="#c0392b", linestyle=":", linewidth=2.0,
                    label=f"Failure Boundary ($T < {res.critical_budget_factor:.2f} T_0$)")
        ax1.axvspan(min(x1) - 0.05, res.critical_budget_factor, color="#e74c3c", alpha=0.12, label="Infeasible Region")
        ax1.set_title("(a) Temporal Budget Compression Cliff", fontsize=11, fontweight="bold")
        ax1.set_xlabel(r"Budget Scaling Factor ($T_{budget} / T_0$)", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Mission Success Rate (%)", fontsize=10, fontweight="bold", color="#27ae60")
        ax1_twin.set_ylabel("Delivered Total Efficacy", fontsize=10, fontweight="bold", color="#2980b9")
        ax1.set_ylim(-5, 105)
        ax1_twin.set_ylim(-0.2, 5.5)
        ax1.grid(True)
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)

        # Panel 2: Workload Task Saturation & Queue Overload
        x2 = res.task_counts
        ax2.plot(x2, [sr * 100.0 for sr in res.task_success_rates], marker="o", color="#e67e22", linewidth=2.2, label="Success Rate (%)")
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x2, res.task_nodes_expanded, marker="^", color="#8e44ad", linestyle="--", linewidth=1.8, label="Nodes Expanded")
        ax2.axvline(x=res.critical_task_capacity, color="#c0392b", linestyle=":", linewidth=2.0,
                    label=f"Capacity Limit ($N > {res.critical_task_capacity}$)")
        ax2.axvspan(res.critical_task_capacity, max(x2) + 1, color="#e74c3c", alpha=0.12, label="Deadline Pruned Region")
        ax2.set_title("(b) Task Queue Overload & Capacity Horizon", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Concurrent Task Count ($N$)", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Success Rate (%)", fontsize=10, fontweight="bold", color="#e67e22")
        ax2_twin.set_ylabel("Search Nodes Expanded", fontsize=10, fontweight="bold", color="#8e44ad")
        ax2.set_ylim(-5, 105)
        ax2.grid(True)
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8)

        # Panel 3: Fleet Depletion & Zone/Trait Infeasibility
        x3 = np.arange(len(res.fleet_scenarios))
        w3 = 0.35
        fleet_labels = [
            s.replace(" Scenario", "").replace(" (No ", "\n(-").replace(")", "")
            for s in res.fleet_scenarios
        ]
        ax3.bar(x3 - w3/2, [sr * 100.0 for sr in res.fleet_success_rates], w3, label="Success Rate (%)", color="#2c3e50", alpha=0.9)
        ax3_twin = ax3.twinx()
        ax3_twin.bar(x3 + w3/2, res.unserviced_task_counts, w3, label="Unserviced Tasks", color="#c0392b", alpha=0.85)
        ax3.set_title("(c) Fleet Depletion & Zone/Trait Infeasibility", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Fleet Archetype Configuration", fontsize=10, fontweight="bold")
        ax3.set_ylabel("Mission Success Rate (%)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3_twin.set_ylabel("Unserviced Task Count", fontsize=10, fontweight="bold", color="#c0392b")
        ax3.set_xticks(x3)
        ax3.set_xticklabels(fleet_labels, fontsize=8)
        ax3.set_ylim(0, 110)
        ax3_twin.set_ylim(0, max(res.unserviced_task_counts) + 2)
        ax3.grid(True, axis="y")
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

        # Panel 4: Active Learning Epistemic Noise Breakdown
        x4 = res.noise_levels
        ax4.plot(x4, res.noise_true_efficacies, marker="o", color="#27ae60", linewidth=2.2, label="Downstream True Efficacy")
        ax4.plot(x4, res.noise_calibration_errors, marker="^", color="#e74c3c", linewidth=2.0, linestyle="--",
                 label=r"Calibration Error $|\hat{y} - y_{true}|$")
        ax4.axvline(x=res.critical_noise_tolerance, color="#c0392b", linestyle=":", linewidth=2.0,
                    label=rf"Noise Tolerance Limit ($\sigma > {res.critical_noise_tolerance:.2f}$)")
        ax4.axvspan(res.critical_noise_tolerance, max(x4) + 0.05, color="#e74c3c", alpha=0.12, label="Noise Degraded Region")
        ax4.set_title(r"(d) Epistemic Observation Noise Breakdown", fontsize=11, fontweight="bold")
        ax4.set_xlabel(r"Sensor / Observation Noise $\sigma$", fontsize=10, fontweight="bold")
        ax4.set_ylabel("Efficacy / Calibration Metric", fontsize=10, fontweight="bold")
        ax4.set_ylim(-0.05, 1.15)
        ax4.grid(True)
        ax4.legend(loc="center left", fontsize=8)

        plt.suptitle("E-ITAGS Failure Boundary & Operational Limits Under Stress Testing",
                     fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path
