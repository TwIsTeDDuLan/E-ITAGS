"""Visualization of ITAGS Graph Search Trees and Pareto Optimization Frontiers."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np

from ..core.graph_search import SearchNode, SearchResult


class SearchTreeVisualizer:
    """Renders hierarchical search tree graphs and Pareto efficacy vs. makespan curves."""

    @staticmethod
    def _hierarchy_pos(G: nx.DiGraph, root: Any, width: float = 1.0, vert_gap: float = 0.2, vert_loc: float = 0, xcenter: float = 0.5) -> Dict[Any, Tuple[float, float]]:
        """Compute hierarchical 2D coordinates for a tree without requiring external graphviz binaries."""
        pos = {root: (xcenter, vert_loc)}
        children = list(G.successors(root))
        if not children:
            return pos

        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos.update(SearchTreeVisualizer._hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap, xcenter=nextx))
        return pos

    def render_search_tree(
        self,
        search_result: SearchResult,
        save_path: Optional[str | Path] = None,
        max_nodes: int = 50,
        show: bool = False,
    ) -> plt.Figure:
        """Render the expanded ITAGS search tree highlighting the optimal solution path."""
        fig, ax = plt.subplots(figsize=(15, 10))

        # Collect nodes to plot: solution path nodes + representative sample of explored nodes
        sol_nodes = set(search_result.solution_path)
        explored = search_result.stats.explored_nodes

        nodes_to_include = list(sol_nodes)
        for n in explored:
            if len(nodes_to_include) >= max_nodes:
                break
            if n not in nodes_to_include:
                nodes_to_include.append(n)

        node_map = {n.node_id: n for n in nodes_to_include}

        # Build DiGraph
        G = nx.DiGraph()
        for n in nodes_to_include:
            G.add_node(n.node_id)
            if n.parent and n.parent.node_id in node_map:
                G.add_edge(n.parent.node_id, n.node_id)

        root_id = 0
        if root_id not in G:
            root_id = nodes_to_include[0].node_id

        # Compute layout coordinates
        try:
            pos = self._hierarchy_pos(G, root_id, width=2.0, vert_gap=0.25)
        except Exception:
            pos = nx.spring_layout(G, seed=42)

        # Distinguish solution path edges and nodes
        sol_node_ids = {n.node_id for n in search_result.solution_path}
        sol_edges = [
            (search_result.solution_path[i].node_id, search_result.solution_path[i + 1].node_id)
            for i in range(len(search_result.solution_path) - 1)
            if search_result.solution_path[i].node_id in G and search_result.solution_path[i + 1].node_id in G
        ]
        other_edges = [e for e in G.edges() if e not in sol_edges]

        # Draw Non-Solution Edges
        nx.draw_networkx_edges(
            G, pos, edgelist=other_edges, ax=ax,
            edge_color="#CBD5E1", width=1.4, alpha=0.7, arrows=True, arrowsize=10
        )
        # Draw Solution Path Edges
        nx.draw_networkx_edges(
            G, pos, edgelist=sol_edges, ax=ax,
            edge_color="#10B981", width=3.0, alpha=1.0, arrows=True, arrowsize=14
        )

        # Node Colors
        node_colors = []
        for n_id in G.nodes():
            if n_id == search_result.goal_node.node_id if search_result.goal_node else -1:
                node_colors.append("#059669")  # Emerald for Goal
            elif n_id == 0:
                node_colors.append("#3B82F6")  # Blue for Root
            elif n_id in sol_node_ids:
                node_colors.append("#34D399")  # Mint for Solution Path
            else:
                node_colors.append("#E2E8F0")  # Soft Slate for Explored

        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors, node_size=1200, edgecolors="#1E293B", linewidths=1.2
        )

        # Node Labels
        labels = {}
        for n_id in G.nodes():
            n = node_map[n_id]
            if n_id == 0:
                labels[n_id] = f"Root\nf={n.f_cost:.2f}"
            elif n.expanded_task_id:
                c_str = "+".join(r_id[-2:] for r_id in n.expanded_coalition.robot_ids) if n.expanded_coalition else ""
                labels[n_id] = f"{n.expanded_task_id}→[{c_str}]\nf={n.f_cost:.2f}"
            else:
                labels[n_id] = f"N{n_id}\nf={n.f_cost:.2f}"

        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold", ax=ax)

        ax.set_title(
            f"E-ITAGS Interleaved Search Tree (Expansions={search_result.stats.nodes_expanded}, "
            f"Makespan={search_result.makespan:.1f}s, Efficacy={search_result.mean_efficacy:.2f})",
            fontsize=13, fontweight="bold", pad=12
        )
        ax.axis("off")

        # Custom Legend
        legend_handles = [
            patches.Patch(facecolor="#3B82F6", edgecolor="#1E293B", label="Root Node"),
            patches.Patch(facecolor="#34D399", edgecolor="#1E293B", label="Solution Path"),
            patches.Patch(facecolor="#059669", edgecolor="#1E293B", label="Goal Node"),
            patches.Patch(facecolor="#E2E8F0", edgecolor="#1E293B", label="Explored Nodes"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9.5)

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved search tree visualization to: {save_path}")

        if show:
            plt.show()

        return fig

    def render_pareto_front(
        self,
        pareto_results: List[SearchResult],
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Render the Pareto frontier comparing mission makespan vs. total allocation efficacy."""
        fig, ax = plt.subplots(figsize=(9, 6.5))

        # Filter successful results
        valid = [r for r in pareto_results if r.solution_found]
        if not valid:
            ax.text(0.5, 0.5, "No feasible solutions found across weights", ha="center", va="center")
            return fig

        makespans = [r.makespan for r in valid]
        mean_effs = [r.mean_efficacy * 100.0 for r in valid]
        weights = [r.weight for r in valid]

        # Scatter points
        sc = ax.scatter(
            makespans, mean_effs,
            c=weights, cmap="viridis", s=180, edgecolors="#0F172A",
            linewidth=1.8, zorder=5
        )

        # Draw connecting line ordered by makespan
        sorted_pairs = sorted(zip(makespans, mean_effs))
        xs = [p[0] for p in sorted_pairs]
        ys = [p[1] for p in sorted_pairs]
        ax.plot(xs, ys, color="#94A3B8", linestyle="--", linewidth=1.8, zorder=3)

        # Label each point with its weight value
        for r in valid:
            label = f"w={r.weight:.2f}\n({r.makespan:.0f}s, {r.mean_efficacy*100:.0f}%)"
            ax.annotate(
                label,
                (r.makespan, r.mean_efficacy * 100.0),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="#CBD5E1")
            )

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Tradeoff Weight w (0=Schedule, 1=Efficacy)", fontsize=10, fontweight="bold")

        ax.set_xlabel("Mission Makespan (Seconds) — [Lower is Faster]", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel("Mean Task Efficacy (%) — [Higher is Better Match]", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_title("E-ITAGS Pareto Frontier: Makespan vs. Trait Efficacy Tradeoff", fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved Pareto frontier plot to: {save_path}")

        if show:
            plt.show()

        return fig
