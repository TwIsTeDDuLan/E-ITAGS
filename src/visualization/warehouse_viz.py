"""2D Warehouse visualization and TraitEngine dashboard using matplotlib."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from ..models.warehouse import WarehouseGrid, Location
from ..models.robot import Robot, RobotType, TRAIT_NAMES
from ..models.task import Task
from ..models.coalition import Coalition
from ..core.trait_engine import TraitEngine


class WarehouseVisualizer:
    """Renders 2D floor maps, allocation diagrams, trait satisfaction heatmaps,
    and realizability analysis for the E-ITAGS system.
    """

    ZONE_COLORS = {
        "receiving": "#E0F2FE",     # Soft light sky blue
        "sorting": "#FEF3C7",       # Soft warm amber
        "staging": "#F3E8FF",       # Soft light purple
        "narrow_aisle": "#FEE2E2",  # Soft salmon red (narrow access restricted)
        "shipping": "#DCFCE7",      # Soft light emerald
    }

    ZONE_BORDER_COLORS = {
        "receiving": "#0284C7",
        "sorting": "#D97706",
        "staging": "#9333EA",
        "narrow_aisle": "#DC2626",
        "shipping": "#16A34A",
    }

    ROBOT_ICONS = {
        RobotType.FORKLIFT: ("🚜", "#2563EB"),
        RobotType.DRONE: ("🚁", "#7C3AED"),
        RobotType.IW_HUB: ("🤖", "#059669"),
    }

    def __init__(self, warehouse: WarehouseGrid):
        self.warehouse = warehouse

    def _draw_zones(self, ax: plt.Axes) -> None:
        """Helper to draw all warehouse zones on a given matplotlib axes."""
        for z_name, zone in self.warehouse.zones.items():
            if zone.cells:
                min_x = min(c.x for c in zone.cells)
                max_x = max(c.x for c in zone.cells) + 1
                min_y = min(c.y for c in zone.cells)
                max_y = max(c.y for c in zone.cells) + 1
                w = max_x - min_x
                h = max_y - min_y

                bg_color = self.ZONE_COLORS.get(z_name, "#F1F5F9")
                border_color = self.ZONE_BORDER_COLORS.get(z_name, "#475569")
                rect = patches.Rectangle(
                    (min_x, min_y), w, h,
                    linewidth=1.8,
                    edgecolor=border_color,
                    facecolor=bg_color,
                    alpha=0.65,
                    linestyle="--" if zone.narrow_access_only else "-",
                )
                ax.add_patch(rect)

                # Zone Label with clean badge
                badge_text = "Narrow Access Only" if zone.narrow_access_only else "General Access"
                label = f"{zone.name.replace('_', ' ').title()}\n({badge_text})"
                ax.text(
                    zone.center.x, zone.center.y,
                    label,
                    ha="center", va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#1E293B",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#CBD5E1")
                )

    def render(
        self,
        robots: Optional[List[Robot]] = None,
        tasks: Optional[List[Task]] = None,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Render standard 2D floor map of the warehouse with zones, robots, and tasks."""
        fig, ax = plt.subplots(figsize=(12, 8))
        self._render_map_on_axis(ax, robots=robots, tasks=tasks)

        ax.set_title("E-ITAGS Autonomous Warehouse Environment", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("X (meters)", fontsize=10)
        ax.set_ylabel("Y (meters)", fontsize=10)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Saved warehouse map to: {save_path}")

        if show:
            plt.show()

        return fig

    def _render_map_on_axis(
        self,
        ax: plt.Axes,
        robots: Optional[List[Robot]] = None,
        tasks: Optional[List[Task]] = None,
        allocations: Optional[Dict[str, Coalition]] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Render map details on a given axis, returning robot coordinate map."""
        ax.set_xlim(-1, self.warehouse.width + 1)
        ax.set_ylim(-1, self.warehouse.height + 1)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.3, color="#94A3B8")

        # 1. Draw Zones
        self._draw_zones(ax)

        # 2. Draw Robots at initial positions
        robot_coords: Dict[str, Tuple[float, float]] = {}
        if robots:
            # Group robots by initial zone so multi-robot fleets cluster neatly inside zones
            zone_robot_counts: Dict[str, int] = {}
            for r in robots:
                z = r.initial_zone
                idx_in_zone = zone_robot_counts.get(z, 0)
                zone_robot_counts[z] = idx_in_zone + 1

                init_center = self.warehouse.get_zone_center(z)
                col = idx_in_zone % 3
                row = idx_in_zone // 3
                rx = init_center.x - 2.0 + (col * 2.0)
                ry = init_center.y + 0.8 - (row * 1.8)
                robot_coords[r.id] = (rx, ry)

                _, color = self.ROBOT_ICONS.get(r.robot_type, ("●", "#0F172A"))
                ax.plot(rx, ry, marker="o", markersize=13, color=color, markeredgecolor="#0F172A", markeredgewidth=1.2)
                ax.text(
                    rx, ry,
                    r.id[-3:],
                    ha="center", va="center",
                    color="white",
                    fontsize=6.0,
                    fontweight="bold"
                )
                ax.text(
                    rx, ry - 0.8,
                    f"{r.name}\n({r.robot_type.value})",
                    ha="center", va="top",
                    fontsize=5.5,
                    color="#0F172A",
                    fontweight="bold"
                )

        # 3. Draw Tasks
        task_coords: Dict[str, Tuple[float, float]] = {}
        if tasks:
            for i, task in enumerate(tasks):
                center = self.warehouse.get_zone_center(task.zone)
                offset_x = (i % 2 - 0.5) * 1.6
                offset_y = (i // 2 - 0.5) * 1.6
                tx = center.x + offset_x
                ty = center.y + offset_y
                task_coords[task.id] = (tx, ty)

                ax.plot(tx, ty, marker="s", markersize=13, color="#DC2626", markeredgecolor="#7F1D1D", markeredgewidth=1.2)
                ax.text(
                    tx, ty,
                    task.id,
                    ha="center", va="center",
                    color="white",
                    fontsize=7,
                    fontweight="bold"
                )
                ax.text(
                    tx, ty - 1.1,
                    task.name,
                    ha="center", va="top",
                    fontsize=6.5,
                    color="#991B1B",
                    fontweight="bold"
                )

        # 4. Draw Allocations (dispatch vectors) if provided
        if allocations and robots:
            for task_id, coalition in allocations.items():
                if task_id in task_coords:
                    tx, ty = task_coords[task_id]
                    for member in coalition.members:
                        if member.id in robot_coords:
                            rx, ry = robot_coords[member.id]
                            # Draw clean directed connection arrow from robot to task
                            arrow = patches.FancyArrowPatch(
                                (rx, ry), (tx, ty),
                                connectionstyle="arc3,rad=0.15",
                                arrowstyle="-|>",
                                mutation_scale=11,
                                color="#2563EB" if member.robot_type == RobotType.FORKLIFT else (
                                    "#7C3AED" if member.robot_type == RobotType.DRONE else "#059669"
                                ),
                                alpha=0.7,
                                linewidth=1.6,
                                linestyle="--"
                            )
                            ax.add_patch(arrow)

        return robot_coords, task_coords

    def render_allocation_map(
        self,
        robots: List[Robot],
        tasks: List[Task],
        allocations: Dict[str, Coalition],
        satisfaction_scores: Optional[Dict[str, float]] = None,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Render spatial warehouse map with allocated coalition dispatches and capability tags."""
        fig, ax = plt.subplots(figsize=(14, 9))
        robot_coords, task_coords = self._render_map_on_axis(ax, robots=robots, tasks=tasks, allocations=allocations)

        # Draw capability tags near tasks using exact task coordinates
        for idx, task in enumerate(tasks):
            if task.id in allocations and task.id in task_coords:
                c = allocations[task.id]
                score = satisfaction_scores.get(task.id, 1.0) if satisfaction_scores else 1.0
                tx, ty = task_coords[task.id]
                badge = f"Task {task.id}: {c}\nS = {score:.2f} ({'PASS' if score == 1.0 else 'PARTIAL'})"
                
                # Alternate badge position vertically or horizontally if tasks are clustered
                badge_offset_y = 1.35 if (idx % 2 == 0) else 1.85
                ax.text(
                    tx, ty + badge_offset_y,
                    badge,
                    ha="center", va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#ECFDF5" if score == 1.0 else "#FFFBEB",
                              edgecolor="#10B981" if score == 1.0 else "#F59E0B", linewidth=1.2)
                )

        ax.set_title("E-ITAGS Coalition Allocation & Spatial Dispatch Map", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("X (meters)", fontsize=10)
        ax.set_ylabel("Y (meters)", fontsize=10)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Saved allocation map to: {save_path}")

        if show:
            plt.show()

        return fig

    def render_satisfaction_matrix(
        self,
        coalitions: List[Coalition],
        tasks: List[Task],
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Render heatmap matrix of TraitEngine.compute_trait_satisfaction(C_k, t_j)."""
        height = max(6.5, min(18.0, len(coalitions) * 0.32))
        fig, ax = plt.subplots(figsize=(11, height))
        self._render_matrix_on_axis(ax, coalitions, tasks)

        ax.set_title("TraitEngine: Coalition × Task Satisfaction Matrix S(C_k, t_j)", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Saved satisfaction matrix heatmap to: {save_path}")

        if show:
            plt.show()

        return fig

    def _render_matrix_on_axis(
        self,
        ax: plt.Axes,
        coalitions: List[Coalition],
        tasks: List[Task],
    ) -> np.ndarray:
        """Helper to render satisfaction matrix heatmap on given axis."""
        n_c = len(coalitions)
        n_t = len(tasks)
        matrix = np.zeros((n_c, n_t))

        for i, c in enumerate(coalitions):
            for j, t in enumerate(tasks):
                score, _ = TraitEngine.compute_trait_satisfaction(c, t)
                matrix[i, j] = score

        # Color map: Red (0.0 incapable), Amber (0.5-0.9 partial), Emerald Green (1.0 capable)
        from matplotlib.colors import LinearSegmentedColormap
        colors = ["#FEE2E2", "#FEF3C7", "#BBF7D0", "#22C55E"]
        cmap = LinearSegmentedColormap.from_list("satisfaction_cmap", colors, N=256)

        im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

        cell_font = max(5.0, min(8.5, 90.0 / n_c))
        ytick_font = max(5.0, min(8.0, 95.0 / n_c))

        # Cell annotations
        for i in range(n_c):
            for j in range(n_t):
                val = matrix[i, j]
                if val == 1.0:
                    text = f"✓ {val:.2f}"
                    color = "#14532D"
                elif val > 0.0:
                    text = f"~ {val:.2f}"
                    color = "#78350F"
                else:
                    text = f"✗ {val:.2f}"
                    color = "#7F1D1D"

                ax.text(j, i, text, ha="center", va="center", fontsize=cell_font, fontweight="bold", color=color)

        ax.set_xticks(np.arange(n_t))
        ax.set_yticks(np.arange(n_c))
        ax.set_xticklabels([f"{t.id}\n{t.name[:14]}" for t in tasks], fontsize=8, fontweight="bold")
        ax.set_yticklabels([str(c) for c in coalitions], fontsize=ytick_font)
        ax.set_xlabel("Tasks", fontsize=9, fontweight="bold")
        ax.set_ylabel("Coalitions C_k", fontsize=9, fontweight="bold")

        return matrix

    def render_realizability_chart(
        self,
        robots: List[Robot],
        query_payloads: Optional[List[float]] = None,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Visualizes realizable trait vectors vs continuous optimizer assumptions."""
        fig, ax = plt.subplots(figsize=(10, 5.5))
        self._render_realizability_on_axis(ax, robots, query_payloads)

        ax.set_title("E-ITAGS Trait Realizability vs Continuous Optimizer Space", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Saved realizability chart to: {save_path}")

        if show:
            plt.show()

        return fig

    def _render_realizability_on_axis(
        self,
        ax: plt.Axes,
        robots: List[Robot],
        query_payloads: Optional[List[float]] = None,
    ) -> None:
        """Helper to render realizability comparison plot on given axis."""
        if query_payloads is None:
            query_payloads = [0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0]

        realizable_pairs = TraitEngine.compute_realizable_trait_set(robots, max_size=3)
        realizable_payloads = set(float(c.aggregate_traits.payload_kg) for c, _ in realizable_pairs)

        y_positions = np.arange(len(query_payloads))
        for idx, p in enumerate(query_payloads):
            is_real = p in realizable_payloads
            bar_color = "#10B981" if is_real else "#EF4444"
            edge_color = "#047857" if is_real else "#B91C1C"

            ax.barh(idx, p, height=0.55, color=bar_color, edgecolor=edge_color, alpha=0.85, linewidth=1.2)

            status_text = "[REALIZABLE]" if is_real else "[UNREALIZABLE] (continuous solver delusion)"
            matched_coalitions = [c for c, _ in realizable_pairs if np.isclose(c.aggregate_traits.payload_kg, p)]
            detail = f" — formed by {matched_coalitions[0]}" if matched_coalitions else " — no physical robot combination"

            ax.text(
                p + 35, idx,
                f"{status_text}{detail}",
                va="center", ha="left",
                fontsize=7.5, fontweight="bold",
                color="#065F46" if is_real else "#991B1B"
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{p:.0f} kg" for p in query_payloads], fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Target Payload Capacity (kg)", fontsize=9, fontweight="bold")
        ax.set_ylabel("Query Trait Demands", fontsize=9, fontweight="bold")
        ax.set_xlim(0, max(query_payloads) * 1.55)
        ax.grid(axis="x", linestyle="--", alpha=0.3, color="#94A3B8")

    def render_trait_breakdown(
        self,
        task: Task,
        coalitions: List[Coalition],
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Render dimensional trait breakdown comparing a task's demands against coalitions."""
        fig, ax = plt.subplots(figsize=(10, 5.5))
        self._render_breakdown_on_axis(ax, task, coalitions)

        ax.set_title(f"Trait Dimension Synergy: Task {task.id} ('{task.name}')", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Saved trait breakdown chart to: {save_path}")

        if show:
            plt.show()

        return fig

    def _render_breakdown_on_axis(
        self,
        ax: plt.Axes,
        task: Task,
        coalitions: List[Coalition],
    ) -> None:
        """Helper to render bar chart of dimensional trait scores on given axis."""
        dimensions = ["payload", "lidar_2d", "camera_rgb", "vertical_lift", "narrow_access"]
        dim_labels = ["Payload", "2D LiDAR", "RGB Camera", "Mast Lift", "Narrow Access"]

        n_dims = len(dimensions)
        n_coals = len(coalitions)
        width = 0.8 / n_coals
        x = np.arange(n_dims)

        colors = ["#2563EB", "#7C3AED", "#059669", "#D97706", "#EC4899"]

        for i, c in enumerate(coalitions):
            score, breakdown = TraitEngine.compute_trait_satisfaction(c, task)
            scores = [breakdown.get(d, 0.0) for d in dimensions]
            offsets = x + (i - (n_coals - 1) / 2) * width
            bar = ax.bar(
                offsets, scores, width,
                label=f"{c} (Overall: {score:.2f})",
                color=colors[i % len(colors)],
                alpha=0.85,
                edgecolor="#0F172A",
                linewidth=0.8,
            )

        ax.axhline(1.0, color="#10B981", linestyle="--", linewidth=1.5, label="Target Requirement (100%)")
        ax.set_xticks(x)
        ax.set_xticklabels(dim_labels, fontsize=8.5, fontweight="bold")
        ax.set_ylabel("Satisfaction Ratio [0.0 - 1.0]", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.25)
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="#94A3B8")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    def render_comprehensive_dashboard(
        self,
        robots: List[Robot],
        tasks: List[Task],
        allocations: Optional[Dict[str, Coalition]] = None,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Renders an all-in-one 4-panel publication-quality research dashboard:
        - Panel 1 (Top-Left): Spatial Warehouse Map with Coalition Dispatches
        - Panel 2 (Top-Right): TraitEngine Satisfaction Matrix Heatmap
        - Panel 3 (Bottom-Left): Physical Realizability Space vs Continuous Demands
        - Panel 4 (Bottom-Right): Collaborative Trait Synergy Breakdown (T4 Collaborative Pallet Sort)
        """
        coalitions = TraitEngine.enumerate_all_coalitions(robots, max_size=3)

        # If no allocation map provided, select best capable coalition for each task
        if allocations is None:
            allocations = {}
            for t in tasks:
                best_c = None
                best_score = -1.0
                for c in coalitions:
                    s, _ = TraitEngine.compute_trait_satisfaction(c, t)
                    if s > best_score:
                        best_score = s
                        best_c = c
                if best_c:
                    allocations[t.id] = best_c

        fig, axs = plt.subplots(2, 2, figsize=(18, 12))

        # Panel 1: Floor Map & Dispatches
        axs[0, 0].set_title("(A) Warehouse Spatial Layout & Coalition Dispatches", fontsize=11, fontweight="bold", pad=8)
        self._render_map_on_axis(axs[0, 0], robots=robots, tasks=tasks, allocations=allocations)

        # Panel 2: Satisfaction Matrix Heatmap
        axs[0, 1].set_title("(B) TraitEngine: Coalition × Task Satisfaction S(C_k, t_j)", fontsize=11, fontweight="bold", pad=8)
        if len(coalitions) > 16:
            # For large fleets (e.g. 41 coalitions), show representative subset: single robots + allocated coalitions
            panel_coalitions = []
            for c in coalitions:
                if c.size == 1 or c in allocations.values():
                    if c not in panel_coalitions:
                        panel_coalitions.append(c)
            for c in coalitions:
                if len(panel_coalitions) >= 14:
                    break
                if c not in panel_coalitions:
                    panel_coalitions.append(c)
            self._render_matrix_on_axis(axs[0, 1], panel_coalitions, tasks)
        else:
            self._render_matrix_on_axis(axs[0, 1], coalitions, tasks)

        # Panel 3: Realizability Space
        axs[1, 0].set_title("(C) Realizability-Aware Trait Manifold vs Continuous Demands", fontsize=11, fontweight="bold", pad=8)
        self._render_realizability_on_axis(axs[1, 0], robots)

        # Panel 4: Collaborative Synergy Breakdown for Task T4 (or first multi-robot task)
        collab_task = next((t for t in tasks if t.id == "T4"), tasks[0])
        comp_coalitions = []
        for c in coalitions:
            if c.size == 1 and len(comp_coalitions) < 3:
                comp_coalitions.append(c)
            elif any(r.robot_type == RobotType.FORKLIFT for r in c.members) and any(r.robot_type == RobotType.DRONE for r in c.members) and c.size == 2:
                if c not in comp_coalitions:
                    comp_coalitions.append(c)
                    break

        axs[1, 1].set_title(f"(D) Trait Synergy: Task {collab_task.id} ('{collab_task.name}')", fontsize=11, fontweight="bold", pad=8)
        self._render_breakdown_on_axis(axs[1, 1], collab_task, comp_coalitions)

        plt.suptitle("E-ITAGS Trait Reasoning & Warehouse Visualization Dashboard", fontsize=15, fontweight="bold", y=0.995)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved comprehensive research dashboard to: {save_path}")

        if show:
            plt.show()

        return fig

    def render_motion_paths(
        self,
        paths: Dict[str, Any],
        robots: Optional[List[Robot]] = None,
        tasks: Optional[List[Task]] = None,
        save_path: Optional[str | Path] = None,
        title: str = "A* Motion Planning & Spatial Trajectories",
        show: bool = False,
    ) -> plt.Figure:
        """Render warehouse floor map overlaid with planned A* robot paths."""
        fig, ax = plt.subplots(figsize=(13, 8.5))
        self._render_map_on_axis(ax, robots=robots, tasks=tasks)

        # Draw warehouse obstacles if any
        if self.warehouse.obstacles:
            for ox, oy in self.warehouse.obstacles:
                rect = patches.Rectangle(
                    (ox, oy), 1.0, 1.0,
                    facecolor="#334155", edgecolor="#0F172A", alpha=0.85
                )
                ax.add_patch(rect)

        # Path colors per robot archetype
        path_styles = {
            RobotType.FORKLIFT: {"color": "#2563EB", "linestyle": "-", "linewidth": 2.6, "label_prefix": "Forklift"},
            RobotType.DRONE: {"color": "#7C3AED", "linestyle": "--", "linewidth": 2.2, "label_prefix": "Drone (Aerial)"},
            RobotType.IW_HUB: {"color": "#059669", "linestyle": "-.", "linewidth": 2.4, "label_prefix": "IW Hub"},
        }

        for label, path_res in paths.items():
            if not path_res.success or not path_res.path:
                continue

            r_type = path_res.robot_type
            style = path_styles.get(r_type, {"color": "#E11D48", "linestyle": "-", "linewidth": 2.0, "label_prefix": label})

            xs = [loc.x + 0.5 for loc in path_res.path]
            ys = [loc.y + 0.5 for loc in path_res.path]

            legend_lbl = f"{label} ({style['label_prefix']}): {path_res.distance:.1f}m, {path_res.travel_time:.1f}s"
            ax.plot(
                xs, ys,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=0.9,
                label=legend_lbl,
                zorder=6,
            )

            # Start and End markers
            ax.scatter(xs[0], ys[0], color=style["color"], s=70, marker="o", edgecolors="white", linewidth=1.5, zorder=7)
            ax.scatter(xs[-1], ys[-1], color=style["color"], s=90, marker="*", edgecolors="white", linewidth=1.5, zorder=7)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("X (meters)", fontsize=10)
        ax.set_ylabel("Y (meters)", fontsize=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9.5)

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved motion paths visualization to: {save_path}")

        if show:
            plt.show()

        return fig
