"""Visualization of Multi-Robot Task Schedules (Gantt charts, timelines, resource utilization)."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from ..models.robot import Robot, RobotType
from ..core.scheduler import Schedule, ScheduleAssignment, RobotTimelineEntry


class ScheduleVisualizer:
    """Renders publication-quality Gantt charts and utilization breakdown diagrams for E-ITAGS schedules."""

    TASK_COLORS = [
        "#3B82F6",  # Blue
        "#10B981",  # Emerald
        "#8B5CF6",  # Violet
        "#EC4899",  # Pink
        "#F97316",  # Orange
        "#06B6D4",  # Cyan
        "#EAB308",  # Yellow
        "#6366F1",  # Indigo
    ]

    TRAVEL_COLOR = "#F59E0B"  # Warm Amber for transit segments
    IDLE_COLOR = "#F1F5F9"    # Soft light slate

    def __init__(self):
        pass

    def render_gantt(
        self,
        schedule: Schedule,
        save_path: Optional[str | Path] = None,
        title: str = "Multi-Robot Warehouse Mission Schedule (Gantt Chart)",
        show: bool = False,
    ) -> plt.Figure:
        """Render an intuitive, high-clarity Gantt chart of the complete multi-robot schedule.

        Visual features:
        - Y-axis: Each robot in the fleet with its archetype icon and ID.
        - X-axis: Mission timeline in seconds and minutes.
        - Orange/amber hatched blocks: Robot transit and zone repositioning.
        - Colored blocks: Active task execution (color-coded per task).
        - Vertical dashed lines: Mission makespan and time budget.
        """
        robots = list(schedule.robots.values())
        n_robots = len(robots)
        makespan = schedule.get_makespan()
        max_x = max(makespan * 1.15, schedule.time_budget * 1.05, 60.0)

        fig_height = max(5.0, 1.2 * n_robots + 2.0)
        fig, ax = plt.subplots(figsize=(14, fig_height))

        # Assign unique colors to tasks
        task_color_map: Dict[str, str] = {}
        for idx, assignment in enumerate(schedule.assignments):
            t_id = assignment.task.id
            if t_id not in task_color_map:
                task_color_map[t_id] = self.TASK_COLORS[len(task_color_map) % len(self.TASK_COLORS)]

        bar_height = 0.55
        y_ticks = []
        y_labels = []

        # Render rows from top to bottom
        for row_idx, robot in enumerate(reversed(robots)):
            y_pos = row_idx
            y_ticks.append(y_pos)

            type_code = "FL" if robot.robot_type == RobotType.FORKLIFT else ("DR" if robot.robot_type == RobotType.DRONE else "IW")
            y_labels.append(f"[{type_code}] {robot.name}\n({robot.id})")

            # Draw background idle bar
            ax.barh(
                y_pos, max_x, height=bar_height,
                left=0, color=self.IDLE_COLOR, edgecolor="#CBD5E1",
                linewidth=0.8, alpha=0.5, zorder=1
            )

            entries = schedule.robot_timelines.get(robot.id, [])
            for entry in entries:
                # 1. Travel segment
                if entry.travel_duration > 0.01:
                    ax.barh(
                        y_pos, entry.travel_duration, height=bar_height,
                        left=entry.travel_start, color=self.TRAVEL_COLOR,
                        edgecolor="#D97706", linewidth=1.2, hatch="//",
                        alpha=0.85, zorder=3
                    )
                    # Text label inside travel bar if wide enough
                    if entry.travel_duration > (max_x * 0.04):
                        label = f"→ {entry.destination_zone}"
                        ax.text(
                            entry.travel_start + entry.travel_duration / 2, y_pos,
                            label, ha="center", va="center",
                            fontsize=7.5, fontweight="bold", color="#78350F", zorder=4
                        )

                # 2. Execution segment
                task_color = task_color_map.get(entry.task_id, "#10B981")
                ax.barh(
                    y_pos, entry.execution_duration, height=bar_height,
                    left=entry.execution_start, color=task_color,
                    edgecolor="#1E293B", linewidth=1.2,
                    alpha=0.9, zorder=3
                )
                # Label inside execution bar
                exec_label = f"{entry.task_id}: {entry.task_name}"
                if entry.execution_duration > (max_x * 0.05):
                    ax.text(
                        entry.execution_start + entry.execution_duration / 2, y_pos,
                        exec_label, ha="center", va="center",
                        fontsize=8.0, fontweight="bold", color="white", zorder=4
                    )

        # Vertical line for mission makespan
        ax.axvline(
            makespan, color="#DC2626", linestyle="--", linewidth=2.0,
            label=f"Makespan: {makespan:.1f}s ({makespan/60:.1f}m)", zorder=5
        )

        # Vertical line for time budget
        if schedule.time_budget > 0 and schedule.time_budget <= max_x:
            ax.axvline(
                schedule.time_budget, color="#475569", linestyle=":", linewidth=1.8,
                label=f"Time Budget: {schedule.time_budget:.1f}s", zorder=5
            )

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=10, fontweight="bold")
        ax.set_xlim(0, max_x)
        ax.set_xlabel("Timeline (Seconds)", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
        ax.grid(axis="x", linestyle=":", alpha=0.6, zorder=0)

        # Secondary top x-axis for minutes
        ax_top = ax.secondary_xaxis('top', functions=(lambda s: s / 60.0, lambda m: m * 60.0))
        ax_top.set_xlabel("Timeline (Minutes)", fontsize=10, fontweight="bold", labelpad=8)

        # Legend entries
        legend_handles = [
            patches.Patch(facecolor=self.TRAVEL_COLOR, hatch="//", edgecolor="#D97706", label="Robot Transit / Repositioning"),
            patches.Patch(facecolor="#10B981", edgecolor="#1E293B", label="Task Execution"),
            patches.Patch(facecolor=self.IDLE_COLOR, edgecolor="#CBD5E1", label="Idle"),
        ]
        # Add tasks to legend
        for t_id, col in task_color_map.items():
            legend_handles.append(patches.Patch(facecolor=col, edgecolor="#1E293B", label=f"Task {t_id}"))

        ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.0, -0.15), ncol=4, fontsize=9)

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def render_robot_utilization(
        self,
        schedule: Schedule,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Render stacked horizontal bar chart showing each robot's utilization percentages."""
        robots = list(schedule.robots.values())
        util = schedule.get_robot_utilization()

        fig, ax = plt.subplots(figsize=(10, max(4.0, 0.8 * len(robots) + 1.5)))

        names = [f"{r.name} ({r.id})" for r in reversed(robots)]
        exec_pcts = [util[r.id]["exec_pct"] for r in reversed(robots)]
        travel_pcts = [util[r.id]["travel_pct"] for r in reversed(robots)]
        idle_pcts = [util[r.id]["idle_pct"] for r in reversed(robots)]

        y_pos = np.arange(len(robots))
        bar_height = 0.55

        # Stacked bars
        b_exec = ax.barh(y_pos, exec_pcts, height=bar_height, color="#10B981", label="Execution (%)")
        b_travel = ax.barh(y_pos, travel_pcts, height=bar_height, left=exec_pcts, color=self.TRAVEL_COLOR, hatch="//", label="Transit (%)")
        left_idle = [e + t for e, t in zip(exec_pcts, travel_pcts)]
        b_idle = ax.barh(y_pos, idle_pcts, height=bar_height, left=left_idle, color="#CBD5E1", label="Idle (%)")

        # Percentage annotations inside bars
        for idx in range(len(robots)):
            e = exec_pcts[idx]
            t = travel_pcts[idx]
            i = idle_pcts[idx]
            if e >= 8:
                ax.text(e / 2, idx, f"{e:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
            if t >= 8:
                ax.text(e + t / 2, idx, f"{t:.0f}%", ha="center", va="center", color="#78350F", fontweight="bold", fontsize=9)
            if i >= 8:
                ax.text(e + t + i / 2, idx, f"{i:.0f}%", ha="center", va="center", color="#334155", fontweight="bold", fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Robot Time Allocation (%)", fontsize=11, fontweight="bold")
        ax.set_title("Fleet Workload & Utilization Breakdown", fontsize=13, fontweight="bold", pad=12)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9)
        ax.grid(axis="x", linestyle=":", alpha=0.5)

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    @staticmethod
    def print_timeline_table(schedule: Schedule) -> None:
        """Print an organized chronological timeline table to the console."""
        print("\n" + "=" * 90)
        print(f"  CHRONOLOGICAL MULTI-ROBOT MISSION TIMELINE (Makespan: {schedule.get_makespan():.1f}s)")
        print("=" * 90)
        print(f"{'Task ID':<8} | {'Task Name':<26} | {'Coalition':<25} | {'Window (Start -> End)':<22} | {'Zone'}")
        print("-" * 90)

        for assign in schedule.assignments:
            win_str = f"{assign.execution_start:>6.1f}s -> {assign.execution_end:>6.1f}s"
            print(
                f"{assign.task.id:<8} | {assign.task.name:<26} | "
                f"{str(assign.coalition):<25} | {win_str:<22} | {assign.task.zone}"
            )
        print("-" * 90 + "\n")
