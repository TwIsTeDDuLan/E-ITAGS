"""Terminal visualization and inspection utilities for E-ITAGS."""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from ..models.robot import Robot, TRAIT_NAMES
from ..models.task import Task
from ..models.coalition import Coalition
from ..models.warehouse import WarehouseGrid
from ..core.trait_engine import TraitEngine


class TerminalDisplay:
    """Rich terminal output for making all E-ITAGS data and steps transparent."""

    @staticmethod
    def print_header(title: str, subtitle: str = "") -> None:
        width = 80
        print("\n" + "=" * width)
        print(f"  {title.upper()}")
        if subtitle:
            print(f"  {subtitle}")
        print("=" * width)

    @staticmethod
    def print_section(title: str) -> None:
        width = 80
        print("\n" + "-" * width)
        print(f"  ▶ {title}")
        print("-" * width)

    @staticmethod
    def display_robots(robots: List[Robot]) -> None:
        """Display the robot roster and trait vectors in a clean table."""
        TerminalDisplay.print_section("ROBOT FLEET & TRAIT VECTORS (q_i)")
        print(f"{'Robot Name':<16} | {'Type':<10} | {'Payload':<8} | {'2D LiDAR':<9} | {'RGB Cam':<8} | {'Mast Lift':<9} | {'Narrow':<7} | {'Speed':<6}")
        print("-" * 92)
        for r in robots:
            t = r.traits
            cam_str = f"{t.has_camera_rgb:.1f}"
            lift_str = "YES" if t.vertical_lift > 0 else "NO"
            lidar_str = "YES (360°)" if t.has_lidar_2d > 0 else "NO"
            narrow_str = "YES" if t.narrow_space_access > 0 else "NO"
            print(f"{r.name:<16} | {r.robot_type.value:<10} | {t.payload_kg:>6.0f}kg | {lidar_str:<9} | {cam_str:<8} | {lift_str:<9} | {narrow_str:<7} | {t.speed_mps:>4.1f}m/s")
        print("-" * 92)

    @staticmethod
    def display_tasks(tasks: List[Task]) -> None:
        """Display tasks and their trait requirements."""
        TerminalDisplay.print_section("WAREHOUSE TASKS & TRAIT DEMANDS")
        print(f"{'Task ID & Name':<30} | {'Zone':<12} | {'Dur(s)':<7} | {'Min Payload':<11} | {'LiDAR':<7} | {'Cam':<6} | {'Lift':<6} | {'Narrow':<6}")
        print("-" * 96)
        for t in tasks:
            req = t.requirements
            cam_str = f">={req.min_has_camera_rgb:.1f}" if req.min_has_camera_rgb > 0 else "—"
            lidar_str = "REQ" if req.min_has_lidar_2d > 0 else "—"
            lift_str = "REQ" if req.min_vertical_lift > 0 else "—"
            narrow_str = "REQ" if req.min_narrow_space_access > 0 else "—"
            payload_str = f">={req.min_payload_kg:.0f}kg" if req.min_payload_kg > 0 else "—"
            print(f"{t.id + ': ' + t.name:<30} | {t.zone:<12} | {t.duration_seconds:>5.0f}s | {payload_str:<11} | {lidar_str:<7} | {cam_str:<6} | {lift_str:<6} | {narrow_str:<6}")
        print("-" * 96)

    @staticmethod
    def display_coalitions(coalitions: List[Coalition]) -> None:
        """Display enumerated coalitions and their aggregated traits."""
        TerminalDisplay.print_section("ENUMERATED COALITIONS & AGGREGATE TRAITS Q(C)")
        print(f"{'Coalition C_k':<32} | {'Size':<4} | {'Payload':<8} | {'2D LiDAR':<9} | {'RGB Cam':<8} | {'Mast Lift':<9} | {'Narrow':<7} | {'Speed':<6}")
        print("-" * 96)
        for c in coalitions:
            q = c.aggregate_traits
            cam_str = f"{q.has_camera_rgb:.1f}"
            lift_str = "YES" if q.vertical_lift > 0 else "NO"
            lidar_str = "YES" if q.has_lidar_2d > 0 else "NO"
            narrow_str = "YES" if q.narrow_space_access > 0 else "NO"
            print(f"{str(c):<32} | {c.size:<4} | {q.payload_kg:>6.0f}kg | {lidar_str:<9} | {cam_str:<8} | {lift_str:<9} | {narrow_str:<7} | {q.speed_mps:>4.1f}m/s")
        print("-" * 96)

    @staticmethod
    def display_satisfaction_matrix(coalitions: List[Coalition], tasks: List[Task]) -> None:
        """Display the Coalition × Task feasibility and trait satisfaction matrix."""
        TerminalDisplay.print_section("COALITION × TASK SATISFACTION MATRIX S(C_k, t_j)")
        
        # Header with task IDs
        task_header = " | ".join(f"{t.id:^10}" for t in tasks)
        print(f"{'Coalition':<30} | {task_header}")
        print("-" * (32 + 13 * len(tasks)))

        for c in coalitions:
            row_items = []
            for t in tasks:
                score, _ = TraitEngine.compute_trait_satisfaction(c, t)
                if score >= 0.999:
                    status = f"✓ {score:.2f}"
                elif score > 0.0:
                    status = f"~ {score:.2f}"
                else:
                    status = "✗ 0.00"
                row_items.append(f"{status:^10}")
            print(f"{str(c):<30} | " + " | ".join(row_items))
        print("-" * (32 + 13 * len(tasks)))
        print("Legend: [✓] Satisfies all requirements | [~] Partial match | [✗] Hard constraint violation")

    @staticmethod
    def display_warehouse_grid(
        warehouse: WarehouseGrid,
        robot_positions: Any = None,
        task_positions: Any = None,
        path: Any = None,
        show_legend: bool = True,
        show_coordinates: bool = True,
        use_color: bool = True,
    ) -> None:
        """Display an ASCII/Unicode 2D floor map of the warehouse grid."""
        warehouse.print_grid(
            robot_positions=robot_positions,
            task_positions=task_positions,
            path=path,
            show_legend=show_legend,
            show_coordinates=show_coordinates,
            use_color=use_color,
        )

