#!/usr/bin/env python3
"""Multi-Robot Fleet Demonstration: Testing TraitEngine & WarehouseVisualizer with Multiple Robots of Each Type.

Scenario:
- 2x Forklifts: Forklift-Alpha (R_FL1), Forklift-Beta (R_FL2) [1000kg, Mast Lift, 2D LiDAR]
- 2x Drones: Drone-SkyEye (R_DR1), Drone-HawkEye (R_DR2) [0kg, RGB Camera 1.0, 5.0m/s]
- 2x IW Hubs: IWHub-FloorRunner (R_IW1), IWHub-FloorGlide (R_IW2) [1000kg, Underride RGB 0.5, Narrow Access]

Total Fleet: 6 robots
Combinatorics: C(6,1) + C(6,2) + C(6,3) = 6 + 15 + 20 = 41 Coalitions
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import json
import numpy as np

from src.models.robot import Robot, RobotType, RobotTraits, TRAIT_NAMES, ZONE_ACCESS
from src.models.task import Task, TaskRequirements
from src.models.coalition import Coalition
from src.models.warehouse import WarehouseGrid
from src.core.trait_engine import TraitEngine
from src.utils.display import TerminalDisplay
from src.visualization.warehouse_viz import WarehouseVisualizer
from demo_trait_engine_and_viz import export_interactive_html


def load_multi_scenario(config_path: Path):
    """Load the multi-robot scenario config."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    robots: list[Robot] = []
    for r_data in data["robots"]:
        r_type = r_data["type"]
        if r_type == "forklift":
            r = Robot.create_forklift(r_data["id"], r_data["name"], r_data.get("initial_zone", "staging"))
        elif r_type == "drone":
            r = Robot.create_drone(r_data["id"], r_data["name"], r_data.get("initial_zone", "staging"))
        elif r_type == "iw_hub":
            r = Robot.create_iw_hub(r_data["id"], r_data["name"], r_data.get("initial_zone", "staging"))
        else:
            raise ValueError(f"Unknown robot type: {r_type}")
        robots.append(r)

    tasks: list[Task] = []
    for t_data in data["tasks"]:
        req_data = t_data["requirements"]
        req = TaskRequirements(
            min_payload_kg=float(req_data.get("min_payload_kg", 0.0)),
            min_has_lidar_2d=float(req_data.get("min_has_lidar_2d", 0.0)),
            min_has_camera_rgb=float(req_data.get("min_has_camera_rgb", 0.0)),
            min_vertical_lift=float(req_data.get("min_vertical_lift", 0.0)),
            min_narrow_space_access=float(req_data.get("min_narrow_space_access", 0.0)),
        )
        task = Task(
            id=t_data["id"],
            name=t_data["name"],
            zone=t_data["zone"],
            duration_seconds=float(t_data["duration_seconds"]),
            requirements=req,
            description=t_data.get("description", ""),
        )
        tasks.append(task)

    return data["scenario_name"], robots, tasks


def main():
    TerminalDisplay.print_header(
        "E-ITAGS MULTI-ROBOT FLEET DEMONSTRATION",
        "Testing TraitEngine & WarehouseVisualizer with 2x Forklifts, 2x Drones, 2x IW Hubs"
    )

    config_path = Path(__file__).parent / "configs" / "warehouse_scenario_multi.yaml"
    scenario_name, robots, tasks = load_multi_scenario(config_path)

    print(f"\n📂 Loaded Scenario: '{scenario_name}'")
    print(f"   Physical Fleet: {len(robots)} Robots (2 Forklifts, 2 Drones, 2 IW Hubs)")
    print(f"   Tasks to Allocate: {len(tasks)} (including 1,800kg Super-Heavy transfer)\n")

    # 1. Display Robot Fleet
    TerminalDisplay.display_robots(robots)

    # 1b. Display Zone Access Constraints
    TerminalDisplay.print_section("ZONE ACCESS CONSTRAINTS")
    print("Each robot type has physical constraints limiting which warehouse zones it can operate in:\n")
    print(f"{'Robot Type':<12} | {'Allowed Zones':<55} | {'Excluded Zones'}")
    print("-" * 95)
    all_zones = {"receiving", "sorting", "staging", "narrow_aisle", "shipping"}
    for rtype in RobotType:
        allowed = ZONE_ACCESS.get(rtype, frozenset())
        excluded = all_zones - allowed
        allowed_str = ", ".join(sorted(allowed)) if allowed else "(none)"
        excluded_str = ", ".join(sorted(excluded)) if excluded else "(none)"
        icon = "🟢" if excluded == set() else "🟡"
        print(f"{icon} {rtype.value:<10} | {allowed_str:<55} | {excluded_str}")
    print()

    # 2. Enumerate Coalitions
    TerminalDisplay.print_section("1. COALITION ENUMERATION (C(6,1) + C(6,2) + C(6,3))")
    coalitions = TraitEngine.enumerate_all_coalitions(robots, max_size=3)
    c_by_size = {1: [c for c in coalitions if c.size == 1],
                 2: [c for c in coalitions if c.size == 2],
                 3: [c for c in coalitions if c.size == 3]}

    print(f"Total Coalitions Enumerated: {len(coalitions)}")
    print(f"  • Size 1 (Individual robots): {len(c_by_size[1])} coalitions")
    print(f"  • Size 2 (Two-robot pairs)  : {len(c_by_size[2])} coalitions")
    print(f"  • Size 3 (Three-robot teams): {len(c_by_size[3])} coalitions\n")

    # Identify Distinct Trait Profiles (Equivalence Classes)
    unique_traits = {}
    for c in coalitions:
        vec_tuple = tuple(np.round(c.to_vector(), 2))
        if vec_tuple not in unique_traits:
            unique_traits[vec_tuple] = []
        unique_traits[vec_tuple].append(c)

    print(f"💡 Key Insight: The {len(coalitions)} coalitions collapse into {len(unique_traits)} distinct physical trait profiles!")
    print("   Example Equivalent Teams (Same capabilities formed by interchangeable robots):")
    for vec, equiv_coalitions in list(unique_traits.items())[:3]:
        names = [str(c) for c in equiv_coalitions]
        print(f"   - Trait Profile {vec[:3]}...: {len(equiv_coalitions)} equivalent teams -> {names[:2]}...")

    # 3. Realizability Analysis
    TerminalDisplay.print_section("2. REALIZABILITY ANALYSIS: EXACT REALIZABILITY vs SURPLUS COVERAGE")
    print("Testing whether trait vectors can be physically formed vs whether the fleet can cover the demand:")
    print(f"{'Target Payload':<16} | {'Exact (allow_surplus=False)':<30} | {'Surplus (allow_surplus=True)':<35}")
    print("-" * 88)

    test_payloads = [0.0, 100.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0]
    for p in test_payloads:
        target = RobotTraits(payload_kg=p)
        is_exact, match_exact = TraitEngine.is_realizable(target, robots, trait_subset=["payload_kg"], allow_surplus=False)
        is_surplus, match_surplus = TraitEngine.is_realizable(target, robots, trait_subset=["payload_kg"], allow_surplus=True)

        exact_msg = f"REAL: {str(match_exact)[:18]}" if is_exact else "UNREALIZABLE (discrete)"
        surplus_msg = f"SATISFIED: {str(match_surplus)[:20]}" if is_surplus else "EXCEEDS MAX FLEET (3000kg)"

        exact_icon = "✅" if is_exact else "❌"
        surplus_icon = "✅" if is_surplus else "❌"

        print(f"{p:>6.0f} kg         | {exact_icon} {exact_msg:<27} | {surplus_icon} {surplus_msg:<32}")

    print("-" * 88)

    # 4. Multi-Robot Task Allocation
    TerminalDisplay.print_section("3. MULTI-ROBOT TASK ALLOCATION & CONCURRENCY")
    allocations: dict[str, Coalition] = {}
    satisfaction_scores: dict[str, float] = {}
    used_robot_ids: set[str] = set()

    for task in tasks:
        best_coalition = None
        best_score = -1.0
        min_size = 999

        # Pre-filter coalitions: only those where ALL members can access the task zone
        zone_accessible = TraitEngine.filter_zone_accessible_coalitions(coalitions, task.zone)
        zone_filtered_count = len(coalitions) - len(zone_accessible)

        # Score remaining candidates, prefer disjoint (unused robots) for true concurrency
        candidates = []
        for c in zone_accessible:
            score, _ = TraitEngine.compute_trait_satisfaction(c, task)
            if score > 0.0:
                is_disjoint = not any(r_id in used_robot_ids for r_id in c.robot_ids)
                candidates.append((score, is_disjoint, c.size, c))

        # Sort: highest score, then disjoint (parallelism), then smallest team size
        candidates.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)

        if candidates:
            best_score, is_disjoint, _, best_coalition = candidates[0]
            allocations[task.id] = best_coalition
            satisfaction_scores[task.id] = best_score
            for r_id in best_coalition.robot_ids:
                used_robot_ids.add(r_id)

            status = "FULL MATCH (1.00)" if best_score == 1.0 else f"PARTIAL ({best_score:.2f})"
            zone_msg = f"(zone filtered {zone_filtered_count} coalitions)" if zone_filtered_count > 0 else ""
            print(f"  • Task {task.id:<3} ('{task.name:<28}') @ {task.zone:<13} -> Assigned: {str(best_coalition):<40} [{status}] {zone_msg}")
        else:
            zone_msg = f"({zone_filtered_count} coalitions eliminated by zone constraints)" if zone_filtered_count > 0 else ""
            print(f"  ✗ Task {task.id:<3} ('{task.name:<28}') @ {task.zone:<13} -> NO FEASIBLE COALITION {zone_msg}")

    # 5. Export Multi-Robot Visualizations
    TerminalDisplay.print_section("4. EXPORTING MULTI-ROBOT VISUALIZATION ARTIFACTS")
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    warehouse = WarehouseGrid()
    viz = WarehouseVisualizer(warehouse)

    # 5.1 Multi-Robot Floor Map
    map_png = output_dir / "multi_robot_map.png"
    viz.render(robots=robots, tasks=tasks, save_path=map_png)

    # 5.2 Allocation Map
    alloc_png = output_dir / "multi_robot_allocation_map.png"
    viz.render_allocation_map(
        robots=robots,
        tasks=tasks,
        allocations=allocations,
        satisfaction_scores=satisfaction_scores,
        save_path=alloc_png,
    )

    # 5.3 Satisfaction Heatmap
    matrix_png = output_dir / "multi_robot_satisfaction_matrix.png"
    viz.render_satisfaction_matrix(coalitions=coalitions, tasks=tasks, save_path=matrix_png)

    # 5.4 Realizability Chart
    real_png = output_dir / "multi_robot_realizability.png"
    viz.render_realizability_chart(robots=robots, query_payloads=test_payloads, save_path=real_png)

    # 5.5 Comprehensive Multi-Robot Dashboard
    dashboard_png = output_dir / "multi_robot_dashboard.png"
    viz.render_comprehensive_dashboard(robots=robots, tasks=tasks, allocations=allocations, save_path=dashboard_png)

    # 5.6 Interactive Web Dashboard
    html_path = output_dir / "multi_robot_interactive.html"
    export_interactive_html(warehouse, robots, tasks, coalitions, allocations, html_path)

    TerminalDisplay.print_header(
        "MULTI-ROBOT VERIFICATION COMPLETE",
        f"All 41 coalitions, trait aggregations, surplus checks & visualizations exported to:\n  {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
