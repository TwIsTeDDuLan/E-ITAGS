#!/usr/bin/env python3
"""Phase 4 & 5 Interactive Demonstration: Dual-Objective Heuristics & Interleaved ITAGS Graph Search.

Demonstrates:
1. Normalized Allocation Cost (NAC) and Time Budget Overrun (TBO) heuristic calculations.
2. Interleaved ITAGS Graph Search:
   - Simultaneous coalition assignment and timeline scheduling.
   - Most-constrained-task-first dynamic ordering.
   - Real-time branch pruning (zone restrictions, deadlines, and schedule conflicts).
3. Tradeoff Weight (w) Sweep across the Pareto frontier:
   - w = 1.0: Efficacy-first (prioritizes trait matching over time).
   - w = 0.0: Schedule-first (prioritizes completion speed).
   - w = 0.5: Balanced optimal compromise.
4. Exporting Search Tree and Pareto Front Visualizations.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import yaml

from src.models.warehouse import WarehouseGrid
from src.models.robot import Robot, RobotType
from src.models.task import Task, TaskRequirements
from src.core.trait_engine import TraitEngine
from src.core.motion_planner import MotionPlanner
from src.core.heuristics import HeuristicEngine
from src.core.graph_search import ITAGSSearch, SearchResult
from src.utils.display import TerminalDisplay
from src.visualization.schedule_viz import ScheduleVisualizer
from src.visualization.search_tree_viz import SearchTreeVisualizer


def load_scenario(config_path: Path):
    """Load warehouse scenario from YAML."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    robots: List[Robot] = []
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

    tasks: List[Task] = []
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
            deadline_seconds=float(t_data["deadline_seconds"]) if "deadline_seconds" in t_data else None,
        )
        tasks.append(task)

    return data["scenario_name"], float(data.get("time_budget_seconds", 3600.0)), robots, tasks


def main():
    TerminalDisplay.print_header(
        "E-ITAGS PHASE 4 & 5 DEMONSTRATION",
        "Dual-Objective Heuristics (NAC+TBO) and Interleaved Graph Search"
    )

    config_path = Path(__file__).parent / "configs" / "warehouse_scenario.yaml"
    scenario_name, time_budget, robots, tasks = load_scenario(config_path)

    print(f"\n📂 Loaded Scenario: '{scenario_name}'")
    print(f"   Fleet: {len(robots)} Robots | Tasks: {len(tasks)} | Mission Budget: {time_budget:.0f}s\n")
    TerminalDisplay.display_robots(robots)
    TerminalDisplay.display_tasks(tasks)

    # 1. Initialize Motion Planner & Search Engine
    warehouse = WarehouseGrid()
    warehouse.add_default_warehouse_racks()
    planner = MotionPlanner(warehouse)

    searcher = ITAGSSearch(
        robots=robots,
        tasks=tasks,
        warehouse=warehouse,
        motion_planner=planner,
        time_budget=time_budget,
        weight=0.5,
        max_coalition_size=2,
    )

    # 2. Heuristic Engine Inspection (Phase 4)
    TerminalDisplay.print_section("1. PHASE 4: DUAL-OBJECTIVE HEURISTICS (NAC + TBO)")
    print("Precomputed Optimistic Upper Bounds (Max achievable satisfaction per task):")
    for t_id, bound in searcher.task_upper_bounds.items():
        task = searcher.task_map[t_id]
        print(f"  • Task {t_id:<3} ('{task.name:<26}') in {task.zone:<13} -> Max Upper Bound: {bound:.2f}")

    print("\nHeuristic Formula:")
    print("  h(u, w) = w · NAC(u) + (1 - w) · TBO(u)")
    print("  where NAC = 1 - (Efficacy_assigned + UpperBounds_unassigned) / |Tasks|")
    print(f"  where TBO = max(0, makespan - {time_budget:.0f}s) / {time_budget:.0f}s\n")

    # 3. Full ITAGS Graph Search Execution (Phase 5)
    TerminalDisplay.print_section("2. PHASE 5: INTERLEAVED ITAGS GRAPH SEARCH (Balanced w = 0.5)")
    res_balanced = searcher.solve(weight=0.5, verbose=True)

    print(f"\nSearch Outcome: {'✅ SUCCESS' if res_balanced.solution_found else '❌ FAILED'}")
    print(f"  • Total Nodes Expanded:   {res_balanced.stats.nodes_expanded}")
    print(f"  • Total Nodes Generated:  {res_balanced.stats.nodes_generated}")
    print(f"  • Pruned (Deadlines):     {res_balanced.stats.nodes_pruned_deadline}")
    print(f"  • Pruned (Schedule):      {res_balanced.stats.nodes_pruned_schedule}")
    print(f"  • Pruned (Duplicate):     {res_balanced.stats.nodes_pruned_closed}")
    print(f"  • Search Computation:     {res_balanced.stats.search_duration_sec * 1000.0:.2f} ms")
    print(f"  • Mission Makespan:       {res_balanced.makespan:.1f}s ({res_balanced.makespan/60:.1f} min)")
    print(f"  • Mean Mission Efficacy:  {res_balanced.mean_efficacy * 100.0:.1f}% (NAC = {res_balanced.nac:.3f})")
    print(f"  • Budget Overrun (TBO):   {res_balanced.tbo:.3f}\n")

    # Display Solution Path Trajectory
    print("Step-by-Step Search Decision Sequence:")
    print(f"{'Step':<5} | {'Task ID':<8} | {'Assigned Coalition':<28} | {'Zone':<13} | {'f-cost':<8} | {'NAC':<6} | {'Makespan'}")
    print("-" * 88)
    for step_idx, node in enumerate(res_balanced.solution_path):
        if step_idx == 0:
            print(f"{'0':<5} | {'[Root]':<8} | {'(Initial Fleet @ Staging)':<28} | {'staging':<13} | {node.f_cost:>6.3f} | {node.nac:>5.3f} | {node.schedule.get_makespan():>7.1f}s")
        else:
            task = searcher.task_map[node.expanded_task_id]
            print(
                f"{step_idx:<5} | {node.expanded_task_id:<8} | "
                f"{str(node.expanded_coalition):<28} | {task.zone:<13} | "
                f"{node.f_cost:>6.3f} | {node.nac:>5.3f} | {node.schedule.get_makespan():>7.1f}s"
            )
    print("-" * 88 + "\n")

    # Display Mission Timeline
    ScheduleVisualizer.print_timeline_table(res_balanced.schedule)

    # 4. Pareto Frontier Sweep across Tradeoff Weights
    TerminalDisplay.print_section("3. PARETO OPTIMIZATION FRONTIER: EFFICACY VS. MAKESPAN (w Sweep)")
    print("Testing how shifting w changes the search priority:\n")

    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    pareto_results = searcher.sweep_pareto_front(weights=weights)

    print(f"{'Weight (w)':<12} | {'Search Mode':<20} | {'Makespan':<12} | {'Mean Efficacy':<15} | {'Expanded':<10} | {'Search Time'}")
    print("-" * 85)
    for r in pareto_results:
        mode = "Schedule-First" if r.weight == 0.0 else ("Efficacy-First" if r.weight == 1.0 else "Balanced")
        print(
            f"{r.weight:<12.2f} | {mode:<20} | {r.makespan:>8.1f}s | "
            f"{r.mean_efficacy * 100.0:>12.1f}% | {r.stats.nodes_expanded:>8} | {r.stats.search_duration_sec*1000:>8.2f}ms"
        )
    print("-" * 85 + "\n")

    # 5. Export Visual Artifacts
    TerminalDisplay.print_section("4. EXPORTING PHASE 4 & 5 VISUALIZATION ARTIFACTS")
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    tree_viz = SearchTreeVisualizer()
    sched_viz = ScheduleVisualizer()

    # 5.1 Search Tree Graph
    tree_png = output_dir / "phase5_search_tree.png"
    tree_viz.render_search_tree(res_balanced, save_path=tree_png)

    # 5.2 Pareto Frontier Curve
    pareto_png = output_dir / "phase5_pareto_front.png"
    tree_viz.render_pareto_front(pareto_results, save_path=pareto_png)

    # 5.3 Optimal Mission Gantt Chart
    gantt_png = output_dir / "phase5_optimal_gantt.png"
    sched_viz.render_gantt(
        schedule=res_balanced.schedule,
        save_path=gantt_png,
        title=f"E-ITAGS Optimal Schedule (w=0.5, Makespan: {res_balanced.makespan:.1f}s, Efficacy: {res_balanced.mean_efficacy*100:.0f}%)",
    )

    TerminalDisplay.print_header(
        "PHASE 4 & 5 DEMONSTRATION COMPLETE",
        f"Search trees, Pareto curves, and schedules exported to:\n  {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
