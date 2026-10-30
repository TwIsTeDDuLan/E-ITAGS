#!/usr/bin/env python3
"""Phase 2 & 3 Interactive Demonstration: A* Motion Planner & Timeline Scheduler.

Demonstrates:
1. Grid-based A* Motion Planning:
   - Obstacle navigation and zone restriction avoidance (Forklift routing around narrow aisles).
   - Unrestricted 3D aerial direct flight for Drones at 5.0 m/s.
   - Specialized floor-level access for IW Hub AMRs.
2. Timeline Scheduling:
   - Synchronized rendezvous for multi-robot coalitions.
   - Interleaved travel times and task durations.
   - Strict deadline tracking and makespan computation.
   - Gantt chart export and fleet resource utilization.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import yaml
import math

from src.models.warehouse import WarehouseGrid, Location
from src.models.robot import Robot, RobotType, RobotTraits
from src.models.task import Task, TaskRequirements
from src.models.coalition import Coalition
from src.core.trait_engine import TraitEngine
from src.core.motion_planner import MotionPlanner, PathResult
from src.core.scheduler import Schedule, ScheduleResult
from src.utils.display import TerminalDisplay
from src.visualization.warehouse_viz import WarehouseVisualizer
from src.visualization.schedule_viz import ScheduleVisualizer


def load_demo_scenario(config_path: Path):
    """Load scenario from YAML."""
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

    return data["scenario_name"], robots, tasks


def main():
    TerminalDisplay.print_header(
        "E-ITAGS PHASE 2 & 3 DEMONSTRATION",
        "A* Motion Planning, Zone-Aware Routing, and Timeline Rendezvous Scheduling"
    )

    # 1. Load Scenario
    config_path = Path(__file__).parent / "configs" / "warehouse_scenario_multi.yaml"
    scenario_name, robots, tasks = load_demo_scenario(config_path)

    print(f"\n📂 Scenario: '{scenario_name}'")
    print(f"   Fleet: {len(robots)} Robots | Tasks: {len(tasks)}\n")
    TerminalDisplay.display_robots(robots)

    # 2. Initialize Warehouse & A* Motion Planner
    warehouse = WarehouseGrid()
    # Add optional central warehouse racks for obstacle navigation demonstration
    warehouse.add_default_warehouse_racks()
    planner = MotionPlanner(warehouse, allow_diagonal=True)

    # 3. Motion Planning Demonstration
    TerminalDisplay.print_section("1. PHASE 2: A* MOTION PLANNING & ZONE-CONSTRAINED ROUTING")
    print("Testing path planning across heterogeneous robot archetypes:\n")

    forklift = next(r for r in robots if r.robot_type == RobotType.FORKLIFT)
    drone = next(r for r in robots if r.robot_type == RobotType.DRONE)
    iw_hub = next(r for r in robots if r.robot_type == RobotType.IW_HUB)

    # Query 1: Forklift Staging -> Shipping (Must navigate around narrow aisle and central racks)
    p_fl = planner.plan_path(
        warehouse.get_zone_center("staging"),
        warehouse.get_zone_center("shipping"),
        robot=forklift,
    )

    # Query 2: Drone Staging -> Sorting (Direct 3D aerial line of sight)
    p_dr = planner.plan_path(
        warehouse.get_zone_center("staging"),
        warehouse.get_zone_center("sorting"),
        robot=drone,
    )

    # Query 3: IW Hub Staging -> Narrow Aisle (Enters restricted low-profile floor stacks)
    p_iw = planner.plan_path(
        warehouse.get_zone_center("staging"),
        warehouse.get_zone_center("narrow_aisle"),
        robot=iw_hub,
    )

    # Query 4: Forklift attempting to enter Narrow Aisle (Physical constraint violation check)
    p_fl_fail = planner.plan_path(
        warehouse.get_zone_center("staging"),
        warehouse.get_zone_center("narrow_aisle"),
        robot=forklift,
    )

    planned_paths = {
        "Forklift (Staging -> Shipping)": p_fl,
        "Drone (Staging -> Sorting)": p_dr,
        "IW Hub (Staging -> Narrow Aisle)": p_iw,
    }

    print(f"{'Path Query':<36} | {'Distance':<10} | {'Travel Time':<12} | {'States Expanded':<16} | {'Status'}")
    print("-" * 92)
    for name, p_res in planned_paths.items():
        status = "✅ REACHED" if p_res.success else "❌ FAILED"
        print(f"{name:<36} | {p_res.distance:>8.1f}m | {p_res.travel_time:>10.1f}s | {p_res.cells_explored:>15} | {status}")

    # Negative check display
    print(f"{'Forklift -> Narrow Aisle (Restricted)':<36} | {'N/A':>10} | {'N/A':>12} | {p_fl_fail.cells_explored:>15} | 🛑 {p_fl_fail.reason}")
    print("-" * 92 + "\n")

    # 4. Precompute Zone-to-Zone Travel Time Matrix
    TerminalDisplay.print_section("2. PRECOMPUTED FLEET TRAVEL TIME MATRIX (SECONDS)")
    time_matrix = planner.compute_travel_time_matrix(robots)
    zones = ["staging", "receiving", "sorting", "shipping", "narrow_aisle"]

    print(f"{'Origin -> Dest':<24} | {'Forklift (2.0 m/s)':<20} | {'Drone (5.0 m/s)':<18} | {'IW Hub (2.0 m/s)'}")
    print("-" * 82)
    for z1 in ["staging", "receiving"]:
        for z2 in ["sorting", "shipping", "narrow_aisle"]:
            t_fl = time_matrix[forklift.id].get((z1, z2), float("inf"))
            t_dr = time_matrix[drone.id].get((z1, z2), float("inf"))
            t_iw = time_matrix[iw_hub.id].get((z1, z2), float("inf"))

            s_fl = f"{t_fl:.1f}s" if not math.isinf(t_fl) else "BLOCKED"
            s_dr = f"{t_dr:.1f}s" if not math.isinf(t_dr) else "BLOCKED"
            s_iw = f"{t_iw:.1f}s" if not math.isinf(t_iw) else "BLOCKED"

            print(f"{z1:<10} -> {z2:<10} | {s_fl:<20} | {s_dr:<18} | {s_iw}")
    print("-" * 82 + "\n")

    # 5. Timeline Scheduling Demonstration
    TerminalDisplay.print_section("3. PHASE 3: MULTI-ROBOT RENDEZVOUS TIMELINE SCHEDULING")
    print("Scheduling warehouse tasks with interleaved travel times, synchronized coalition start, and makespan tracking:\n")

    schedule = Schedule(robots=robots, time_budget=1800.0)
    all_coalitions = TraitEngine.enumerate_all_coalitions(robots, max_size=2)

    # Sequentially assign tasks to best feasible coalition
    for task in tasks:
        # Filter zone accessible coalitions
        accessible_coalitions = TraitEngine.filter_zone_accessible_coalitions(all_coalitions, task.zone)

        # Score candidates
        candidates = []
        for c in accessible_coalitions:
            satisfaction, _ = TraitEngine.compute_trait_satisfaction(c, task)
            if satisfaction >= 0.8:  # Good match threshold
                feasible, _, start_t, end_t = schedule.can_assign(c, task, planner)
                if feasible:
                    candidates.append((satisfaction, -end_t, c))

        # Pick best satisfaction, then earliest finish
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

        if candidates:
            best_sat, _, best_c = candidates[0]
            res = schedule.add_assignment(best_c, task, planner)
            assign = res.assignment
            print(f"  📅 Scheduled Task {task.id:<3} ('{task.name:<26}')")
            print(f"     • Assigned Team: {best_c}")
            print(f"     • Zone: {task.zone}")
            print(f"     • Synchronized Window: {assign.execution_start:.1f}s -> {assign.execution_end:.1f}s (Dur: {task.duration_seconds}s)")
            print(f"     • Current Mission Makespan: {schedule.get_makespan():.1f}s\n")
        else:
            print(f"  ✗ Task {task.id:<3} ('{task.name:<26}') could not be scheduled\n")

    # Display Chronological Timeline Table
    sched_viz = ScheduleVisualizer()
    sched_viz.print_timeline_table(schedule)

    # 6. Fleet Resource Utilization Breakdown
    TerminalDisplay.print_section("4. FLEET WORKLOAD & UTILIZATION METRICS")
    util = schedule.get_robot_utilization()
    print(f"{'Robot ID':<10} | {'Robot Name':<16} | {'Execution %':<14} | {'Transit %':<12} | {'Idle %':<10} | {'Total Active (s)'}")
    print("-" * 82)
    for r in robots:
        u = util[r.id]
        total_active = u["total_exec_sec"] + u["total_travel_sec"]
        print(
            f"{r.id:<10} | {r.name:<16} | {u['exec_pct']:>11.1f}% | "
            f"{u['travel_pct']:>9.1f}% | {u['idle_pct']:>8.1f}% | {total_active:>14.1f}s"
        )
    print("-" * 82)
    print(f"  Mission Makespan: {schedule.get_makespan():.1f}s ({schedule.get_makespan()/60:.1f} minutes)")
    print(f"  Time Budget:      {schedule.time_budget:.1f}s (Overrun: {schedule.get_time_budget_overrun():.1f}s)\n")

    # 7. Export Visualizations
    TerminalDisplay.print_section("5. EXPORTING PHASE 2 & 3 VISUALIZATION ARTIFACTS")
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 7.1 Motion Paths Floor Map
    path_png = output_dir / "phase2_motion_paths.png"
    warehouse_viz = WarehouseVisualizer(warehouse)
    warehouse_viz.render_motion_paths(
        paths=planned_paths,
        robots=robots,
        tasks=tasks,
        save_path=path_png,
        title="E-ITAGS Phase 2: A* Motion Paths & Zone Routing",
    )

    # 7.2 Mission Gantt Chart
    gantt_png = output_dir / "phase3_mission_gantt.png"
    sched_viz.render_gantt(
        schedule=schedule,
        save_path=gantt_png,
        title=f"E-ITAGS Phase 3: Mission Schedule Gantt Chart (Makespan: {schedule.get_makespan():.1f}s)",
    )

    # 7.3 Fleet Utilization Chart
    util_png = output_dir / "phase3_fleet_utilization.png"
    sched_viz.render_robot_utilization(
        schedule=schedule,
        save_path=util_png,
    )

    TerminalDisplay.print_header(
        "PHASE 2 & 3 VERIFICATION COMPLETE",
        f"Artifacts successfully exported to:\n  {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
