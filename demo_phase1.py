#!/usr/bin/env python3
"""Interactive demonstration of Phase 0 & Phase 1: E-ITAGS Trait Models and Coalition Reasoning."""
from pathlib import Path
import yaml

from src.models.robot import Robot, RobotType, RobotTraits
from src.models.task import Task, TaskRequirements
from src.models.coalition import Coalition
from src.models.warehouse import WarehouseGrid
from src.core.trait_engine import TraitEngine
from src.utils.display import TerminalDisplay
from src.visualization.warehouse_viz import WarehouseVisualizer


def load_scenario(config_path: Path):
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    robots = []
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

    tasks = []
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
        "E-ITAGS: Phase 0 & 1 Interactive Walkthrough",
        "Autonomous Forklift, Drone & IW Hub Sorting System"
    )

    config_path = Path(__file__).parent / "configs" / "warehouse_scenario.yaml"
    scenario_name, robots, tasks = load_scenario(config_path)

    print(f"\n📂 Loaded Scenario: '{scenario_name}' from {config_path.name}")
    print(f"   Robots in fleet: {len(robots)} | Tasks to allocate: {len(tasks)}")

    # 1. Display Robot Fleet
    TerminalDisplay.display_robots(robots)

    # 2. Display Warehouse Tasks
    TerminalDisplay.display_tasks(tasks)

    # 3. Enumerate Coalitions
    coalitions = TraitEngine.enumerate_all_coalitions(robots, max_size=3)
    TerminalDisplay.display_coalitions(coalitions)

    # 4. Display Satisfaction Matrix
    TerminalDisplay.display_satisfaction_matrix(coalitions, tasks)

    # 5. Highlight Realizability
    TerminalDisplay.print_section("REALIZABILITY ANALYSIS (E-ITAGS Core Innovation)")
    print("In continuous optimization, solvers often assume a coalition with any arbitrary trait exists")
    print("(e.g. 1,500kg payload). But physically, available team traits are strictly DISCRETE.\n")

    print("  [Single Trait Query: Payload]")
    payload_queries = [0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0]
    for p in payload_queries:
        q_trait = RobotTraits(payload_kg=p)
        is_realizable, match = TraitEngine.is_realizable(q_trait, robots, trait_subset=["payload_kg"])
        symbol = "✅ REALIZABLE" if is_realizable else "❌ UNREALIZABLE"
        match_info = f"formed by {match}" if match else "No combination of robots yields this payload"
        print(f"    • Payload = {p:>6.0f} kg -> {symbol:<15} ({match_info})")

    print("\n  [Full Multi-Trait Vector Queries]")
    vector_queries = [
        ("Forklift + Drone Collaborative Vector", Coalition([robots[0], robots[1]]).aggregate_traits),
        ("Hypothetical Drone with Mast Lift", RobotTraits(payload_kg=0.0, has_camera_rgb=1.0, vertical_lift=1.0)),
    ]
    for label, q_trait in vector_queries:
        is_realizable, match = TraitEngine.is_realizable(q_trait, robots)
        symbol = "✅ REALIZABLE" if is_realizable else "❌ UNREALIZABLE"
        match_info = f"formed by {match}" if match else "Physically impossible with available robots"
        print(f"    • {label:<40} -> {symbol:<15} ({match_info})")

    # 6. Detailed Task-by-Task Diagnostic
    TerminalDisplay.print_section("WHY COALITIONS SUCCEED OR FAIL (DETAILED DIAGNOSTIC)")
    for task in tasks:
        print(f"\n[Task {task.id}: {task.name}] (Zone: {task.zone})")
        print(f"  Needs: {task.description}")
        for c in coalitions:
            score, breakdown = TraitEngine.compute_trait_satisfaction(c, task)
            if score == 1.0:
                print(f"    ✓ {str(c):<30} -> FULLY CAPABLE (Score: {score:.2f})")
            elif score > 0.0:
                print(f"    ~ {str(c):<30} -> PARTIAL (Score: {score:.2f}, breakdown: {breakdown})")
            else:
                # Find why it failed
                failed_reasons = []
                req = task.requirements
                agg = c.aggregate_traits
                if req.min_payload_kg > agg.payload_kg:
                    failed_reasons.append(f"Payload deficit ({agg.payload_kg:.0f} < {req.min_payload_kg:.0f}kg)")
                if req.min_vertical_lift > agg.vertical_lift:
                    failed_reasons.append("No vertical mast lift")
                if req.min_has_camera_rgb > agg.has_camera_rgb:
                    failed_reasons.append("Missing RGB camera")
                if req.min_has_lidar_2d > agg.has_lidar_2d:
                    failed_reasons.append("Missing 2D LiDAR")
                if req.min_narrow_space_access > agg.narrow_space_access:
                    failed_reasons.append("Cannot enter narrow space")
                print(f"    ✗ {str(c):<30} -> INCAPABLE: {', '.join(failed_reasons)}")

    # 7. Render 2D Warehouse Map & Comprehensive Research Dashboard
    TerminalDisplay.print_section("EXPORTING 2D WAREHOUSE VISUALIZATIONS")
    warehouse = WarehouseGrid()
    viz = WarehouseVisualizer(warehouse)
    img_path = Path(__file__).parent / "outputs" / "warehouse_map.png"
    dashboard_path = Path(__file__).parent / "outputs" / "warehouse_trait_dashboard.png"
    viz.render(robots=robots, tasks=tasks, save_path=img_path)
    viz.render_comprehensive_dashboard(robots=robots, tasks=tasks, save_path=dashboard_path)

    TerminalDisplay.print_header("PHASE 0 & 1 COMPLETE", f"All data models, trait mechanics & realizability verified!\n  Map exported to: {img_path}\n  Dashboard exported to: {dashboard_path}")


if __name__ == "__main__":
    main()
