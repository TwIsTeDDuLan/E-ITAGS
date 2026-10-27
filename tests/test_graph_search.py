"""Unit tests for Phase 5: ITAGSSearch & SearchTreeVisualizer."""
import sys
from pathlib import Path
import pytest
import math
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.warehouse import WarehouseGrid
from src.models.robot import Robot
from src.models.task import Task, TaskRequirements
from src.models.coalition import Coalition
from src.core.motion_planner import MotionPlanner
from src.core.graph_search import ITAGSSearch, SearchResult
from src.visualization.search_tree_viz import SearchTreeVisualizer


@pytest.fixture
def search_setup(tmp_path):
    warehouse = WarehouseGrid()
    planner = MotionPlanner(warehouse)

    fl = Robot.create_forklift("R_FL1", "Forklift-1", initial_zone="staging")
    dr = Robot.create_drone("R_DR1", "Drone-1", initial_zone="staging")
    iw = Robot.create_iw_hub("R_IW1", "IWHub-1", initial_zone="staging")
    robots = [fl, dr, iw]

    t1 = Task(
        id="T1", name="Heavy Lift", zone="receiving", duration_seconds=100.0,
        requirements=TaskRequirements(min_payload_kg=800.0, min_vertical_lift=1.0, min_has_lidar_2d=1.0)
    )
    t2 = Task(
        id="T2", name="Survey", zone="sorting", duration_seconds=60.0,
        requirements=TaskRequirements(min_has_camera_rgb=1.0)
    )
    t3 = Task(
        id="T3", name="Narrow Move", zone="narrow_aisle", duration_seconds=90.0,
        requirements=TaskRequirements(min_payload_kg=500.0, min_narrow_space_access=1.0)
    )
    tasks = [t1, t2, t3]

    searcher = ITAGSSearch(
        robots=robots,
        tasks=tasks,
        warehouse=warehouse,
        motion_planner=planner,
        time_budget=1800.0,
        weight=0.5,
        max_coalition_size=2,
    )

    viz = SearchTreeVisualizer()

    return warehouse, planner, robots, tasks, searcher, viz, tmp_path


def test_search_solves_scenario(search_setup):
    warehouse, planner, robots, tasks, searcher, viz, _ = search_setup

    res = searcher.solve(weight=0.5)
    assert res.solution_found is True
    assert len(res.assignments) == 3
    assert set(res.assignments.keys()) == {"T1", "T2", "T3"}

    # Verify task allocations match physical capabilities
    # T1 requires high lift -> Forklift
    assert any(r.robot_type.value == "forklift" for r in res.assignments["T1"].members)
    # T2 requires RGB camera -> Drone
    assert any(r.robot_type.value == "drone" for r in res.assignments["T2"].members)
    # T3 requires narrow access -> IW Hub
    assert any(r.robot_type.value == "iw_hub" for r in res.assignments["T3"].members)

    assert res.makespan > 0.0
    assert res.mean_efficacy > 0.8
    assert len(res.solution_path) == 4  # Root + 3 task additions


def test_most_constrained_task_selection(search_setup):
    warehouse, planner, robots, tasks, searcher, viz, _ = search_setup

    # T3 (narrow aisle) is strictly accessible only to IW Hub (1 coalition)
    # T2 (sorting) is accessible to Forklift, Drone, IW Hub (many coalitions)
    unassigned = ["T2", "T3"]
    chosen = searcher.select_next_task(unassigned)
    assert chosen.id == "T3"


def test_infeasible_task_no_solution(search_setup):
    warehouse, planner, robots, tasks, searcher, viz, _ = search_setup

    # Create impossible task requiring 5000kg payload (fleet max is 2000kg)
    impossible_task = Task(
        id="T_IMP", name="Impossible Load", zone="sorting", duration_seconds=100.0,
        requirements=TaskRequirements(min_payload_kg=5000.0)
    )
    imp_searcher = ITAGSSearch(
        robots=robots,
        tasks=[impossible_task],
        warehouse=warehouse,
        motion_planner=planner,
    )

    res = imp_searcher.solve()
    assert res.solution_found is False
    assert res.goal_node is None


def test_tight_deadline_pruning(search_setup):
    warehouse, planner, robots, tasks, searcher, viz, _ = search_setup

    # Task requiring forklift (takes ~10s to travel + 50s duration) with deadline 20s
    tight_task = Task(
        id="T_TIGHT", name="Urgent Lift", zone="receiving", duration_seconds=50.0,
        deadline_seconds=20.0,
        requirements=TaskRequirements(min_payload_kg=800.0, min_vertical_lift=1.0)
    )
    tight_searcher = ITAGSSearch(
        robots=robots,
        tasks=[tight_task],
        warehouse=warehouse,
        motion_planner=planner,
    )

    res = tight_searcher.solve()
    assert res.solution_found is False
    assert tight_searcher.task_candidates["T_TIGHT"] != []  # was candidate, but pruned by scheduler!


def test_pareto_front_sweep(search_setup):
    warehouse, planner, robots, tasks, searcher, viz, _ = search_setup

    results = searcher.sweep_pareto_front(weights=[0.0, 0.5, 1.0])
    assert len(results) == 3
    for r in results:
        assert r.solution_found is True
        assert r.makespan > 0.0


def test_search_tree_and_pareto_visualizations(search_setup):
    warehouse, planner, robots, tasks, searcher, viz, tmp_path = search_setup

    res = searcher.solve(weight=0.5)

    # 1. Search Tree Visualizer
    tree_path = tmp_path / "test_search_tree.png"
    fig_tree = viz.render_search_tree(res, save_path=tree_path)
    assert isinstance(fig_tree, plt.Figure)
    assert tree_path.exists()
    plt.close(fig_tree)

    # 2. Pareto Front Visualizer
    pareto_results = searcher.sweep_pareto_front(weights=[0.0, 0.5, 1.0])
    pareto_path = tmp_path / "test_pareto.png"
    fig_pareto = viz.render_pareto_front(pareto_results, save_path=pareto_path)
    assert isinstance(fig_pareto, plt.Figure)
    assert pareto_path.exists()
    plt.close(fig_pareto)
