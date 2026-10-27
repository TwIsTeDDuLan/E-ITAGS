"""Unit tests for WarehouseVisualizer and its integration with TraitEngine."""
import sys
from pathlib import Path
import pytest
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.robot import Robot
from src.models.task import Task, TaskRequirements
from src.models.coalition import Coalition
from src.models.warehouse import WarehouseGrid
from src.core.trait_engine import TraitEngine
from src.visualization.warehouse_viz import WarehouseVisualizer


@pytest.fixture
def scenario_setup(tmp_path):
    warehouse = WarehouseGrid()
    viz = WarehouseVisualizer(warehouse)

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
    tasks = [t1, t2]

    return viz, robots, tasks, tmp_path


def test_warehouse_viz_render(scenario_setup):
    viz, robots, tasks, tmp_path = scenario_setup
    save_path = tmp_path / "test_map.png"
    fig = viz.render(robots=robots, tasks=tasks, save_path=save_path)
    assert isinstance(fig, plt.Figure)
    assert save_path.exists()
    plt.close(fig)


def test_warehouse_viz_satisfaction_matrix(scenario_setup):
    viz, robots, tasks, tmp_path = scenario_setup
    coalitions = TraitEngine.enumerate_all_coalitions(robots, max_size=2)
    save_path = tmp_path / "test_matrix.png"
    fig = viz.render_satisfaction_matrix(coalitions=coalitions, tasks=tasks, save_path=save_path)
    assert isinstance(fig, plt.Figure)
    assert save_path.exists()
    plt.close(fig)


def test_warehouse_viz_realizability(scenario_setup):
    viz, robots, tasks, tmp_path = scenario_setup
    save_path = tmp_path / "test_realizability.png"
    fig = viz.render_realizability_chart(robots=robots, query_payloads=[0.0, 500.0, 1000.0], save_path=save_path)
    assert isinstance(fig, plt.Figure)
    assert save_path.exists()
    plt.close(fig)


def test_warehouse_viz_allocation_map(scenario_setup):
    viz, robots, tasks, tmp_path = scenario_setup
    allocations = {"T1": Coalition([robots[0]]), "T2": Coalition([robots[1]])}
    save_path = tmp_path / "test_allocations.png"
    fig = viz.render_allocation_map(robots=robots, tasks=tasks, allocations=allocations, save_path=save_path)
    assert isinstance(fig, plt.Figure)
    assert save_path.exists()
    plt.close(fig)


def test_warehouse_viz_comprehensive_dashboard(scenario_setup):
    viz, robots, tasks, tmp_path = scenario_setup
    save_path = tmp_path / "test_dashboard.png"
    fig = viz.render_comprehensive_dashboard(robots=robots, tasks=tasks, save_path=save_path)
    assert isinstance(fig, plt.Figure)
    assert save_path.exists()
    plt.close(fig)
