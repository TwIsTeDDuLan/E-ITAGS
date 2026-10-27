"""Unit tests for Phase 4: Dual-Objective HeuristicEngine (NAC + TBO)."""
import sys
from pathlib import Path
import pytest
import math

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.warehouse import WarehouseGrid
from src.models.robot import Robot
from src.models.task import Task, TaskRequirements
from src.models.coalition import Coalition
from src.core.trait_engine import TraitEngine
from src.core.heuristics import HeuristicEngine


@pytest.fixture
def heuristic_setup():
    fl = Robot.create_forklift("R_FL1", "Forklift-1", initial_zone="staging")
    dr = Robot.create_drone("R_DR1", "Drone-1", initial_zone="staging")
    iw = Robot.create_iw_hub("R_IW1", "IWHub-1", initial_zone="staging")
    robots = [fl, dr, iw]

    coalitions = TraitEngine.enumerate_all_coalitions(robots, max_size=2)

    t1 = Task(
        id="T1", name="Heavy Lift", zone="receiving", duration_seconds=100.0,
        requirements=TaskRequirements(min_payload_kg=800.0, min_vertical_lift=1.0, min_has_lidar_2d=1.0)
    )
    t2 = Task(
        id="T2", name="Survey", zone="sorting", duration_seconds=60.0,
        requirements=TaskRequirements(min_has_camera_rgb=1.0)
    )
    t3_narrow = Task(
        id="T3", name="Narrow Move", zone="narrow_aisle", duration_seconds=90.0,
        requirements=TaskRequirements(min_payload_kg=500.0, min_narrow_space_access=1.0)
    )
    tasks = [t1, t2, t3_narrow]

    upper_bounds = HeuristicEngine.compute_optimistic_task_upper_bounds(tasks, coalitions)

    return robots, coalitions, tasks, upper_bounds, fl, dr, iw


def test_upper_bounds_computation(heuristic_setup):
    robots, coalitions, tasks, upper_bounds, fl, dr, iw = heuristic_setup

    assert "T1" in upper_bounds
    assert "T2" in upper_bounds
    assert "T3" in upper_bounds

    # Forklift satisfies T1 (receiving) with score 1.0
    assert upper_bounds["T1"] == 1.0
    # Drone satisfies T2 (sorting) with score 1.0
    assert upper_bounds["T2"] == 1.0
    # IW Hub satisfies T3 (narrow aisle) with score 1.0
    assert upper_bounds["T3"] == 1.0


def test_nac_perfect_allocation(heuristic_setup):
    robots, coalitions, tasks, upper_bounds, fl, dr, iw = heuristic_setup

    assignments = {
        "T1": Coalition([fl]),
        "T2": Coalition([dr]),
        "T3": Coalition([iw]),
    }

    nac = HeuristicEngine.compute_nac(assignments, tasks, upper_bounds)
    assert math.isclose(nac, 0.0, abs_tol=1e-5)


def test_nac_partial_allocation_and_monotonicity(heuristic_setup):
    robots, coalitions, tasks, upper_bounds, fl, dr, iw = heuristic_setup

    # At root (empty assignments), NAC is based purely on upper bounds
    root_nac = HeuristicEngine.compute_nac({}, tasks, upper_bounds)
    assert math.isclose(root_nac, 0.0, abs_tol=1e-5)  # all upper bounds are 1.0

    # If we assign a sub-optimal coalition (e.g. Drone to T1 receiving gives score 0.0)
    bad_assignments = {"T1": Coalition([dr])}
    bad_nac = HeuristicEngine.compute_nac(bad_assignments, tasks, upper_bounds)
    # T1=0.0, T2=1.0 (est), T3=1.0 (est) -> mean efficacy = 2.0 / 3 = 0.667 -> NAC = 0.333
    assert bad_nac > root_nac
    assert math.isclose(bad_nac, 1.0 / 3.0, rel_tol=1e-3)


def test_tbo_penalty_behavior():
    time_budget = 1000.0

    # Case 1: Within budget
    tbo_safe = HeuristicEngine.compute_tbo(makespan=800.0, time_budget=time_budget)
    assert tbo_safe == 0.0

    # Case 2: Exactly on budget
    tbo_exact = HeuristicEngine.compute_tbo(makespan=1000.0, time_budget=time_budget)
    assert tbo_exact == 0.0

    # Case 3: 20% overtime (1200s vs 1000s)
    tbo_over = HeuristicEngine.compute_tbo(makespan=1200.0, time_budget=time_budget)
    assert math.isclose(tbo_over, 0.2, rel_tol=1e-4)


def test_combined_heuristic_weighting():
    nac = 0.2
    tbo = 0.4

    # Pure efficacy (w = 1.0)
    h_eff = HeuristicEngine.compute_combined_heuristic(nac, tbo, weight=1.0)
    assert math.isclose(h_eff, nac, rel_tol=1e-5)

    # Pure schedule (w = 0.0)
    h_sched = HeuristicEngine.compute_combined_heuristic(nac, tbo, weight=0.0)
    assert math.isclose(h_sched, tbo, rel_tol=1e-5)

    # Balanced (w = 0.5)
    h_mid = HeuristicEngine.compute_combined_heuristic(nac, tbo, weight=0.5)
    assert math.isclose(h_mid, 0.5 * nac + 0.5 * tbo, rel_tol=1e-5)


def test_heuristic_evaluate_method(heuristic_setup):
    robots, coalitions, tasks, upper_bounds, fl, dr, iw = heuristic_setup
    assignments = {"T1": Coalition([fl])}

    h_cost, nac, tbo = HeuristicEngine.evaluate(
        assignments=assignments,
        makespan=1200.0,
        all_tasks=tasks,
        task_upper_bounds=upper_bounds,
        time_budget=1000.0,
        weight=0.5,
    )

    assert nac >= 0.0
    assert math.isclose(tbo, 0.2, rel_tol=1e-4)
    assert math.isclose(h_cost, 0.5 * nac + 0.5 * tbo, rel_tol=1e-4)
