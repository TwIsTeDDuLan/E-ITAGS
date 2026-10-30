"""Unit tests for Phase 6 & 7: Trait-Efficacy Modeling and Realizability-Aware Active Learning."""
import sys
from pathlib import Path
import pytest
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.warehouse import WarehouseGrid
from src.models.robot import Robot, RobotType
from src.models.task import Task, TaskRequirements
from src.models.coalition import Coalition
from src.core.trait_engine import TraitEngine
from src.core.motion_planner import MotionPlanner
from src.core.graph_search import ITAGSSearch
from src.learning.efficacy_map import GroundTruthEfficacyOracle, HandCraftedEfficacyMap
from src.learning.active_learner import (
    SamplingStrategy,
    RealizabilityProjector,
    LearnedEfficacyGP,
    ActiveLearner,
)
from src.visualization.efficacy_viz import EfficacyVisualizer


@pytest.fixture
def learning_setup(tmp_path):
    fl = Robot.create_forklift("R_FL1", "Forklift-1", initial_zone="staging")
    dr = Robot.create_drone("R_DR1", "Drone-1", initial_zone="staging")
    iw = Robot.create_iw_hub("R_IW1", "IWHub-1", initial_zone="staging")
    robots = [fl, dr, iw]

    coalitions = TraitEngine.enumerate_all_coalitions(robots, max_size=2)

    t1 = Task(
        id="T1", name="Heavy Lift", zone="receiving", duration_seconds=100.0,
        requirements=TaskRequirements(min_payload_kg=800.0, min_vertical_lift=1.0)
    )
    t4_collab = Task(
        id="T4", name="Collaborative Sort", zone="sorting", duration_seconds=120.0,
        requirements=TaskRequirements(min_payload_kg=800.0, min_vertical_lift=1.0, min_has_camera_rgb=1.0)
    )
    tasks = [t1, t4_collab]

    oracle = GroundTruthEfficacyOracle(noise_std=0.01, seed=42)
    learner = ActiveLearner(tasks=tasks, oracle=oracle, random_seed=42)
    viz = EfficacyVisualizer()

    return robots, coalitions, tasks, t1, t4_collab, fl, dr, iw, oracle, learner, viz, tmp_path


def test_ground_truth_oracle_physical_thresholds(learning_setup):
    robots, coalitions, tasks, t1, t4_collab, fl, dr, iw, oracle, learner, viz, _ = learning_setup

    # Drone has zero payload and no mast lift -> cannot do heavy lift T1
    eff_dr = oracle.predict(Coalition([dr]), t1)
    assert eff_dr == 0.0

    # Forklift has 1000kg and mast lift -> high efficacy on T1
    eff_fl = oracle.predict(Coalition([fl]), t1)
    assert eff_fl > 0.85

    # IW Hub in receiving zone is physically blocked by ZONE_ACCESS
    eff_iw = oracle.predict(Coalition([iw]), t1)
    assert eff_iw == 0.0


def test_ground_truth_oracle_synergy(learning_setup):
    robots, coalitions, tasks, t1, t4_collab, fl, dr, iw, oracle, learner, viz, _ = learning_setup

    # T4 requires both mast lift and RGB camera
    # Solo forklift lacks camera -> lower efficacy
    eff_fl = oracle.predict(Coalition([fl]), t4_collab)
    # Team {Forklift + Drone} has both mast lift and high-res camera -> synergy bonus!
    eff_team = oracle.predict(Coalition([fl, dr]), t4_collab)

    assert eff_team > eff_fl
    assert eff_team > 0.90


def test_realizability_projector(learning_setup):
    robots, coalitions, tasks, t1, t4_collab, fl, dr, iw, oracle, learner, viz, _ = learning_setup
    projector = RealizabilityProjector()

    # Continuous target: 1100kg, RGB camera 0.8, mast lift 1.0
    continuous_target = np.array([1100.0, 1.0, 0.8, 1.0, 0.0, 2.0])

    accessible = TraitEngine.filter_zone_accessible_coalitions(coalitions, "sorting")
    best_c, best_vec, dist = projector.project(continuous_target, accessible)

    # Closest physical coalition should be {Forklift + Drone}
    assert best_c.contains(fl) and best_c.contains(dr)
    assert dist >= 0.0


def test_gp_uncertainty_reduction(learning_setup):
    robots, coalitions, tasks, t1, t4_collab, fl, dr, iw, oracle, learner, viz, _ = learning_setup
    gp = LearnedEfficacyGP(task_id="T4", random_state=42)

    vec = Coalition([fl, dr]).to_vector()

    # Prior uncertainty is 0.5
    _, prior_std = gp.predict_with_uncertainty(vec)
    assert prior_std == 0.5

    # Add observations
    true_val = oracle.predict_vector(vec, t4_collab)
    gp.add_observation(vec, true_val)
    gp.add_observation(vec, true_val)

    # Posterior uncertainty must decrease significantly
    pred_mean, post_std = gp.predict_with_uncertainty(vec)
    assert post_std < prior_std
    assert np.isclose(pred_mean, true_val, atol=0.1)


def test_active_learning_exact_strategy_reduces_rmse(learning_setup):
    robots, coalitions, tasks, t1, t4_collab, fl, dr, iw, oracle, learner, viz, _ = learning_setup
    accessible = TraitEngine.filter_zone_accessible_coalitions(coalitions, t4_collab.zone)

    initial_rmse = learner.compute_rmse(t4_collab, accessible)

    # Perform 5 active queries using EXACT realizable strategy
    for _ in range(5):
        learner.query(t4_collab, accessible, strategy=SamplingStrategy.EXACT)

    final_rmse = learner.compute_rmse(t4_collab, accessible)
    assert final_rmse < initial_rmse


def test_active_learning_all_strategies_execute(learning_setup):
    robots, coalitions, tasks, t1, t4_collab, fl, dr, iw, oracle, learner, viz, _ = learning_setup
    accessible = TraitEngine.filter_zone_accessible_coalitions(coalitions, t1.zone)

    # Test each strategy executes cleanly
    for strat in [SamplingStrategy.EXACT, SamplingStrategy.BOX, SamplingStrategy.UNCONSTRAINED, SamplingStrategy.RANDOM]:
        rec = learner.query(t1, accessible, strategy=strat)
        assert rec.observed_efficacy >= 0.0
        assert rec.coalition in accessible


def test_itags_search_with_learned_gp_efficacy(learning_setup):
    robots, coalitions, tasks, t1, t4_collab, fl, dr, iw, oracle, learner, viz, _ = learning_setup
    warehouse = WarehouseGrid()
    planner = MotionPlanner(warehouse)

    # Train GP models with a few observations
    for t in tasks:
        acc = TraitEngine.filter_zone_accessible_coalitions(coalitions, t.zone)
        for _ in range(3):
            learner.query(t, acc, strategy=SamplingStrategy.EXACT)

    # Define learned efficacy function
    def learned_efficacy_fn(c: Coalition, task: Task) -> float:
        model = learner.models.get(task.id)
        if model:
            return model.predict(c, task)
        return 0.5

    # Run ITAGSSearch using learned GP efficacy function!
    searcher = ITAGSSearch(
        robots=robots,
        tasks=tasks,
        warehouse=warehouse,
        motion_planner=planner,
        time_budget=1800.0,
        weight=0.5,
    )
    res = searcher.solve(weight=0.5, efficacy_fn=learned_efficacy_fn)

    assert res.solution_found is True
    assert len(res.assignments) == len(tasks)
    assert res.makespan > 0.0


def test_efficacy_visualizations(learning_setup):
    robots, coalitions, tasks, t1, t4_collab, fl, dr, iw, oracle, learner, viz, tmp_path = learning_setup
    accessible = TraitEngine.filter_zone_accessible_coalitions(coalitions, t4_collab.zone)

    # 1. Learning curves
    rmse_dict = {
        "exact": [0.45, 0.30, 0.18, 0.09],
        "random": [0.45, 0.40, 0.32, 0.28],
    }
    curve_path = tmp_path / "test_learning_curves.png"
    fig1 = viz.render_learning_curves(rmse_dict, save_path=curve_path)
    assert isinstance(fig1, plt.Figure)
    assert curve_path.exists()
    plt.close(fig1)

    # 2. GP Posterior
    gp = learner.models[t4_collab.id]
    learner.query(t4_collab, accessible, strategy=SamplingStrategy.EXACT)
    post_path = tmp_path / "test_gp_posterior.png"
    fig2 = viz.render_gp_posterior(t4_collab, accessible, gp, oracle, save_path=post_path)
    assert isinstance(fig2, plt.Figure)
    assert post_path.exists()
    plt.close(fig2)

    # 3. Realizability Surface
    surf_path = tmp_path / "test_surface.png"
    fig3 = viz.render_realizability_manifold_surface(t4_collab, oracle, accessible, save_path=surf_path)
    assert isinstance(fig3, plt.Figure)
    assert surf_path.exists()
    plt.close(fig3)
