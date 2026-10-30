"""Unit tests for Phase 8: Full System Benchmarking & Comparative Experiments."""
import os
import pytest
import numpy as np

from src.experiments.benchmark_suite import (
    EITAGSBenchmarkSuite,
    ComparativeITAGSResult,
    OptimalityGapResult,
    ActiveLearningBenchmarkResult,
    ClosedLoopBenchmarkResult,
    ScalabilityBenchmarkResult,
)
from src.visualization.benchmark_viz import BenchmarkVisualizer


@pytest.fixture
def suite():
    return EITAGSBenchmarkSuite(random_seed=42)


@pytest.fixture
def viz(tmp_path):
    return BenchmarkVisualizer(output_dir=str(tmp_path))


def test_benchmark_fleet_and_tasks(suite):
    """Test standard fleet and task generators."""
    base_fleet = suite.get_base_fleet()
    assert len(base_fleet) == 3

    multi_fleet = suite.get_multi_fleet()
    assert len(multi_fleet) == 6

    scaled_fleet = suite.get_scaled_fleet(8)
    assert len(scaled_fleet) == 8

    base_tasks = suite.get_base_tasks()
    assert len(base_tasks) == 5

    multi_tasks = suite.get_multi_tasks()
    assert len(multi_tasks) == 7


def test_experiment1_eitags_vs_itags(suite):
    """Test Experiment 1: E-ITAGS improves efficacy over standard ITAGS under time budgets."""
    results = suite.run_experiment1_eitags_vs_itags(problem_instances=2)
    assert len(results) == 2

    for r in results:
        assert isinstance(r, ComparativeITAGSResult)
        # E-ITAGS should achieve higher or equal efficacy compared to makespan-only ITAGS
        assert r.eitags_efficacy >= r.itags_efficacy - 1e-4
        assert r.itags_search_time > 0.0
        assert r.eitags_search_time > 0.0


def test_experiment2_suboptimality_bounds(suite):
    """Test Experiment 2: Empirical validation of theoretical suboptimality bounds (Paper Eq. 5)."""
    res = suite.run_experiment2_suboptimality_bounds(alpha_step=0.2)
    assert isinstance(res, OptimalityGapResult)
    assert len(res.alpha_values) >= 5
    assert res.bounds_respected is True

    # Check bounds hold for each alpha
    for actual, theor in zip(res.actual_gaps, res.theoretical_bounds):
        assert actual <= theor + 1e-5


def test_experiment3_active_learning_regrets(suite):
    """Test Experiment 3: Active learning cumulative regret, BUR, and runtime scalability."""
    res = suite.run_experiment3_active_learning_regrets(
        num_trials=2,
        num_iterations=4,
        team_sizes=[3, 4],
    )
    assert isinstance(res, ActiveLearningBenchmarkResult)
    assert set(res.strategies) == {"exact", "box", "unconstrained", "random"}

    for s in res.strategies:
        regrets = res.cumulative_regrets[s]
        assert regrets.shape == (2, 4)
        # Cumulative regret must be monotonically non-decreasing over iterations
        for trial in range(2):
            diffs = np.diff(regrets[trial])
            assert np.all(diffs >= -1e-5)

        burs = res.best_uncovered_rewards[s]
        assert burs.shape == (2, 4)
        # BUR must be monotonically non-decreasing
        for trial in range(2):
            diffs = np.diff(burs[trial])
            assert np.all(diffs >= -1e-5)

        assert len(res.runtime_per_step[s]) == 2


def test_experiment4_closed_loop_allocation(suite):
    """Test Experiment 4: Downstream graph search with actively learned models."""
    res = suite.run_experiment4_closed_loop_allocation(
        weights=[0.5],
        num_trials=1,
        query_budget=2,
    )
    assert isinstance(res, ClosedLoopBenchmarkResult)
    assert "exact" in res.strategies
    assert "oracle" in res.strategies

    # Check predicted and true efficacies are populated
    for strat in res.strategies:
        assert len(res.predicted_efficacies[strat][0.5]) == 1
        assert len(res.true_efficacies[strat][0.5]) == 1
        assert res.true_efficacies[strat][0.5][0] > 0.0


def test_experiment5_scalability(suite):
    """Test Experiment 5: Multi-robot warehouse scalability stress test."""
    res = suite.run_experiment5_scalability()
    assert isinstance(res, ScalabilityBenchmarkResult)
    assert len(res.scenario_names) == 3
    assert res.fleet_sizes == [3, 6, 8]
    assert res.task_counts == [5, 7, 8]

    for ms, nodes in zip(res.search_runtimes_ms, res.nodes_expanded):
        assert ms > 0.0
        assert nodes > 0


def test_benchmark_visualizations(suite, viz, tmp_path):
    """Test that all publication-grade figures render without error."""
    # 1. Run quick experiments to get data objects
    exp1_res = suite.run_experiment1_eitags_vs_itags(problem_instances=2)
    exp2_res = suite.run_experiment2_suboptimality_bounds(alpha_step=0.25)
    exp3_res = suite.run_experiment3_active_learning_regrets(num_trials=2, num_iterations=3, team_sizes=[3, 4])
    exp4_res = suite.run_experiment4_closed_loop_allocation(weights=[0.5], num_trials=1, query_budget=2)

    # 2. Render each figure
    p5 = viz.plot_efficacy_vs_makespan(exp1_res)
    assert os.path.exists(p5) and os.path.getsize(p5) > 1000

    p6 = viz.plot_runtime_breakdown(exp1_res)
    assert os.path.exists(p6) and os.path.getsize(p6) > 1000

    p7 = viz.plot_suboptimality_bounds(exp2_res)
    assert os.path.exists(p7) and os.path.getsize(p7) > 1000

    p9 = viz.plot_cumulative_regret(exp3_res)
    assert os.path.exists(p9) and os.path.getsize(p9) > 1000

    p10 = viz.plot_single_step_runtime(exp3_res)
    assert os.path.exists(p10) and os.path.getsize(p10) > 1000

    p11 = viz.plot_best_uncovered_reward(exp3_res)
    assert os.path.exists(p11) and os.path.getsize(p11) > 1000

    p12 = viz.plot_closed_loop_efficacy(exp4_res)
    assert os.path.exists(p12) and os.path.getsize(p12) > 1000

    p_comp = viz.plot_comprehensive_paper_figure(exp1_res, exp2_res, exp3_res, exp4_res)
    assert os.path.exists(p_comp) and os.path.getsize(p_comp) > 1000
