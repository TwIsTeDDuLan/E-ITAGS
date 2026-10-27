
"""Unit tests for Phase 3: Timeline Scheduler & ScheduleVisualizer."""
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
from src.core.scheduler import Schedule, ScheduleResult
from src.visualization.schedule_viz import ScheduleVisualizer


@pytest.fixture
def scheduler_setup(tmp_path):
    warehouse = WarehouseGrid()
    planner = MotionPlanner(warehouse)

    fl = Robot.create_forklift("R_FL1", "Forklift-1", initial_zone="staging")
    dr = Robot.create_drone("R_DR1", "Drone-1", initial_zone="staging")
    iw = Robot.create_iw_hub("R_IW1", "IWHub-1", initial_zone="staging")
    robots = [fl, dr, iw]

    t1 = Task(
        id="T1", name="Receiving Pallet", zone="receiving", duration_seconds=100.0,
        requirements=TaskRequirements(min_payload_kg=800.0, min_vertical_lift=1.0)
    )
    t2 = Task(
        id="T2", name="Sorting Pallet", zone="sorting", duration_seconds=80.0,
        requirements=TaskRequirements(min_payload_kg=500.0)
    )
    t3_tight_deadline = Task(
        id="T3", name="Urgent Survey", zone="sorting", duration_seconds=60.0,
        deadline_seconds=50.0, requirements=TaskRequirements(min_has_camera_rgb=1.0)
    )

    schedule = Schedule(robots=robots, time_budget=3600.0)
    viz = ScheduleVisualizer()

    return warehouse, planner, robots, fl, dr, iw, [t1, t2, t3_tight_deadline], schedule, viz, tmp_path


def test_schedule_initial_state(scheduler_setup):
    *_, schedule, _, _ = scheduler_setup
    assert schedule.get_makespan() == 0.0
    assert schedule.check_time_budget() is True
    for r_id in schedule.robots:
        assert schedule.get_robot_availability(r_id) == 0.0
        assert schedule.get_robot_current_zone(r_id) == "staging"


def test_single_task_assignment(scheduler_setup):
    warehouse, planner, robots, fl, dr, iw, tasks, schedule, viz, _ = scheduler_setup
    t1 = tasks[0]  # receiving, 100s
    c = Coalition([fl])

    travel_time = planner.compute_travel_time("staging", "receiving", fl)
    res = schedule.add_assignment(c, t1, planner)

    assert res.success is True
    expected_makespan = travel_time + t1.duration_seconds
    assert math.isclose(schedule.get_makespan(), expected_makespan, rel_tol=1e-3)
    assert schedule.get_robot_current_zone(fl.id) == "receiving"
    assert len(schedule.robot_timelines[fl.id]) == 1


def test_coalition_synchronization(scheduler_setup):
    warehouse, planner, robots, fl, dr, iw, tasks, schedule, viz, _ = scheduler_setup
    t_sort = tasks[1]  # sorting, 80s
    # Forklift (2 m/s) and Drone (5 m/s)
    c = Coalition([fl, dr])

    t_fl = planner.compute_travel_time("staging", "sorting", fl)
    t_dr = planner.compute_travel_time("staging", "sorting", dr)
    assert t_dr < t_fl  # drone arrives faster

    res = schedule.add_assignment(c, t_sort, planner)
    assert res.success is True

    # Execution cannot begin until forklift arrives (t_fl)
    assign = res.assignment
    assert math.isclose(assign.execution_start, t_fl, rel_tol=1e-3)
    assert math.isclose(assign.execution_end, t_fl + t_sort.duration_seconds, rel_tol=1e-3)

    # Both robots finish at the same time
    assert math.isclose(schedule.get_robot_availability(fl.id), assign.execution_end, rel_tol=1e-3)
    assert math.isclose(schedule.get_robot_availability(dr.id), assign.execution_end, rel_tol=1e-3)


def test_no_double_booking_sequential(scheduler_setup):
    warehouse, planner, robots, fl, dr, iw, tasks, schedule, viz, _ = scheduler_setup
    t1, t2 = tasks[0], tasks[1]
    c1 = Coalition([fl])
    c2 = Coalition([fl])  # same robot

    res1 = schedule.add_assignment(c1, t1, planner)
    assert res1.success is True
    finish1 = schedule.get_robot_availability(fl.id)

    # Forklift is now at receiving, must move to sorting for T2
    res2 = schedule.add_assignment(c2, t2, planner)
    assert res2.success is True
    assign2 = res2.assignment

    # T2 execution must start AFTER finish1
    assert assign2.execution_start >= finish1


def test_deadline_enforcement(scheduler_setup):
    warehouse, planner, robots, fl, dr, iw, tasks, schedule, viz, _ = scheduler_setup
    t_tight = tasks[2]  # deadline = 50.0s, duration = 60s
    c = Coalition([dr])

    res = schedule.add_assignment(c, t_tight, planner)
    assert res.success is False
    assert "deadline" in res.reason.lower()


def test_duplicate_task_rejection(scheduler_setup):
    warehouse, planner, robots, fl, dr, iw, tasks, schedule, viz, _ = scheduler_setup
    t1 = tasks[0]
    c = Coalition([fl])

    res1 = schedule.add_assignment(c, t1, planner)
    assert res1.success is True

    res2 = schedule.add_assignment(c, t1, planner)
    assert res2.success is False
    assert "already scheduled" in res2.reason.lower()


def test_schedule_copy_isolation(scheduler_setup):
    warehouse, planner, robots, fl, dr, iw, tasks, schedule, viz, _ = scheduler_setup
    t1, t2 = tasks[0], tasks[1]

    schedule.add_assignment(Coalition([fl]), t1, planner)
    orig_makespan = schedule.get_makespan()
    assert schedule.get_robot_availability(dr.id) == 0.0

    # Clone schedule
    branched_sched = schedule.copy()
    assert branched_sched.get_makespan() == orig_makespan

    # Assign t2 to drone on branched schedule
    res = branched_sched.add_assignment(Coalition([dr]), t2, planner)
    assert res.success is True

    # Drone on branched schedule has work
    assert branched_sched.get_robot_availability(dr.id) > 0.0
    # Original schedule MUST be completely isolated: Drone has 0.0 availability
    assert schedule.get_robot_availability(dr.id) == 0.0
    assert t2.id not in schedule.assigned_task_ids
    assert t2.id in branched_sched.assigned_task_ids

    # Assigning another task to Forklift on branched schedule extends makespan
    t_more = Task(id="T_MORE", name="Extra Move", zone="receiving", duration_seconds=50.0, requirements=TaskRequirements())
    res_more = branched_sched.add_assignment(Coalition([fl]), t_more, planner)
    assert res_more.success is True
    assert branched_sched.get_makespan() > orig_makespan
    # Original schedule makespan is still untouched
    assert schedule.get_makespan() == orig_makespan


def test_robot_utilization_breakdown(scheduler_setup):
    warehouse, planner, robots, fl, dr, iw, tasks, schedule, viz, _ = scheduler_setup
    t1 = tasks[0]
    schedule.add_assignment(Coalition([fl]), t1, planner)

    util = schedule.get_robot_utilization()
    assert fl.id in util
    fl_stats = util[fl.id]
    total_pct = fl_stats["travel_pct"] + fl_stats["exec_pct"] + fl_stats["idle_pct"]
    assert math.isclose(total_pct, 100.0, abs_tol=0.2)
    assert fl_stats["exec_pct"] > 0.0


def test_schedule_visualizations(scheduler_setup):
    warehouse, planner, robots, fl, dr, iw, tasks, schedule, viz, tmp_path = scheduler_setup
    t1, t2 = tasks[0], tasks[1]
    schedule.add_assignment(Coalition([fl]), t1, planner)
    schedule.add_assignment(Coalition([dr]), t2, planner)

    # 1. Gantt chart render
    gantt_path = tmp_path / "test_gantt.png"
    fig1 = viz.render_gantt(schedule, save_path=gantt_path)
    assert isinstance(fig1, plt.Figure)
    assert gantt_path.exists()
    plt.close(fig1)

    # 2. Utilization chart render
    util_path = tmp_path / "test_utilization.png"
    fig2 = viz.render_robot_utilization(schedule, save_path=util_path)
    assert isinstance(fig2, plt.Figure)
    assert util_path.exists()
    plt.close(fig2)
