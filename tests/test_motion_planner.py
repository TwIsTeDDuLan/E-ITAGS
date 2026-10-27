"""Unit tests for Phase 2: A* MotionPlanner."""
import sys
from pathlib import Path
import pytest
import math

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.warehouse import WarehouseGrid, Location
from src.models.robot import Robot, RobotType
from src.models.coalition import Coalition
from src.core.motion_planner import MotionPlanner, PathResult


@pytest.fixture
def planner_setup():
    warehouse = WarehouseGrid()
    planner = MotionPlanner(warehouse)

    fl = Robot.create_forklift("R_FL1", "Forklift-1", initial_zone="staging")
    dr = Robot.create_drone("R_DR1", "Drone-1", initial_zone="staging")
    iw = Robot.create_iw_hub("R_IW1", "IWHub-1", initial_zone="staging")
    robots = [fl, dr, iw]

    return warehouse, planner, fl, dr, iw, robots


def test_motion_planner_basic_path(planner_setup):
    warehouse, planner, fl, dr, iw, robots = planner_setup
    start = warehouse.get_zone_center("staging")      # (5, 7)
    goal = warehouse.get_zone_center("receiving")     # (4, 15)

    res = planner.plan_path(start, goal, robot=fl)
    assert res.success is True
    assert len(res.path) >= 2
    assert res.path[0] == start
    assert res.path[-1] == goal
    assert res.distance > 0.0
    assert math.isclose(res.travel_time, res.distance / fl.traits.speed_mps, rel_tol=1e-3)
    assert res.cells_explored > 0


def test_forklift_routes_around_narrow_aisle(planner_setup):
    warehouse, planner, fl, dr, iw, robots = planner_setup
    # Forklift moves from staging (5, 7) to shipping (22, 5)
    start = warehouse.get_zone_center("staging")
    goal = warehouse.get_zone_center("shipping")

    res = planner.plan_path(start, goal, robot=fl)
    assert res.success is True

    # None of the path waypoints may be in narrow_aisle
    narrow_cells = set(warehouse.zones["narrow_aisle"].cells)
    for loc in res.path:
        assert loc not in narrow_cells, f"Forklift illegally entered narrow aisle cell: {loc}"


def test_forklift_cannot_enter_narrow_aisle(planner_setup):
    warehouse, planner, fl, dr, iw, robots = planner_setup
    start = warehouse.get_zone_center("staging")
    goal = warehouse.get_zone_center("narrow_aisle")

    res = planner.plan_path(start, goal, robot=fl)
    assert res.success is False
    assert res.distance == float("inf")
    assert "inaccessible" in res.reason.lower()


def test_iw_hub_can_enter_narrow_aisle(planner_setup):
    warehouse, planner, fl, dr, iw, robots = planner_setup
    start = warehouse.get_zone_center("staging")
    goal = warehouse.get_zone_center("narrow_aisle")

    res = planner.plan_path(start, goal, robot=iw)
    assert res.success is True
    assert res.path[-1] == goal
    assert res.distance > 0.0


def test_iw_hub_cannot_enter_receiving_or_shipping(planner_setup):
    warehouse, planner, fl, dr, iw, robots = planner_setup
    start = warehouse.get_zone_center("staging")
    rec_goal = warehouse.get_zone_center("receiving")
    ship_goal = warehouse.get_zone_center("shipping")

    res_rec = planner.plan_path(start, rec_goal, robot=iw)
    assert res_rec.success is False
    assert "inaccessible" in res_rec.reason.lower()

    res_ship = planner.plan_path(start, ship_goal, robot=iw)
    assert res_ship.success is False
    assert "inaccessible" in res_ship.reason.lower()


def test_drone_direct_flight(planner_setup):
    warehouse, planner, fl, dr, iw, robots = planner_setup
    start = warehouse.get_zone_center("staging")    # (5, 7)
    goal = warehouse.get_zone_center("sorting")     # (22, 15)

    res = planner.plan_path(start, goal, robot=dr)
    assert res.success is True
    euclid_dist = start.euclidean_distance_to(goal)
    assert math.isclose(res.distance, euclid_dist, rel_tol=1e-3)
    assert math.isclose(res.travel_time, euclid_dist / dr.traits.speed_mps, rel_tol=1e-3)
    assert res.cells_explored == 1  # Direct calculation


def test_coalition_travel_time(planner_setup):
    warehouse, planner, fl, dr, iw, robots = planner_setup
    # Coalition of Forklift (2.0 m/s) and Drone (5.0 m/s)
    coalition = Coalition([fl, dr])

    t_fl = planner.compute_travel_time("staging", "sorting", fl)
    t_dr = planner.compute_travel_time("staging", "sorting", dr)

    # Drone is much faster, so coalition travel time is bounded by forklift
    assert t_dr < t_fl

    t_coalition = planner.compute_coalition_travel_time(coalition, to_zone="sorting", from_zone="staging")
    assert math.isclose(t_coalition, max(t_fl, t_dr), rel_tol=1e-3)


def test_obstacle_avoidance(planner_setup):
    warehouse, planner, fl, dr, iw, robots = planner_setup
    # Put a wall of obstacles across x=10, y=0..10
    wall = [(10, y) for y in range(0, 11)]
    warehouse.add_obstacles(wall)
    planner.clear_cache()

    start = Location(5, 5)
    goal = Location(15, 5)

    res = planner.plan_path(start, goal, robot_type=RobotType.FORKLIFT, speed_mps=2.0)
    assert res.success is True
    # Verify no cell in res.path is in wall
    for loc in res.path:
        assert (loc.x, loc.y) not in warehouse.obstacles


def test_travel_time_matrix(planner_setup):
    warehouse, planner, fl, dr, iw, robots = planner_setup
    matrix = planner.compute_travel_time_matrix(robots)

    assert fl.id in matrix
    assert dr.id in matrix
    assert iw.id in matrix

    # Diagonal (same zone) must be 0.0
    for r in robots:
        for z in warehouse.zones.keys():
            assert matrix[r.id][(z, z)] == 0.0

    # Drone time between staging and sorting must be faster than forklift time
    assert matrix[dr.id][("staging", "sorting")] < matrix[fl.id][("staging", "sorting")]


def test_path_caching(planner_setup):
    warehouse, planner, fl, dr, iw, robots = planner_setup
    start = warehouse.get_zone_center("staging")
    goal = warehouse.get_zone_center("receiving")

    res1 = planner.plan_path(start, goal, robot=fl)
    res2 = planner.plan_path(start, goal, robot=fl)

    assert res1.distance == res2.distance
    assert res1.travel_time == res2.travel_time
    assert res1.path == res2.path
