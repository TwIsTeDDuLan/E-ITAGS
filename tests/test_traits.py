"""Unit tests for Phase 0 and Phase 1: Models, Traits, Coalitions, and TraitEngine."""
import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.robot import Robot, RobotTraits, RobotType
from src.models.task import Task, TaskRequirements
from src.models.coalition import Coalition
from src.core.trait_engine import TraitEngine


@pytest.fixture
def robot_fleet():
    fl = Robot.create_forklift("R_FL1", "Forklift-1")
    dr = Robot.create_drone("R_DR1", "Drone-1")
    iw = Robot.create_iw_hub("R_IW1", "IWHub-1")
    return [fl, dr, iw]


def test_robot_traits():
    fl = Robot.create_forklift("fl", "Forklift")
    dr = Robot.create_drone("dr", "Drone")
    iw = Robot.create_iw_hub("iw", "IW Hub")

    # Forklift
    assert fl.traits.payload_kg == 1000.0
    assert fl.traits.has_lidar_2d == 1.0
    assert fl.traits.vertical_lift == 1.0
    assert fl.traits.narrow_space_access == 0.0

    # Drone
    assert dr.traits.payload_kg == 0.0
    assert dr.traits.has_camera_rgb == 1.0
    assert dr.traits.speed_mps == 5.0

    # IW Hub has 2D LiDAR, 1000kg payload, narrow space access, underride camera
    assert iw.traits.payload_kg == 1000.0
    assert iw.traits.has_lidar_2d == 1.0
    assert iw.traits.has_camera_rgb == 0.5
    assert iw.traits.vertical_lift == 0.0
    assert iw.traits.narrow_space_access == 1.0


def test_coalition_aggregation(robot_fleet):
    fl, dr, iw = robot_fleet

    # Single robot coalition
    c_fl = Coalition([fl])
    assert c_fl.aggregate_traits.payload_kg == 1000.0
    assert c_fl.aggregate_traits.vertical_lift == 1.0

    # Two robot coalition: Forklift + Drone
    c_fl_dr = Coalition([fl, dr])
    assert c_fl_dr.aggregate_traits.payload_kg == 1000.0  # 1000 + 0
    assert c_fl_dr.aggregate_traits.has_camera_rgb == 1.0  # from drone
    assert c_fl_dr.aggregate_traits.has_lidar_2d == 1.0    # from forklift
    assert c_fl_dr.aggregate_traits.vertical_lift == 1.0   # from forklift
    assert c_fl_dr.aggregate_traits.speed_mps == 2.0       # min(2.0, 5.0)

    # Two robot coalition: IW Hub + Drone
    c_iw_dr = Coalition([iw, dr])
    assert c_iw_dr.aggregate_traits.payload_kg == 1000.0
    assert c_iw_dr.aggregate_traits.narrow_space_access == 1.0
    assert c_iw_dr.aggregate_traits.has_lidar_2d == 1.0


def test_coalition_enumeration(robot_fleet):
    # 3 robots -> 2^3 - 1 = 7 non-empty coalitions
    coalitions = TraitEngine.enumerate_all_coalitions(robot_fleet, max_size=3)
    assert len(coalitions) == 7

    sizes = [c.size for c in coalitions]
    assert sizes.count(1) == 3
    assert sizes.count(2) == 3
    assert sizes.count(3) == 1


def test_trait_satisfaction(robot_fleet):
    fl, dr, iw = robot_fleet
    c_fl = Coalition([fl])
    c_dr = Coalition([dr])
    c_iw = Coalition([iw])
    c_fl_dr = Coalition([fl, dr])

    # T1: High rack heavy move (requires vertical mast lift and payload)
    t1 = Task(
        id="T1", name="Rack Move", zone="receiving", duration_seconds=100,
        requirements=TaskRequirements(min_payload_kg=800, min_vertical_lift=1.0, min_has_lidar_2d=1.0)
    )
    score_fl, _ = TraitEngine.compute_trait_satisfaction(c_fl, t1)
    score_dr, _ = TraitEngine.compute_trait_satisfaction(c_dr, t1)
    score_iw, _ = TraitEngine.compute_trait_satisfaction(c_iw, t1)

    assert score_fl == 1.0
    assert score_dr == 0.0  # missing payload, lift, lidar
    assert score_iw == 0.0  # missing vertical lift

    # T2: Aerial survey (requires camera)
    t2 = Task(
        id="T2", name="Survey", zone="sorting", duration_seconds=50,
        requirements=TaskRequirements(min_has_camera_rgb=1.0)
    )
    score_fl_t2, _ = TraitEngine.compute_trait_satisfaction(c_fl, t2)
    score_dr_t2, _ = TraitEngine.compute_trait_satisfaction(c_dr, t2)
    assert score_dr_t2 == 1.0
    assert score_fl_t2 == 0.0

    # T3: Tight stack move (requires narrow access, payload, lidar corner detection)
    t3 = Task(
        id="T3", name="Tight Stack", zone="narrow_aisle", duration_seconds=80,
        requirements=TaskRequirements(min_payload_kg=600, min_has_lidar_2d=1.0, min_narrow_space_access=1.0)
    )
    score_iw_t3, _ = TraitEngine.compute_trait_satisfaction(c_iw, t3)
    score_fl_t3, _ = TraitEngine.compute_trait_satisfaction(c_fl, t3)
    assert score_iw_t3 == 1.0
    assert score_fl_t3 == 0.0  # Forklift cannot fit in narrow stacks

    # T4: Collaborative task (requires both lift, payload, AND high-res camera)
    t4 = Task(
        id="T4", name="Guided Sort", zone="sorting", duration_seconds=120,
        requirements=TaskRequirements(min_payload_kg=800, min_vertical_lift=1.0, min_has_camera_rgb=1.0, min_has_lidar_2d=1.0)
    )
    score_fl_t4, _ = TraitEngine.compute_trait_satisfaction(c_fl, t4)
    score_dr_t4, _ = TraitEngine.compute_trait_satisfaction(c_dr, t4)
    score_coalition, _ = TraitEngine.compute_trait_satisfaction(c_fl_dr, t4)

    assert score_fl_t4 == 0.0  # missing camera
    assert score_dr_t4 == 0.0  # missing payload and lift
    assert score_coalition == 1.0  # Coalition succeeds!


def test_realizability(robot_fleet):
    realizable_set = TraitEngine.compute_realizable_trait_set(robot_fleet)
    assert len(realizable_set) == 7

    # Test an unrealizable trait vector (e.g. 1500kg payload)
    unrealizable = RobotTraits(payload_kg=1500.0)
    is_realizable, _ = TraitEngine.is_realizable(unrealizable, robot_fleet)
    assert not is_realizable

    # Test a realizable trait vector (e.g. 2000kg payload from Forklift + IW Hub)
    fl, dr, iw = robot_fleet
    c_fl_iw = Coalition([fl, iw])
    is_realizable_2k, matched_c = TraitEngine.is_realizable(c_fl_iw.aggregate_traits, robot_fleet)
    assert is_realizable_2k
    assert matched_c == c_fl_iw

    # Test allow_surplus behavior (e.g. 100kg payload)
    target_100kg = RobotTraits(payload_kg=100.0)
    # Exact check: False (no coalition has exactly 100kg)
    is_real_exact, _ = TraitEngine.is_realizable(target_100kg, robot_fleet, trait_subset=["payload_kg"], allow_surplus=False)
    assert not is_real_exact

    # Surplus check: True (Forklift/IWHub has 1000kg >= 100kg)
    is_real_surplus, matched_surplus = TraitEngine.is_realizable(target_100kg, robot_fleet, trait_subset=["payload_kg"], allow_surplus=True)
    assert is_real_surplus
    assert matched_surplus is not None
    assert matched_surplus.aggregate_traits.payload_kg >= 100.0

    # Surplus check with excessive requirement: False (2500kg > max fleet 2000kg)
    target_2500kg = RobotTraits(payload_kg=2500.0)
    is_real_exceed, _ = TraitEngine.is_realizable(target_2500kg, robot_fleet, trait_subset=["payload_kg"], allow_surplus=True)
    assert not is_real_exceed

def test_zone_access_individual_robots():
    """Test that each robot type respects its zone access constraints."""
    fl = Robot.create_forklift("fl", "Forklift")
    dr = Robot.create_drone("dr", "Drone")
    iw = Robot.create_iw_hub("iw", "IW Hub")

    # Drone can access ALL zones
    for zone in ["receiving", "sorting", "staging", "narrow_aisle", "shipping"]:
        assert dr.can_access_zone(zone), f"Drone should access {zone}"

    # Forklift can access everything EXCEPT narrow_aisle
    assert fl.can_access_zone("receiving")
    assert fl.can_access_zone("sorting")
    assert fl.can_access_zone("staging")
    assert fl.can_access_zone("shipping")
    assert not fl.can_access_zone("narrow_aisle"), "Forklift must NOT access narrow_aisle"

    # IW Hub can only access sorting, narrow_aisle, staging
    assert iw.can_access_zone("sorting")
    assert iw.can_access_zone("narrow_aisle")
    assert iw.can_access_zone("staging")
    assert not iw.can_access_zone("receiving"), "IW Hub must NOT access receiving"
    assert not iw.can_access_zone("shipping"), "IW Hub must NOT access shipping"


def test_zone_access_coalitions():
    """Test that coalition zone access requires ALL members to have access."""
    fl = Robot.create_forklift("fl", "Forklift")
    dr = Robot.create_drone("dr", "Drone")
    iw = Robot.create_iw_hub("iw", "IW Hub")

    # Forklift + Drone coalition: can access receiving (both allowed), NOT narrow_aisle (forklift blocked)
    c_fl_dr = Coalition([fl, dr])
    assert c_fl_dr.can_all_access_zone("receiving")
    assert c_fl_dr.can_all_access_zone("sorting")
    assert not c_fl_dr.can_all_access_zone("narrow_aisle")

    # IW Hub + Drone coalition: can access sorting, narrow_aisle, staging; NOT receiving, shipping
    c_iw_dr = Coalition([iw, dr])
    assert c_iw_dr.can_all_access_zone("sorting")
    assert c_iw_dr.can_all_access_zone("narrow_aisle")
    assert c_iw_dr.can_all_access_zone("staging")
    assert not c_iw_dr.can_all_access_zone("receiving")
    assert not c_iw_dr.can_all_access_zone("shipping")

    # Forklift + IW Hub: sorting and staging only (intersection)
    c_fl_iw = Coalition([fl, iw])
    assert c_fl_iw.can_all_access_zone("sorting")
    assert c_fl_iw.can_all_access_zone("staging")
    assert not c_fl_iw.can_all_access_zone("narrow_aisle")  # forklift blocked
    assert not c_fl_iw.can_all_access_zone("receiving")     # iw_hub blocked
    assert not c_fl_iw.can_all_access_zone("shipping")      # iw_hub blocked


def test_zone_access_in_satisfaction():
    """Test that compute_trait_satisfaction returns 0.0 when coalition can't access task zone."""
    fl = Robot.create_forklift("fl", "Forklift")
    iw = Robot.create_iw_hub("iw", "IW Hub")

    c_fl = Coalition([fl])

    # Forklift trying to do a narrow_aisle task — zone blocked regardless of traits
    narrow_task = Task(
        id="T_NA", name="Narrow Task", zone="narrow_aisle", duration_seconds=100,
        requirements=TaskRequirements(min_payload_kg=500, min_has_lidar_2d=1.0)
    )
    score, breakdown = TraitEngine.compute_trait_satisfaction(c_fl, narrow_task)
    assert score == 0.0, "Forklift cannot access narrow_aisle"
    assert breakdown["zone_access"] == 0.0

    # IW Hub doing narrow_aisle task — zone accessible, traits satisfied
    c_iw = Coalition([iw])
    score_iw, breakdown_iw = TraitEngine.compute_trait_satisfaction(c_iw, narrow_task)
    assert score_iw > 0.0, "IW Hub can access narrow_aisle and has required traits"
    assert breakdown_iw["zone_access"] == 1.0

    # IW Hub trying receiving zone — zone blocked
    receiving_task = Task(
        id="T_RV", name="Receiving Task", zone="receiving", duration_seconds=100,
        requirements=TaskRequirements(min_payload_kg=500)
    )
    score_iw_rv, breakdown_iw_rv = TraitEngine.compute_trait_satisfaction(c_iw, receiving_task)
    assert score_iw_rv == 0.0, "IW Hub cannot access receiving zone"
    assert breakdown_iw_rv["zone_access"] == 0.0


def test_filter_zone_accessible_coalitions():
    """Test that filter_zone_accessible_coalitions correctly filters."""
    fl = Robot.create_forklift("fl", "Forklift")
    dr = Robot.create_drone("dr", "Drone")
    iw = Robot.create_iw_hub("iw", "IW Hub")

    coalitions = TraitEngine.enumerate_all_coalitions([fl, dr, iw], max_size=3)
    assert len(coalitions) == 7

    # narrow_aisle: only coalitions without forklift AND without IW Hub being blocked
    # Actually: forklift is blocked from narrow_aisle. Drone and IW Hub can enter.
    narrow_accessible = TraitEngine.filter_zone_accessible_coalitions(coalitions, "narrow_aisle")
    for c in narrow_accessible:
        assert c.can_all_access_zone("narrow_aisle")
        # No forklift should be in any of these coalitions
        for r in c.members:
            assert r.robot_type != RobotType.FORKLIFT

    # receiving: IW Hub is blocked from receiving
    receiving_accessible = TraitEngine.filter_zone_accessible_coalitions(coalitions, "receiving")
    for c in receiving_accessible:
        for r in c.members:
            assert r.robot_type != RobotType.IW_HUB


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

