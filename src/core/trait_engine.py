"""Trait Engine: Aggregation, satisfaction scoring, and realizability verification."""
from __future__ import annotations
from itertools import combinations
from typing import List, Tuple, Dict, Set, Optional, Any
import numpy as np

from ..models.robot import Robot, RobotTraits, TRAIT_NAMES, ZONE_ACCESS
from ..models.task import Task, TaskRequirements
from ..models.coalition import Coalition


class TraitEngine:
    """Core mathematical engine for trait-based multi-robot reasoning."""

    @staticmethod
    def aggregate_coalition_traits(coalition: Coalition) -> RobotTraits:
        """Compute the aggregate trait vector Q = aggregate({q_i}) for coalition members."""
        return coalition.aggregate_traits

    @staticmethod
    def enumerate_all_coalitions(robots: List[Robot], max_size: int = 3) -> List[Coalition]:
        """Enumerate all valid non-empty coalitions up to max_size.
        
        Args:
            robots: Pool of physical robots.
            max_size: Maximum coalition team size (default 3).
            
        Returns:
            List of unique Coalition instances, ordered by size then id.
        """
        all_coalitions: List[Coalition] = []
        n_robots = len(robots)
        effective_max = min(max_size, n_robots)

        for k in range(1, effective_max + 1):
            for subset in combinations(robots, k):
                all_coalitions.append(Coalition(subset))

        return all_coalitions

    @staticmethod
    def compute_realizable_trait_set(robots: List[Robot], max_size: int = 3) -> List[Tuple[Coalition, np.ndarray]]:
        """Compute the set of all physically realizable trait vectors for the given robot pool.
        
        Crucial for E-ITAGS Realizability-Aware Active Learning:
        Only trait vectors that can actually be formed by combining physical robots
        exist in the realizable search space.
        """
        coalitions = TraitEngine.enumerate_all_coalitions(robots, max_size)
        return [(c, c.to_vector()) for c in coalitions]

    @staticmethod
    def is_realizable(
        target_traits: RobotTraits,
        robots: List[Robot],
        trait_subset: Optional[List[str]] = None,
        tolerance: float = 1e-3,
        allow_surplus: bool = False,
    ) -> Tuple[bool, Optional[Coalition]]:
        """Check whether a target trait vector is realizable by any subset of the robot pool.
        
        Args:
            target_traits: Target trait specifications.
            robots: Available physical robots.
            trait_subset: Optional list of trait names to check (e.g. ['payload_kg']).
                          If None, checks all dimensions in TRAIT_NAMES.
            tolerance: Float comparison tolerance.
            allow_surplus: If True, checks whether any coalition meets or exceeds the target
                           traits (Q(C) >= q* - tolerance).
                           If False (default), checks for exact physical realizability
                           (Q(C) == q* within tolerance).
        """
        if trait_subset is None:
            indices = list(range(len(TRAIT_NAMES)))
        else:
            indices = [TRAIT_NAMES.index(name) for name in trait_subset if name in TRAIT_NAMES]

        target_vec = target_traits.to_vector()[indices]
        for coalition, c_vec in TraitEngine.compute_realizable_trait_set(robots):
            if allow_surplus:
                if np.all(c_vec[indices] >= target_vec - tolerance):
                    return True, coalition
            else:
                if np.allclose(target_vec, c_vec[indices], atol=tolerance):
                    return True, coalition
        return False, None

    @staticmethod
    def filter_zone_accessible_coalitions(
        coalitions: List[Coalition],
        zone_name: str,
    ) -> List[Coalition]:
        """Filter coalitions to only those where ALL members can access the given zone.

        Args:
            coalitions: List of coalitions to filter.
            zone_name: Target warehouse zone name.

        Returns:
            Subset of coalitions where every member robot is allowed in the zone.
        """
        return [c for c in coalitions if c.can_all_access_zone(zone_name)]

    @staticmethod
    def compute_trait_satisfaction(coalition: Coalition, task: Task) -> Tuple[float, Dict[str, float]]:
        """Calculate the trait satisfaction score S(C, task) in [0, 1].

        Zone access is enforced as a hard constraint: if any coalition member
        cannot physically access the task's zone, the score is 0.0.
        
        Returns:
            overall_score: Float in [0.0, 1.0], where 1.0 means all requirements fully satisfied/exceeded.
            breakdown: Per-dimension satisfaction dictionary for debugging & visibility.
        """
        breakdown: Dict[str, float] = {}

        # Zone access hard constraint: every member must be able to enter the task zone
        if not coalition.can_all_access_zone(task.zone):
            blocked = coalition.get_zone_blocked_members(task.zone)
            breakdown["zone_access"] = 0.0
            # Fill remaining dimensions with 0.0 for visibility
            for key in ["payload", "lidar_2d", "camera_rgb", "vertical_lift", "narrow_access"]:
                breakdown[key] = 0.0
            return 0.0, breakdown

        breakdown["zone_access"] = 1.0

        agg = coalition.aggregate_traits
        req = task.requirements

        # 1. Payload satisfaction: ratio of available to needed (capped at 1.0)
        if req.min_payload_kg > 0:
            payload_sat = min(1.0, agg.payload_kg / req.min_payload_kg)
        else:
            payload_sat = 1.0
        breakdown["payload"] = payload_sat

        # 2. LiDAR 2D satisfaction
        if req.min_has_lidar_2d > 0:
            lidar_sat = 1.0 if agg.has_lidar_2d >= req.min_has_lidar_2d else 0.0
        else:
            lidar_sat = 1.0
        breakdown["lidar_2d"] = lidar_sat

        # 3. Camera RGB satisfaction
        if req.min_has_camera_rgb > 0:
            cam_sat = min(1.0, agg.has_camera_rgb / req.min_has_camera_rgb)
        else:
            cam_sat = 1.0
        breakdown["camera_rgb"] = cam_sat

        # 4. Vertical mast lift satisfaction
        if req.min_vertical_lift > 0:
            lift_sat = 1.0 if agg.vertical_lift >= req.min_vertical_lift else 0.0
        else:
            lift_sat = 1.0
        breakdown["vertical_lift"] = lift_sat

        # 5. Narrow space access satisfaction
        if req.min_narrow_space_access > 0:
            narrow_sat = 1.0 if agg.narrow_space_access >= req.min_narrow_space_access else 0.0
        else:
            narrow_sat = 1.0
        breakdown["narrow_access"] = narrow_sat

        # Combined satisfaction: geometric mean or weighted average
        # In trait-based MRTA, missing a hard required trait renders task satisfaction zero
        # If any mandatory binary trait is missing (0.0), task cannot be completed
        has_mandatory_failure = (
            (req.min_has_lidar_2d > 0 and lidar_sat == 0.0) or
            (req.min_has_camera_rgb > 0 and cam_sat == 0.0) or
            (req.min_vertical_lift > 0 and lift_sat == 0.0) or
            (req.min_narrow_space_access > 0 and narrow_sat == 0.0) or
            (req.min_payload_kg > 0 and agg.payload_kg < req.min_payload_kg)
        )

        if has_mandatory_failure:
            overall_score = 0.0
        else:
            # Average satisfaction across all evaluated dimensions
            # (zone_access is always 1.0 here since we passed the hard check above)
            trait_scores = [v for k, v in breakdown.items() if k != "zone_access"]
            overall_score = float(np.mean(trait_scores))

        return overall_score, breakdown
