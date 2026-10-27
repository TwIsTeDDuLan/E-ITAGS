"""Coalition data model representing a team of cooperating robots."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, FrozenSet, Set
import numpy as np
from .robot import Robot, RobotTraits, TRAIT_NAMES


@dataclass(frozen=True)
class Coalition:
    """A subset of robots C_k = {r_1, r_2, ...} formed to perform a task.
    
    Immutable and hashable so it can be stored in sets or dictionary keys.
    """
    members: Tuple[Robot, ...]

    def __init__(self, robots: Tuple[Robot, ...] | Set[Robot] | list[Robot]):
        # Store sorted by id for canonical hash and equality
        sorted_members = tuple(sorted(robots, key=lambda r: r.id))
        object.__setattr__(self, "members", sorted_members)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def robot_ids(self) -> Tuple[str, ...]:
        return tuple(r.id for r in self.members)

    @property
    def aggregate_traits(self) -> RobotTraits:
        """Aggregate the traits of member robots according to domain physics:
        
        - payload_kg: sum of payloads
        - has_lidar_2d: max (1.0 if any member has 2D LiDAR)
        - has_camera_rgb: max (1.0 if drone present, 0.5 if iw_hub, etc.)
        - vertical_lift: max (1.0 if forklift present)
        - narrow_space_access: min of ground members (or 1.0 if only IW Hub/Drone)
        - speed_mps: min of members (team moves at pace of slowest member)
        """
        if not self.members:
            return RobotTraits()

        total_payload = sum(r.traits.payload_kg for r in self.members)
        has_lidar = max(r.traits.has_lidar_2d for r in self.members)
        has_camera = max(r.traits.has_camera_rgb for r in self.members)
        has_lift = max(r.traits.vertical_lift for r in self.members)

        # Narrow space access: true if no bulky ground vehicle (forklift) is forced in
        has_narrow = min(r.traits.narrow_space_access for r in self.members if r.traits.payload_kg > 0) \
            if any(r.traits.payload_kg > 0 for r in self.members) else 1.0

        # Transit speed of coalition is dictated by slowest member
        coalition_speed = min(r.traits.speed_mps for r in self.members)

        return RobotTraits(
            payload_kg=total_payload,
            has_lidar_2d=has_lidar,
            has_camera_rgb=has_camera,
            vertical_lift=has_lift,
            narrow_space_access=float(has_narrow),
            speed_mps=coalition_speed,
        )

    def to_vector(self) -> np.ndarray:
        return self.aggregate_traits.to_vector()

    def contains(self, robot: Robot) -> bool:
        return robot in self.members

    def contains_id(self, robot_id: str) -> bool:
        return robot_id in self.robot_ids

    def overlaps_with(self, other: Coalition) -> bool:
        """True if this coalition shares any robot with another coalition."""
        return bool(set(self.robot_ids) & set(other.robot_ids))

    def can_all_access_zone(self, zone_name: str) -> bool:
        """Check if ALL members of this coalition can physically operate in the given zone.

        A coalition can only be assigned to a task in a zone if every member robot
        is allowed to enter that zone (per ZONE_ACCESS constraints).
        """
        return all(r.can_access_zone(zone_name) for r in self.members)

    def get_zone_blocked_members(self, zone_name: str) -> list[Robot]:
        """Return list of members that CANNOT access the given zone."""
        return [r for r in self.members if not r.can_access_zone(zone_name)]

    def __repr__(self) -> str:
        names = "+".join(r.name for r in self.members)
        return f"[{names}]" if names else "[]"

    def __lt__(self, other: Coalition) -> bool:
        if not isinstance(other, Coalition):
            return NotImplemented
        return (len(self.members), self.robot_ids) < (len(other.members), other.robot_ids)

