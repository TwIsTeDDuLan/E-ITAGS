"""Robot and Trait data models."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any
import numpy as np


TRAIT_NAMES = [
    "payload_kg",           # Max carrying capacity (kg)
    "has_lidar_2d",         # 2D LiDAR coverage (0.0 or 1.0) for safety/corner detection
    "has_camera_rgb",       # RGB camera feed (0.0 to 1.0)
    "vertical_lift",        # High mast vertical lift (0.0 or 1.0)
    "narrow_space_access",  # Floor-level narrow aisle / under-stack access (0.0 or 1.0)
    "speed_mps",            # Maximum linear transit speed (m/s)
]


class RobotType(str, Enum):
    FORKLIFT = "forklift"
    DRONE = "drone"
    IW_HUB = "iw_hub"


# Zone access constraints per robot type.
# Maps each RobotType to the set of warehouse zone names it can physically operate in.
# Drone: aerial — unrestricted access to all zones
# Forklift: large mast-lift vehicle — cannot enter narrow_aisle (clearance too low)
# IW Hub: low-profile AMR — restricted to sorting, narrow_aisle, staging (its operational domain)
ZONE_ACCESS: Dict[RobotType, frozenset[str]] = {
    RobotType.FORKLIFT: frozenset({"receiving", "sorting", "staging", "shipping"}),
    RobotType.DRONE:    frozenset({"receiving", "sorting", "staging", "narrow_aisle", "shipping"}),
    RobotType.IW_HUB:   frozenset({"sorting", "narrow_aisle", "staging"}),
}


@dataclass(frozen=True)
class RobotTraits:
    """Individual capability traits vector q_i in R^d."""
    payload_kg: float = 0.0
    has_lidar_2d: float = 0.0
    has_camera_rgb: float = 0.0
    vertical_lift: float = 0.0
    narrow_space_access: float = 0.0
    speed_mps: float = 1.0

    def to_vector(self) -> np.ndarray:
        """Return numeric vector in canonical TRAIT_NAMES order."""
        return np.array([
            self.payload_kg,
            self.has_lidar_2d,
            self.has_camera_rgb,
            self.vertical_lift,
            self.narrow_space_access,
            self.speed_mps,
        ], dtype=np.float64)

    def to_dict(self) -> Dict[str, float]:
        return {
            "payload_kg": self.payload_kg,
            "has_lidar_2d": self.has_lidar_2d,
            "has_camera_rgb": self.has_camera_rgb,
            "vertical_lift": self.vertical_lift,
            "narrow_space_access": self.narrow_space_access,
            "speed_mps": self.speed_mps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RobotTraits:
        return cls(
            payload_kg=float(data.get("payload_kg", 0.0)),
            has_lidar_2d=float(data.get("has_lidar_2d", 0.0)),
            has_camera_rgb=float(data.get("has_camera_rgb", 0.0)),
            vertical_lift=float(data.get("vertical_lift", 0.0)),
            narrow_space_access=float(data.get("narrow_space_access", 0.0)),
            speed_mps=float(data.get("speed_mps", 1.0)),
        )


@dataclass(frozen=True)
class Robot:
    """A physical robot agent r_i with known traits and base parameters."""
    id: str
    name: str
    robot_type: RobotType
    traits: RobotTraits
    initial_zone: str = "staging"

    @classmethod
    def create_forklift(cls, id: str, name: str, initial_zone: str = "staging") -> Robot:
        """Forklift: 1000kg, 2D LiDAR, Mast vertical lift, Ackermann transit."""
        traits = RobotTraits(
            payload_kg=1000.0,
            has_lidar_2d=1.0,
            has_camera_rgb=0.0,
            vertical_lift=1.0,
            narrow_space_access=0.0,
            speed_mps=2.0,
        )
        return cls(id=id, name=name, robot_type=RobotType.FORKLIFT, traits=traits, initial_zone=initial_zone)

    @classmethod
    def create_drone(cls, id: str, name: str, initial_zone: str = "staging") -> Robot:
        """Drone: Overhead high-resolution RGB visual feed, agile speed, zero payload."""
        traits = RobotTraits(
            payload_kg=0.0,
            has_lidar_2d=0.0,
            has_camera_rgb=1.0,
            vertical_lift=0.0,
            narrow_space_access=0.0,
            speed_mps=5.0,
        )
        return cls(id=id, name=name, robot_type=RobotType.DRONE, traits=traits, initial_zone=initial_zone)

    @classmethod
    def create_iw_hub(cls, id: str, name: str, initial_zone: str = "staging") -> Robot:
        """IW Hub AMR: Low-profile 220mm, 1000kg, Dual 2D LiDAR scanners, docking camera."""
        traits = RobotTraits(
            payload_kg=1000.0,
            has_lidar_2d=1.0,
            has_camera_rgb=0.5,
            vertical_lift=0.0,
            narrow_space_access=1.0,
            speed_mps=2.0,
        )
        return cls(id=id, name=name, robot_type=RobotType.IW_HUB, traits=traits, initial_zone=initial_zone)

    def can_access_zone(self, zone_name: str) -> bool:
        """Check if this robot type is physically allowed to operate in the given zone.

        Uses the ZONE_ACCESS registry. Unknown zones are treated as inaccessible.
        """
        allowed = ZONE_ACCESS.get(self.robot_type, frozenset())
        return zone_name in allowed

    def __repr__(self) -> str:
        return f"{self.name}({self.robot_type.value}, id={self.id})"
