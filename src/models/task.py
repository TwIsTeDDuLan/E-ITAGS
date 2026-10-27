"""Task and TaskRequirement models."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from .robot import TRAIT_NAMES


@dataclass(frozen=True)
class TaskRequirements:
    """Trait demands for a task.
    
    Can represent soft desires or minimum thresholds:
    - min_payload_kg: minimum collective payload needed
    - min_has_lidar_2d: whether LiDAR 2D perception is needed (0.0 to 1.0)
    - min_has_camera_rgb: whether visual feed is needed (0.0 to 1.0)
    - min_vertical_lift: whether high mast lifting is needed (0.0 to 1.0)
    - min_narrow_space_access: whether tight-space/underride access is needed (0.0 to 1.0)
    """
    min_payload_kg: float = 0.0
    min_has_lidar_2d: float = 0.0
    min_has_camera_rgb: float = 0.0
    min_vertical_lift: float = 0.0
    min_narrow_space_access: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Return requirement vector matching TRAIT_NAMES order (excluding speed)."""
        return np.array([
            self.min_payload_kg,
            self.min_has_lidar_2d,
            self.min_has_camera_rgb,
            self.min_vertical_lift,
            self.min_narrow_space_access,
        ], dtype=np.float64)

    def to_dict(self) -> Dict[str, float]:
        return {
            "min_payload_kg": self.min_payload_kg,
            "min_has_lidar_2d": self.min_has_lidar_2d,
            "min_has_camera_rgb": self.min_has_camera_rgb,
            "min_vertical_lift": self.min_vertical_lift,
            "min_narrow_space_access": self.min_narrow_space_access,
        }


@dataclass(frozen=True)
class Task:
    """A multi-robot task t_j in STEAM/E-ITAGS."""
    id: str
    name: str
    zone: str
    duration_seconds: float
    requirements: TaskRequirements
    description: str = ""
    deadline_seconds: Optional[float] = None
    release_time_seconds: float = 0.0

    def __repr__(self) -> str:
        return f"Task({self.id}: '{self.name}', zone={self.zone}, dur={self.duration_seconds}s)"
