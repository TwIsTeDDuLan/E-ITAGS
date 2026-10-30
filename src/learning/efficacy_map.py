"""Continuous Trait-Efficacy Modeling and Warehouse Simulation Oracle."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import math

from ..models.robot import RobotTraits, TRAIT_NAMES
from ..models.task import Task
from ..models.coalition import Coalition


class EfficacyMap(ABC):
    """Abstract base class for mapping coalition traits to continuous task performance."""

    @abstractmethod
    def predict(self, coalition: Coalition, task: Task) -> float:
        """Predict the task execution efficacy in [0.0, 1.0] for a given coalition and task."""
        pass

    @abstractmethod
    def predict_vector(self, trait_vector: np.ndarray, task: Task) -> float:
        """Predict efficacy directly from a 6D aggregate trait vector."""
        pass


class GroundTruthEfficacyOracle(EfficacyMap):
    """Realistic warehouse task simulation oracle providing true performance evaluations.
    
    Models continuous physical phenomena:
    1. Sigmoid payload stability and safety margins.
    2. Camera resolution and visual clarity scaling.
    3. Multi-robot sensor fusion and aerial-ground synergy bonuses.
    4. Hard physical feasibility thresholds (mast lift and narrow aisle clearance).
    5. Realistic stochastic observation noise: xi_obs ~ N(xi*, sigma_noise^2).
    """

    def __init__(self, noise_std: float = 0.02, seed: Optional[int] = 42):
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def evaluate_true_efficacy(self, traits: RobotTraits, task: Task) -> float:
        """Evaluate deterministic underlying ground-truth efficacy xi*(Q, task)."""
        req = task.requirements

        # 1. Hard physical thresholds: mast lift and narrow access
        if req.min_vertical_lift > 0 and traits.vertical_lift < req.min_vertical_lift:
            return 0.0

        if req.min_narrow_space_access > 0 and traits.narrow_space_access < req.min_narrow_space_access:
            return 0.0

        # 2. Continuous Payload Response: Sigmoid curve
        if req.min_payload_kg > 0:
            # Centered at required payload, steep transition
            margin = traits.payload_kg - req.min_payload_kg
            if traits.payload_kg < req.min_payload_kg:
                # Insufficient payload drops off steeply
                payload_factor = 1.0 / (1.0 + np.exp(-0.015 * margin))
            else:
                # Excess payload provides a modest structural safety margin
                payload_factor = 0.85 + 0.15 * min(1.0, margin / 500.0)
        else:
            payload_factor = 1.0

        # 3. Vision Perception Response (RGB Camera)
        if req.min_has_camera_rgb > 0:
            cam_ratio = traits.has_camera_rgb / req.min_has_camera_rgb
            cam_factor = min(1.0, cam_ratio ** 0.8)
        else:
            # Extra visual clarity gives a subtle 5% quality bonus
            cam_factor = 1.0 + 0.05 * traits.has_camera_rgb

        # 4. LiDAR Perception Response
        if req.min_has_lidar_2d > 0:
            lidar_factor = 1.0 if traits.has_lidar_2d >= 1.0 else 0.2
        else:
            lidar_factor = 1.0

        # 5. Multi-Robot Collaboration / Aerial-Ground Synergy Bonus
        synergy_bonus = 0.0
        # If task requires both heavy lifting and vision (like T4 Collaborative Pallet Sort),
        # having both a high-reach mast and overhead aerial drone camera unlocks peak synergy
        if req.min_vertical_lift > 0 and req.min_has_camera_rgb >= 0.8:
            if traits.vertical_lift >= 1.0 and traits.has_camera_rgb >= 1.0:
                synergy_bonus = 0.12  # Significant collaborative boost

        # 6. Combined Base Efficacy
        base_efficacy = (0.45 * payload_factor + 0.35 * cam_factor + 0.20 * lidar_factor) + synergy_bonus
        return float(np.clip(base_efficacy, 0.0, 1.0))

    def predict(self, coalition: Coalition, task: Task) -> float:
        """Evaluate coalition on task respecting zone access constraints with observation noise."""
        if not coalition.can_all_access_zone(task.zone):
            return 0.0

        true_eff = self.evaluate_true_efficacy(coalition.aggregate_traits, task)
        if true_eff <= 0.0:
            return 0.0

        if self.noise_std > 0:
            noise = float(self.rng.normal(0.0, self.noise_std))
            return float(np.clip(true_eff + noise, 0.0, 1.0))
        return true_eff

    def predict_vector(self, trait_vector: np.ndarray, task: Task) -> float:
        """Evaluate efficacy directly from trait vector without noise."""
        traits = RobotTraits(
            payload_kg=float(trait_vector[0]),
            has_lidar_2d=float(trait_vector[1]),
            has_camera_rgb=float(trait_vector[2]),
            vertical_lift=float(trait_vector[3]),
            narrow_space_access=float(trait_vector[4]),
            speed_mps=float(trait_vector[5]) if len(trait_vector) > 5 else 2.0,
        )
        return self.evaluate_true_efficacy(traits, task)


class HandCraftedEfficacyMap(EfficacyMap):
    """Deterministic, noise-free efficacy model for benchmarking and testing."""

    def __init__(self):
        self.oracle = GroundTruthEfficacyOracle(noise_std=0.0)

    def predict(self, coalition: Coalition, task: Task) -> float:
        return self.oracle.predict(coalition, task)

    def predict_vector(self, trait_vector: np.ndarray, task: Task) -> float:
        return self.oracle.predict_vector(trait_vector, task)
