"""Realizability-Aware Active Learning with Gaussian Processes for Trait-Efficacy Modeling."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel

from ..models.task import Task
from ..models.coalition import Coalition
from .efficacy_map import EfficacyMap, GroundTruthEfficacyOracle


class SamplingStrategy(str, Enum):
    """The four active learning acquisition strategies analyzed in the paper."""
    EXACT = "exact"                    # Evaluates acquisition function directly on discrete realizable set Ω
    UNCONSTRAINED = "unconstrained"    # Optimizes in continuous R^U, then projects onto nearest in Ω
    BOX = "box"                        # Optimizes within continuous bounding box [0, q_max], then projects onto Ω
    RANDOM = "random"                  # Randomly samples realizable coalitions without acquisition model


@dataclass
class QueryObservation:
    """Record of an active learning evaluation trial."""
    iteration: int
    task_id: str
    coalition: Coalition
    requested_traits: np.ndarray       # Continuous trait vector requested by acquisition
    actual_traits: np.ndarray          # Realizable trait vector actually executed
    observed_efficacy: float           # Measured performance label from simulation oracle
    projection_distance: float         # Euclidean distance between requested and realizable traits
    predicted_mean: float = 0.0        # GP prior mean before query
    predicted_std: float = 0.0         # GP prior uncertainty before query


class RealizabilityProjector:
    """Projects continuous trait vectors onto the nearest physically realizable coalition in Ω."""

    @staticmethod
    def project(
        target_traits: np.ndarray,
        candidates: List[Coalition],
        scale_weights: Optional[np.ndarray] = None,
    ) -> Tuple[Coalition, np.ndarray, float]:
        """Solve: A* = argmin_{C in candidates} ||Q(C) - target_traits||^2_W.

        Args:
            target_traits: Continuous 6D trait vector.
            candidates: Finite pool of realizable coalitions.
            scale_weights: Optional dimension scaling weights (e.g. 1/1000 for payload).

        Returns:
            (best_coalition, projected_traits_vector, weighted_distance)
        """
        if not candidates:
            raise ValueError("Candidate pool cannot be empty for realizability projection.")

        if scale_weights is None:
            # Normalize dimensions so payload in kg doesn't dominate camera [0, 1]
            # [payload, lidar, camera, lift, narrow, speed]
            scale_weights = np.array([1.0 / 1000.0, 1.0, 1.0, 1.0, 1.0, 1.0 / 5.0], dtype=np.float64)

        best_coalition = candidates[0]
        min_dist = float("inf")
        best_vec = candidates[0].to_vector()

        for c in candidates:
            c_vec = c.to_vector()
            # Truncate to matching dimensions if needed
            dim = min(len(target_traits), len(c_vec))
            diff = (c_vec[:dim] - target_traits[:dim]) * scale_weights[:dim]
            dist = float(np.sum(diff ** 2))
            if dist < min_dist:
                min_dist = dist
                best_coalition = c
                best_vec = c_vec

        return best_coalition, best_vec, float(np.sqrt(min_dist))


class LearnedEfficacyGP(EfficacyMap):
    """Gaussian Process model for an individual task's trait-efficacy map xi_m(Q)."""

    def __init__(self, task_id: str, random_state: int = 42):
        self.task_id = task_id
        self.random_state = random_state

        # Feature normalization scale: [payload, lidar, camera, lift, narrow, speed]
        self.scale = np.array([1000.0, 1.0, 1.0, 1.0, 1.0, 5.0], dtype=np.float64)

        # Matern 5/2 kernel with automatic relevance determination (ARD) + WhiteKernel noise
        kernel = C(1.0, (1e-3, 1e2)) * Matern(
            length_scale=np.ones(6),
            length_scale_bounds=(1e-2, 1e3),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-4,
            normalize_y=False,
            n_restarts_optimizer=3,
            random_state=random_state,
        )
        self.is_fitted = False
        self.X_train: List[np.ndarray] = []
        self.y_train: List[float] = []

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return np.atleast_2d(X) / self.scale

    def add_observation(self, trait_vector: np.ndarray, efficacy: float) -> None:
        """Add an observed training point and refit the Gaussian Process."""
        self.X_train.append(np.array(trait_vector, dtype=np.float64))
        self.y_train.append(float(efficacy))
        self.fit()

    def fit(self) -> None:
        """Fit the GP model to accumulated observations."""
        if len(self.X_train) == 0:
            return
        X = self._normalize(np.array(self.X_train))
        y = np.array(self.y_train, dtype=np.float64)
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.gp.fit(X, y)
        self.is_fitted = True

    def predict_with_uncertainty(self, trait_vector: np.ndarray) -> Tuple[float, float]:
        """Predict posterior mean mu and standard deviation sigma for a trait vector."""
        if not self.is_fitted:
            # Prior: uniform uninformative mean 0.5, high epistemic uncertainty 0.5
            return 0.5, 0.5

        X_norm = self._normalize(trait_vector)
        mean, std = self.gp.predict(X_norm, return_std=True)
        return float(np.clip(mean[0], 0.0, 1.0)), float(std[0])

    def predict(self, coalition: Coalition, task: Task) -> float:
        """Evaluate predicted mean efficacy for coalition."""
        mean, _ = self.predict_with_uncertainty(coalition.to_vector())
        return mean

    def predict_vector(self, trait_vector: np.ndarray, task: Task) -> float:
        mean, _ = self.predict_with_uncertainty(trait_vector)
        return mean


class ActiveLearner:
    """Realizability-Aware Active Learning manager coordinating queries, GPs, and oracles."""

    def __init__(
        self,
        tasks: List[Task],
        oracle: GroundTruthEfficacyOracle,
        random_seed: int = 42,
    ):
        self.tasks = tasks
        self.task_map = {t.id: t for t in tasks}
        self.oracle = oracle
        self.rng = np.random.default_rng(random_seed)
        self.projector = RealizabilityProjector()

        # Dedicated GP model per task
        self.models: Dict[str, LearnedEfficacyGP] = {
            t.id: LearnedEfficacyGP(t.id, random_state=random_seed) for t in tasks
        }
        self.history: List[QueryObservation] = []

    def select_query(
        self,
        task: Task,
        candidates: List[Coalition],
        strategy: SamplingStrategy = SamplingStrategy.EXACT,
        beta: float = 2.0,
    ) -> Tuple[Coalition, np.ndarray, float, float]:
        """Select the next query data point using the specified sampling strategy.

        Returns:
            (selected_coalition, requested_traits_vector, predicted_mean, predicted_std)
        """
        model = self.models[task.id]

        # 1. RANDOM BASELINE
        if strategy == SamplingStrategy.RANDOM:
            c = self.rng.choice(candidates)
            vec = c.to_vector()
            m, s = model.predict_with_uncertainty(vec)
            return c, vec, m, s

        # 2. EXACT DISCRETE REALIZABLE SAMPLING (Paper Strategy 2)
        if strategy == SamplingStrategy.EXACT:
            best_c = candidates[0]
            best_ucb = -float("inf")
            best_mean, best_std = 0.5, 0.5

            for c in candidates:
                vec = c.to_vector()
                m, s = model.predict_with_uncertainty(vec)
                # Upper Confidence Bound (UCB) acquisition
                ucb = m + beta * s
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_c = c
                    best_mean, best_std = m, s

            return best_c, best_c.to_vector(), best_mean, best_std

        # 3. BOX-CONSTRAINED SAMPLING (Paper Strategy 3)
        if strategy == SamplingStrategy.BOX:
            # Compute trait bounding box [0, q_max]
            candidate_vecs = np.array([c.to_vector() for c in candidates])
            q_max = np.max(candidate_vecs, axis=0)
            q_min = np.min(candidate_vecs, axis=0)

            # Sample random candidates within bounding box and find highest continuous UCB
            best_continuous = q_min.copy()
            best_ucb = -float("inf")

            # Dense sampling inside box
            for _ in range(100):
                sample = self.rng.uniform(q_min, q_max)
                m, s = model.predict_with_uncertainty(sample)
                ucb = m + beta * s
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_continuous = sample

            # Project continuous sample onto discrete realizable candidates
            proj_c, proj_vec, _ = self.projector.project(best_continuous, candidates)
            m, s = model.predict_with_uncertainty(proj_vec)
            return proj_c, best_continuous, m, s

        # 4. UNCONSTRAINED SAMPLING (Paper Strategy 1)
        if strategy == SamplingStrategy.UNCONSTRAINED:
            candidate_vecs = np.array([c.to_vector() for c in candidates])
            q_max = np.max(candidate_vecs, axis=0) * 1.5  # Expanded unconstrained domain
            q_min = np.zeros_like(q_max)

            best_continuous = q_min.copy()
            best_ucb = -float("inf")

            for _ in range(100):
                sample = self.rng.uniform(q_min, q_max)
                m, s = model.predict_with_uncertainty(sample)
                ucb = m + beta * s
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_continuous = sample

            # Project onto nearest realizable candidate
            proj_c, proj_vec, _ = self.projector.project(best_continuous, candidates)
            m, s = model.predict_with_uncertainty(proj_vec)
            return proj_c, best_continuous, m, s

        # Fallback to EXACT
        return self.select_query(task, candidates, strategy=SamplingStrategy.EXACT, beta=beta)

    def query(
        self,
        task: Task,
        candidates: List[Coalition],
        strategy: SamplingStrategy = SamplingStrategy.EXACT,
        beta: float = 2.0,
    ) -> QueryObservation:
        """Perform one active learning cycle: select query, observe oracle label, update GP."""
        # 1. Select query
        chosen_c, req_traits, pred_m, pred_s = self.select_query(task, candidates, strategy=strategy, beta=beta)
        actual_traits = chosen_c.to_vector()

        # Compute projection distance
        dim = min(len(req_traits), len(actual_traits))
        proj_dist = float(np.linalg.norm(req_traits[:dim] - actual_traits[:dim]))

        # 2. Observe simulated performance from warehouse oracle
        obs_efficacy = self.oracle.predict(chosen_c, task)

        # 3. Update task's Gaussian Process model
        model = self.models[task.id]
        model.add_observation(actual_traits, obs_efficacy)

        # 4. Record trial
        record = QueryObservation(
            iteration=len(self.history) + 1,
            task_id=task.id,
            coalition=chosen_c,
            requested_traits=req_traits,
            actual_traits=actual_traits,
            observed_efficacy=obs_efficacy,
            projection_distance=proj_dist,
            predicted_mean=pred_m,
            predicted_std=pred_s,
        )
        self.history.append(record)
        return record

    def compute_rmse(self, task: Task, candidates: List[Coalition]) -> float:
        """Compute the Root Mean Squared Error between current GP predictions and ground truth oracle."""
        model = self.models[task.id]
        errors = []
        for c in candidates:
            y_pred, _ = model.predict_with_uncertainty(c.to_vector())
            # Ground truth without noise for clean benchmark RMSE
            y_true = self.oracle.predict_vector(c.to_vector(), task)
            errors.append((y_pred - y_true) ** 2)
        return float(np.sqrt(np.mean(errors))) if errors else 0.0

    def run_benchmark_campaign(
        self,
        task: Task,
        candidates: List[Coalition],
        num_queries: int = 15,
        strategy: SamplingStrategy = SamplingStrategy.EXACT,
    ) -> List[float]:
        """Run an active learning campaign for a task, returning RMSE after each query."""
        rmse_history: List[float] = [self.compute_rmse(task, candidates)]

        for _ in range(num_queries):
            self.query(task, candidates, strategy=strategy)
            rmse = self.compute_rmse(task, candidates)
            rmse_history.append(rmse)

        return rmse_history
