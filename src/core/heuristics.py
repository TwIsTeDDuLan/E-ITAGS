"""Dual-Objective Heuristics Engine for E-ITAGS: NAC (Efficacy) and TBO (Makespan)."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

from ..models.task import Task
from ..models.coalition import Coalition
from .trait_engine import TraitEngine


class HeuristicEngine:
    """Computes Normalized Allocation Cost (NAC) and Time Budget Overrun (TBO).
    
    Mathematical Foundation:
    1. NAC(node) = 1.0 - (Sum_{assigned} S(C_j, t_j) + Sum_{unassigned} S_max(t_k)) / |Tasks|
       - Lower NAC indicates superior trait matching / higher mission efficacy.
       - Admissible: S_max upper-bounds any future assignment, guaranteeing A* admissibility.
    2. TBO(node) = max(0.0, makespan - time_budget) / time_budget
       - Lower TBO indicates adherence to warehouse operational time constraints.
    3. Combined Heuristic:
       h(node, w) = w * NAC(node) + (1.0 - w) * TBO(node)
       where w in [0.0, 1.0] balances quality against mission makespan.
    """

    @staticmethod
    def compute_optimistic_task_upper_bounds(
        tasks: List[Task],
        coalitions: List[Coalition],
        efficacy_fn: Optional[Callable[[Coalition, Task], float]] = None,
    ) -> Dict[str, float]:
        """Precompute the maximum achievable satisfaction or efficacy score for each task.

        Only considers coalitions that can physically enter the task's zone.
        This provides a tight, admissible upper bound for unassigned tasks.

        Args:
            tasks: List of warehouse tasks.
            coalitions: Set of physically realizable coalitions.
            efficacy_fn: Optional continuous efficacy function; defaults to TraitEngine satisfaction.

        Returns:
            Dict mapping task_id -> max_achievable_score in [0.0, 1.0].
        """
        upper_bounds: Dict[str, float] = {}

        for task in tasks:
            accessible = TraitEngine.filter_zone_accessible_coalitions(coalitions, task.zone)
            if not accessible:
                upper_bounds[task.id] = 0.0
                continue

            scores = []
            for c in accessible:
                if efficacy_fn is not None:
                    score = efficacy_fn(c, task)
                else:
                    score, _ = TraitEngine.compute_trait_satisfaction(c, task)
                scores.append(score)

            upper_bounds[task.id] = max(scores) if scores else 0.0

        return upper_bounds

    @staticmethod
    def compute_nac(
        assignments: Dict[str, Coalition],
        all_tasks: List[Task],
        task_upper_bounds: Dict[str, float],
        efficacy_fn: Optional[Callable[[Coalition, Task], float]] = None,
    ) -> float:
        """Compute the Normalized Allocation Cost (NAC) for a search node.

        Args:
            assignments: Mapping of {task_id: assigned_coalition} for tasks assigned so far.
            all_tasks: The complete set of tasks in the scenario.
            task_upper_bounds: Precomputed maximum possible score for each task.
            efficacy_fn: Optional continuous efficacy function.

        Returns:
            NAC in [0.0, 1.0], where 0.0 represents theoretical maximum efficacy.
        """
        total_tasks = len(all_tasks)
        if total_tasks == 0:
            return 0.0

        task_map = {t.id: t for t in all_tasks}
        accumulated_efficacy = 0.0

        for task_id, task in task_map.items():
            if task_id in assignments:
                coalition = assignments[task_id]
                if efficacy_fn is not None:
                    score = efficacy_fn(coalition, task)
                else:
                    score, _ = TraitEngine.compute_trait_satisfaction(coalition, task)
                accumulated_efficacy += score
            else:
                # Use optimistic upper bound for unassigned task
                accumulated_efficacy += task_upper_bounds.get(task_id, 1.0)

        mean_efficacy = accumulated_efficacy / total_tasks
        # Bound between [0.0, 1.0]
        nac = max(0.0, min(1.0, 1.0 - mean_efficacy))
        return float(nac)

    @staticmethod
    def compute_tbo(makespan: float, time_budget: float) -> float:
        """Compute Time Budget Overrun (TBO) penalty.

        Args:
            makespan: Current schedule makespan in seconds.
            time_budget: Allowable mission time budget in seconds.

        Returns:
            TBO ratio >= 0.0 (0.0 means within budget).
        """
        if time_budget <= 0.0:
            return 0.0
        overrun = max(0.0, makespan - time_budget)
        return float(overrun / time_budget)

    @staticmethod
    def compute_combined_heuristic(
        nac: float,
        tbo: float,
        weight: float = 0.5,
    ) -> float:
        """Compute the weighted dual-objective heuristic value h(u, w).

        Args:
            nac: Normalized Allocation Cost in [0.0, 1.0].
            tbo: Time Budget Overrun (>= 0.0).
            weight: Trade-off parameter w in [0.0, 1.0].
                    w = 1.0 -> purely optimizes trait matching quality.
                    w = 0.0 -> purely optimizes schedule completion time.

        Returns:
            Weighted heuristic value h >= 0.0.
        """
        w = max(0.0, min(1.0, weight))
        return float(w * nac + (1.0 - w) * tbo)

    @classmethod
    def evaluate(
        cls,
        assignments: Dict[str, Coalition],
        makespan: float,
        all_tasks: List[Task],
        task_upper_bounds: Dict[str, float],
        time_budget: float,
        weight: float = 0.5,
        efficacy_fn: Optional[Callable[[Coalition, Task], float]] = None,
    ) -> Tuple[float, float, float]:
        """Convenience method returning (h_cost, nac, tbo) in a single call."""
        nac = cls.compute_nac(assignments, all_tasks, task_upper_bounds, efficacy_fn=efficacy_fn)
        tbo = cls.compute_tbo(makespan, time_budget)
        h_cost = cls.compute_combined_heuristic(nac, tbo, weight=weight)
        return h_cost, nac, tbo
