"""Interleaved ITAGS Graph Search: Simultaneous Task Allocation and Timeline Scheduling."""
from __future__ import annotations
from dataclasses import dataclass, field
import heapq
import time
from typing import Dict, List, Optional, Tuple, Set, Callable, Any

from ..models.warehouse import WarehouseGrid
from ..models.robot import Robot
from ..models.task import Task
from ..models.coalition import Coalition
from .trait_engine import TraitEngine
from .motion_planner import MotionPlanner
from .scheduler import Schedule, ScheduleResult
from .heuristics import HeuristicEngine


@dataclass
class SearchNode:
    """A node in the ITAGS interleaved search graph."""
    node_id: int
    assignments: Dict[str, Coalition]      # {task_id: coalition}
    unassigned_task_ids: List[str]         # Remaining task IDs to schedule
    schedule: Schedule                     # Interleaved multi-robot timeline
    g_cost: float                          # Accumulated cost so far
    h_cost: float                          # Heuristic estimate to goal
    f_cost: float                          # Total priority f = g + h
    nac: float                             # Normalized Allocation Cost
    tbo: float                             # Time Budget Overrun
    depth: int                             # Number of tasks assigned
    parent: Optional[SearchNode] = None    # Predecessor node
    expanded_task_id: Optional[str] = None # Which task was assigned at this step
    expanded_coalition: Optional[Coalition] = None

    def __lt__(self, other: SearchNode) -> bool:
        """Tie-breaking for PriorityQueue:
        1. Lowest f_cost
        2. Deepest node first (dive toward goal on ties instead of BFS explosion)
        3. Lowest makespan
        4. Unique node_id
        """
        if abs(self.f_cost - other.f_cost) > 1e-6:
            return self.f_cost < other.f_cost
        if self.depth != other.depth:
            return self.depth > other.depth  # Prefer deeper nodes
        if abs(self.schedule.get_makespan() - other.schedule.get_makespan()) > 1e-3:
            return self.schedule.get_makespan() < other.schedule.get_makespan()
        return self.node_id < other.node_id

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SearchNode):
            return False
        return self.node_id == other.node_id

    @property
    def state_hash(self) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Canonical hash representing the partial assignment state."""
        return tuple(sorted((t_id, c.robot_ids) for t_id, c in self.assignments.items()))

    def __repr__(self) -> str:
        return (
            f"SearchNode(id={self.node_id}, depth={self.depth}, "
            f"f={self.f_cost:.3f}, g={self.g_cost:.3f}, h={self.h_cost:.3f}, "
            f"NAC={self.nac:.3f}, TBO={self.tbo:.3f}, "
            f"assigned={len(self.assignments)}, unassigned={len(self.unassigned_task_ids)})"
        )


@dataclass
class SearchStatistics:
    """Diagnostic metrics captured during graph search."""
    nodes_expanded: int = 0
    nodes_generated: int = 0
    nodes_pruned_zone: int = 0
    nodes_pruned_schedule: int = 0
    nodes_pruned_deadline: int = 0
    nodes_pruned_zero_sat: int = 0
    nodes_pruned_closed: int = 0
    search_duration_sec: float = 0.0
    solution_found: bool = False
    explored_nodes: List[SearchNode] = field(default_factory=list)


@dataclass
class SearchResult:
    """The result of an ITAGS graph search."""
    solution_found: bool
    goal_node: Optional[SearchNode]
    schedule: Optional[Schedule]
    assignments: Dict[str, Coalition]
    makespan: float
    total_efficacy: float
    mean_efficacy: float
    nac: float
    tbo: float
    weight: float
    stats: SearchStatistics
    solution_path: List[SearchNode] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.solution_found:
            return (
                f"SearchResult(SUCCESS, tasks={len(self.assignments)}, "
                f"makespan={self.makespan:.1f}s, mean_eff={self.mean_efficacy:.2f}, "
                f"NAC={self.nac:.3f}, TBO={self.tbo:.3f}, "
                f"expanded={self.stats.nodes_expanded}, time={self.stats.search_duration_sec:.3f}s)"
            )
        return (
            f"SearchResult(FAILED, expanded={self.stats.nodes_expanded}, "
            f"time={self.stats.search_duration_sec:.3f}s)"
        )


class ITAGSSearch:
    """The core Incremental Task Allocation Graph Search (ITAGS) algorithm.
    
    Interleaves trait satisfaction evaluation, physical A* motion planning,
    and timeline constraint scheduling into a unified search tree.
    """

    def __init__(
        self,
        robots: List[Robot],
        tasks: List[Task],
        warehouse: WarehouseGrid,
        motion_planner: MotionPlanner,
        time_budget: float = 3600.0,
        weight: float = 0.5,
        max_coalition_size: int = 3,
        max_expansions: int = 5000,
        enforce_strict_budget: bool = False,
    ):
        self.robots = robots
        self.tasks = tasks
        self.task_map: Dict[str, Task] = {t.id: t for t in tasks}
        self.warehouse = warehouse
        self.motion_planner = motion_planner
        self.time_budget = time_budget
        self.weight = weight
        self.max_coalition_size = max_coalition_size
        self.max_expansions = max_expansions
        self.enforce_strict_budget = enforce_strict_budget

        # 1. Enumerate all realizable coalitions
        self.coalitions = TraitEngine.enumerate_all_coalitions(robots, max_size=max_coalition_size)

        # 2. Precompute optimistic upper bounds for NAC admissibility
        self.task_upper_bounds = HeuristicEngine.compute_optimistic_task_upper_bounds(
            self.tasks, self.coalitions
        )

        # 3. Precompute candidate coalitions per task
        self.task_candidates: Dict[str, List[Coalition]] = {}
        for t in self.tasks:
            accessible = TraitEngine.filter_zone_accessible_coalitions(self.coalitions, t.zone)
            # Retain coalitions with positive trait satisfaction and order by:
            # 1. Highest satisfaction score (descending)
            # 2. Smallest team size (ascending - don't tie up extra robots unnecessarily)
            scored = []
            for c in accessible:
                score, _ = TraitEngine.compute_trait_satisfaction(c, t)
                if score > 0.0:
                    scored.append((score, c.size, c))
            scored.sort(key=lambda item: (-item[0], item[1]))
            self.task_candidates[t.id] = [item[2] for item in scored]

    def select_next_task(self, unassigned_task_ids: List[str]) -> Task:
        """Select the next task to assign using the Most Constrained Task First heuristic.
        
        Ranked by:
        1. Fewest candidate coalitions (most physically restricted).
        2. Earliest deadline (if specified).
        3. Longest task duration (longest operations scheduled first).
        """
        def task_difficulty_key(t_id: str) -> Tuple[int, float, float]:
            t = self.task_map[t_id]
            n_candidates = len(self.task_candidates.get(t_id, []))
            deadline = t.deadline_seconds if t.deadline_seconds is not None else float("inf")
            neg_duration = -t.duration_seconds
            return (n_candidates, deadline, neg_duration)

        best_task_id = min(unassigned_task_ids, key=task_difficulty_key)
        return self.task_map[best_task_id]

    def solve(
        self,
        weight: Optional[float] = None,
        efficacy_fn: Optional[Callable[[Coalition, Task], float]] = None,
        verbose: bool = False,
    ) -> SearchResult:
        """Execute the ITAGS A* graph search to find an optimal allocation and schedule."""
        start_time = time.perf_counter()
        w = weight if weight is not None else self.weight
        stats = SearchStatistics()

        # Initialize Root Node
        initial_schedule = Schedule(
            robots=self.robots,
            time_budget=self.time_budget,
            enforce_strict_budget=self.enforce_strict_budget,
        )

        root_h, root_nac, root_tbo = HeuristicEngine.evaluate(
            assignments={},
            makespan=0.0,
            all_tasks=self.tasks,
            task_upper_bounds=self.task_upper_bounds,
            time_budget=self.time_budget,
            weight=w,
            efficacy_fn=efficacy_fn,
        )

        root_node = SearchNode(
            node_id=0,
            assignments={},
            unassigned_task_ids=[t.id for t in self.tasks],
            schedule=initial_schedule,
            g_cost=0.0,
            h_cost=root_h,
            f_cost=root_h,
            nac=root_nac,
            tbo=root_tbo,
            depth=0,
            parent=None,
        )

        # OPEN Priority Queue and CLOSED set
        open_set: List[SearchNode] = []
        heapq.heappush(open_set, root_node)
        closed_set: Set[Tuple[Tuple[str, Tuple[str, ...]], ...]] = set()

        next_node_id = 1
        goal_node: Optional[SearchNode] = None

        if verbose:
            print(f"🚀 Starting ITAGS Search: Tasks={len(self.tasks)}, Fleet={len(self.robots)}, Weight(w)={w:.2f}")

        while open_set and stats.nodes_expanded < self.max_expansions:
            current = heapq.heappop(open_set)

            # Check if this state has already been evaluated with better cost
            if current.state_hash in closed_set:
                stats.nodes_pruned_closed += 1
                continue

            closed_set.add(current.state_hash)
            stats.nodes_expanded += 1
            stats.explored_nodes.append(current)

            if verbose and stats.nodes_expanded % 25 == 0:
                print(
                    f"   [Exp {stats.nodes_expanded:>4}] Depth={current.depth}/{len(self.tasks)} | "
                    f"OPEN={len(open_set):>3} | f={current.f_cost:.3f} (NAC={current.nac:.3f}, TBO={current.tbo:.3f})"
                )

            # GOAL TEST: All tasks assigned
            if not current.unassigned_task_ids:
                goal_node = current
                stats.solution_found = True
                if verbose:
                    print(f"✅ Goal Node Found at expansion {stats.nodes_expanded}!")
                break

            # 1. Pick next task using Most Constrained First heuristic
            next_task = self.select_next_task(current.unassigned_task_ids)

            # 2. Get candidate coalitions
            candidates = self.task_candidates.get(next_task.id, [])

            # Branch across candidate coalitions
            for coalition in candidates:
                stats.nodes_generated += 1

                # 3. Interleaved Schedule Check: Physical motion + Timeline
                feasible, reason, exec_start, exec_end = current.schedule.can_assign(
                    coalition, next_task, self.motion_planner
                )

                if not feasible:
                    if "deadline" in reason.lower():
                        stats.nodes_pruned_deadline += 1
                    else:
                        stats.nodes_pruned_schedule += 1
                    continue

                # 4. Clone schedule and apply assignment
                child_schedule = current.schedule.copy()
                add_res = child_schedule.add_assignment(coalition, next_task, self.motion_planner)
                if not add_res.success:
                    stats.nodes_pruned_schedule += 1
                    continue

                # 5. Compute Costs & Heuristics
                child_assignments = {**current.assignments, next_task.id: coalition}
                child_makespan = child_schedule.get_makespan()

                child_h, child_nac, child_tbo = HeuristicEngine.evaluate(
                    assignments=child_assignments,
                    makespan=child_makespan,
                    all_tasks=self.tasks,
                    task_upper_bounds=self.task_upper_bounds,
                    time_budget=self.time_budget,
                    weight=w,
                    efficacy_fn=efficacy_fn,
                )

                # g-cost accumulates assignment cost: 1 - satisfaction
                sat_score, _ = TraitEngine.compute_trait_satisfaction(coalition, next_task)
                step_cost = (1.0 - sat_score) / len(self.tasks)
                child_g = current.g_cost + step_cost
                child_f = child_g + child_h

                child_unassigned = [t_id for t_id in current.unassigned_task_ids if t_id != next_task.id]

                child_node = SearchNode(
                    node_id=next_node_id,
                    assignments=child_assignments,
                    unassigned_task_ids=child_unassigned,
                    schedule=child_schedule,
                    g_cost=child_g,
                    h_cost=child_h,
                    f_cost=child_f,
                    nac=child_nac,
                    tbo=child_tbo,
                    depth=current.depth + 1,
                    parent=current,
                    expanded_task_id=next_task.id,
                    expanded_coalition=coalition,
                )
                next_node_id += 1

                if child_node.state_hash not in closed_set:
                    heapq.heappush(open_set, child_node)

        stats.search_duration_sec = time.perf_counter() - start_time

        # Reconstruct path if goal found
        solution_path: List[SearchNode] = []
        if goal_node:
            curr = goal_node
            while curr is not None:
                solution_path.append(curr)
                curr = curr.parent
            solution_path.reverse()

            # Compute efficacy metrics
            scores = [
                TraitEngine.compute_trait_satisfaction(goal_node.assignments[t.id], t)[0]
                for t in self.tasks
            ]
            tot_eff = sum(scores)
            mean_eff = tot_eff / len(self.tasks) if self.tasks else 0.0

            return SearchResult(
                solution_found=True,
                goal_node=goal_node,
                schedule=goal_node.schedule,
                assignments=goal_node.assignments,
                makespan=goal_node.schedule.get_makespan(),
                total_efficacy=tot_eff,
                mean_efficacy=mean_eff,
                nac=goal_node.nac,
                tbo=goal_node.tbo,
                weight=w,
                stats=stats,
                solution_path=solution_path,
            )

        return SearchResult(
            solution_found=False,
            goal_node=None,
            schedule=None,
            assignments={},
            makespan=float("inf"),
            total_efficacy=0.0,
            mean_efficacy=0.0,
            nac=1.0,
            tbo=float("inf"),
            weight=w,
            stats=stats,
        )

    def sweep_pareto_front(
        self,
        weights: Optional[List[float]] = None,
        verbose: bool = False,
    ) -> List[SearchResult]:
        """Run search across a range of w weights to discover the Pareto efficacy-makespan front."""
        if weights is None:
            weights = [0.0, 0.25, 0.5, 0.75, 1.0]

        results: List[SearchResult] = []
        for w in weights:
            if verbose:
                print(f"\n--- Sweeping Weight w = {w:.2f} ---")
            res = self.solve(weight=w, verbose=verbose)
            results.append(res)

        return results
