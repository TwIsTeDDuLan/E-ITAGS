"""Timeline Scheduler for Multi-Robot Task Allocation and Interleaved Graph Search."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import copy
from typing import Dict, List, Optional, Tuple, Any, Set

from ..models.robot import Robot
from ..models.task import Task
from ..models.coalition import Coalition
from .motion_planner import MotionPlanner


@dataclass(frozen=True)
class RobotTimelineEntry:
    """An interval of activity on an individual robot's timeline."""
    task_id: str
    task_name: str
    origin_zone: str
    destination_zone: str
    travel_start: float          # Time when robot departs origin zone
    travel_end: float            # Time when robot arrives at task zone
    execution_start: float       # Time when task execution begins (synchronized)
    execution_end: float         # Time when task execution finishes

    @property
    def total_duration(self) -> float:
        return self.execution_end - self.travel_start

    @property
    def travel_duration(self) -> float:
        return self.travel_end - self.travel_start

    @property
    def execution_duration(self) -> float:
        return self.execution_end - self.execution_start


@dataclass(frozen=True)
class ScheduleAssignment:
    """A scheduled task execution in the master schedule."""
    task: Task
    coalition: Coalition
    execution_start: float
    execution_end: float
    robot_travel_times: Dict[str, float]       # {robot_id: seconds}
    robot_departure_times: Dict[str, float]    # {robot_id: timestamp}
    robot_origins: Dict[str, str]              # {robot_id: zone_name}

    def __repr__(self) -> str:
        return (
            f"ScheduleAssignment(Task={self.task.id} '{self.task.name}', "
            f"Coalition={self.coalition}, "
            f"Window=[{self.execution_start:.1f}s -> {self.execution_end:.1f}s], "
            f"Zone='{self.task.zone}')"
        )


@dataclass(frozen=True)
class ScheduleResult:
    """Outcome of attempting to add an assignment to a schedule."""
    success: bool
    makespan: float
    assignment: Optional[ScheduleAssignment] = None
    reason: str = ""

    def __repr__(self) -> str:
        if self.success:
            return f"ScheduleResult(SUCCESS, makespan={self.makespan:.1f}s)"
        return f"ScheduleResult(FAILED, reason='{self.reason}')"


class Schedule:
    """Multi-robot timeline manager tracking robot availability, locations, and tasks.
    
    In E-ITAGS, the scheduler is interleaved directly into node expansion:
    Whenever the search attempts to assign coalition C to task T, it queries
    the schedule to determine:
    1. Earliest time all coalition members can rendezvous at the task zone.
    2. Whether deadline and temporal constraints are met.
    3. The resulting makespan and resource utilization.
    """

    def __init__(
        self,
        robots: List[Robot],
        time_budget: float = 3600.0,
        enforce_strict_budget: bool = False,
    ):
        self.robots: Dict[str, Robot] = {r.id: r for r in robots}
        self.time_budget = time_budget
        self.enforce_strict_budget = enforce_strict_budget

        # Robot dynamic states
        self.robot_current_zone: Dict[str, str] = {r.id: r.initial_zone for r in robots}
        self.robot_available_time: Dict[str, float] = {r.id: 0.0 for r in robots}
        self.robot_timelines: Dict[str, List[RobotTimelineEntry]] = {r.id: [] for r in robots}

        # Global assignments
        self.assignments: List[ScheduleAssignment] = []
        self.assigned_task_ids: Set[str] = set()

    def get_robot_availability(self, robot_id: str) -> float:
        """Return the earliest time this robot is free."""
        return self.robot_available_time.get(robot_id, 0.0)

    def get_robot_current_zone(self, robot_id: str) -> str:
        """Return the current zone where this robot is stationed."""
        return self.robot_current_zone.get(robot_id, "staging")

    def get_makespan(self) -> float:
        """Return the maximum completion time across all robots."""
        if not self.robot_available_time:
            return 0.0
        return max(self.robot_available_time.values())

    def check_time_budget(self, budget: Optional[float] = None) -> bool:
        """Check if current makespan is within the given or configured time budget."""
        limit = budget if budget is not None else self.time_budget
        return self.get_makespan() <= limit

    def get_time_budget_overrun(self, budget: Optional[float] = None) -> float:
        """Return overrun duration in seconds (0.0 if within budget)."""
        limit = budget if budget is not None else self.time_budget
        return max(0.0, self.get_makespan() - limit)

    def can_assign(
        self,
        coalition: Coalition,
        task: Task,
        motion_planner: MotionPlanner,
    ) -> Tuple[bool, str, float, float]:
        """Dry-run check whether coalition can be scheduled for task without mutating state.

        Returns:
            (is_feasible, reason, computed_start_time, computed_end_time)
        """
        if task.id in self.assigned_task_ids:
            return False, f"Task {task.id} is already scheduled", 0.0, 0.0

        if not coalition.members:
            return False, "Coalition cannot be empty", 0.0, 0.0

        # Check all members exist in schedule
        for r in coalition.members:
            if r.id not in self.robots:
                return False, f"Robot {r.id} not in schedule fleet", 0.0, 0.0

        # Compute arrival times at task zone for each member
        arrival_times: List[float] = []
        travel_times: Dict[str, float] = {}
        for r in coalition.members:
            curr_zone = self.robot_current_zone[r.id]
            t_travel = motion_planner.compute_travel_time(curr_zone, task.zone, r)
            if math.isinf(t_travel):
                return False, f"Robot {r.name} cannot navigate to zone '{task.zone}'", 0.0, 0.0

            avail_time = self.robot_available_time[r.id]
            arrival_time = avail_time + t_travel
            arrival_times.append(arrival_time)
            travel_times[r.id] = t_travel

        # Execution begins once ALL coalition members have arrived and release time has passed
        synchronized_arrival = max(arrival_times)
        exec_start = max(synchronized_arrival, task.release_time_seconds)
        exec_end = exec_start + task.duration_seconds

        # Check task deadline
        if task.deadline_seconds is not None and exec_end > task.deadline_seconds:
            return (
                False,
                f"Exceeds task deadline {task.deadline_seconds:.1f}s (finishes at {exec_end:.1f}s)",
                exec_start,
                exec_end,
            )

        # Check strict time budget if configured
        if self.enforce_strict_budget and exec_end > self.time_budget:
            return (
                False,
                f"Exceeds mission time budget {self.time_budget:.1f}s (finishes at {exec_end:.1f}s)",
                exec_start,
                exec_end,
            )

        return True, "", exec_start, exec_end

    def add_assignment(
        self,
        coalition: Coalition,
        task: Task,
        motion_planner: MotionPlanner,
    ) -> ScheduleResult:
        """Schedule a coalition to execute a task, updating timelines and positions.

        Args:
            coalition: Team of robots executing the task.
            task: The task to perform.
            motion_planner: A* motion planner for travel times.

        Returns:
            ScheduleResult with success flag, new makespan, and assignment details.
        """
        feasible, reason, exec_start, exec_end = self.can_assign(coalition, task, motion_planner)
        if not feasible:
            return ScheduleResult(success=False, makespan=self.get_makespan(), reason=reason)

        # Record assignment details
        robot_travel_times: Dict[str, float] = {}
        robot_departure_times: Dict[str, float] = {}
        robot_origins: Dict[str, str] = {}

        for r in coalition.members:
            curr_zone = self.robot_current_zone[r.id]
            t_travel = motion_planner.compute_travel_time(curr_zone, task.zone, r)
            # Just-in-time departure: robot departs so it arrives precisely at exec_start
            t_depart = exec_start - t_travel

            robot_origins[r.id] = curr_zone
            robot_travel_times[r.id] = t_travel
            robot_departure_times[r.id] = t_depart

            # Add to robot timeline
            entry = RobotTimelineEntry(
                task_id=task.id,
                task_name=task.name,
                origin_zone=curr_zone,
                destination_zone=task.zone,
                travel_start=t_depart,
                travel_end=exec_start,
                execution_start=exec_start,
                execution_end=exec_end,
            )
            self.robot_timelines[r.id].append(entry)

            # Update robot state
            self.robot_current_zone[r.id] = task.zone
            self.robot_available_time[r.id] = exec_end

        # Create master assignment record
        assignment = ScheduleAssignment(
            task=task,
            coalition=coalition,
            execution_start=exec_start,
            execution_end=exec_end,
            robot_travel_times=robot_travel_times,
            robot_departure_times=robot_departure_times,
            robot_origins=robot_origins,
        )
        self.assignments.append(assignment)
        self.assigned_task_ids.add(task.id)

        return ScheduleResult(
            success=True,
            makespan=self.get_makespan(),
            assignment=assignment,
        )

    def get_robot_utilization(self, makespan: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """Compute active, travel, and idle breakdown for each robot in percent."""
        total_span = makespan if makespan is not None else self.get_makespan()
        if total_span <= 0.0:
            return {
                r_id: {"travel_pct": 0.0, "exec_pct": 0.0, "idle_pct": 100.0}
                for r_id in self.robots
            }

        stats: Dict[str, Dict[str, float]] = {}
        for r_id, entries in self.robot_timelines.items():
            total_travel = sum(e.travel_duration for e in entries)
            total_exec = sum(e.execution_duration for e in entries)
            total_active = total_travel + total_exec
            idle = max(0.0, total_span - total_active)

            stats[r_id] = {
                "travel_pct": round((total_travel / total_span) * 100.0, 1),
                "exec_pct": round((total_exec / total_span) * 100.0, 1),
                "idle_pct": round((idle / total_span) * 100.0, 1),
                "total_travel_sec": round(total_travel, 1),
                "total_exec_sec": round(total_exec, 1),
            }
        return stats

    def copy(self) -> Schedule:
        """Create a deep copy of the schedule for graph search branching."""
        new_sched = Schedule(
            robots=list(self.robots.values()),
            time_budget=self.time_budget,
            enforce_strict_budget=self.enforce_strict_budget,
        )
        new_sched.robot_current_zone = dict(self.robot_current_zone)
        new_sched.robot_available_time = dict(self.robot_available_time)
        new_sched.robot_timelines = {
            r_id: list(entries) for r_id, entries in self.robot_timelines.items()
        }
        new_sched.assignments = list(self.assignments)
        new_sched.assigned_task_ids = set(self.assigned_task_ids)
        return new_sched

    def get_summary(self) -> Dict[str, Any]:
        """Return high-level schedule summary dictionary."""
        return {
            "num_assignments": len(self.assignments),
            "makespan_seconds": self.get_makespan(),
            "time_budget_seconds": self.time_budget,
            "within_budget": self.check_time_budget(),
            "overrun_seconds": self.get_time_budget_overrun(),
            "robot_final_zones": dict(self.robot_current_zone),
        }
