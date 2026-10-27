"""A* Motion Planner with physical zone access constraints and multi-robot travel time modeling."""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import math
from typing import List, Tuple, Dict, Optional, Set

from ..models.warehouse import WarehouseGrid, Location
from ..models.robot import Robot, RobotType, ZONE_ACCESS
from ..models.coalition import Coalition


@dataclass(frozen=True)
class PathResult:
    """Result of an A* path planning query."""
    path: List[Location]
    distance: float                  # Path distance in meters (1 cell = 1 meter)
    travel_time: float               # Duration in seconds (distance / speed)
    cells_explored: int              # Number of A* search states expanded
    success: bool
    robot_type: Optional[RobotType] = None
    start: Optional[Location] = None
    goal: Optional[Location] = None
    reason: str = ""

    def __repr__(self) -> str:
        status = "OK" if self.success else f"FAIL ({self.reason})"
        return (
            f"PathResult(status={status}, len={len(self.path)} steps, "
            f"dist={self.distance:.1f}m, time={self.travel_time:.1f}s, explored={self.cells_explored})"
        )


class MotionPlanner:
    """A* grid motion planner tailored for heterogeneous warehouse robots.
    
    Features:
    1. Respects physical warehouse obstacles (shelving, walls, racks).
    2. Enforces zone-level access constraints (e.g. forklifts cannot traverse narrow aisles).
    3. Handles 3D aerial flight for drones (unrestricted direct line-of-sight flight).
    4. Computes individual and coalition travel times (rendezvous / slowest-member pace).
    5. Caches path solutions for lightning-fast graph search in later phases.
    """

    def __init__(self, warehouse: WarehouseGrid, allow_diagonal: bool = True):
        self.warehouse = warehouse
        self.allow_diagonal = allow_diagonal
        # Cache key: (start_coord, goal_coord, robot_type) -> PathResult
        self._path_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int], Optional[RobotType]], PathResult] = {}

    def clear_cache(self) -> None:
        """Clear cached paths."""
        self._path_cache.clear()

    def plan_path(
        self,
        start: Location,
        goal: Location,
        robot: Optional[Robot] = None,
        robot_type: Optional[RobotType] = None,
        speed_mps: Optional[float] = None,
    ) -> PathResult:
        """Find the optimal collision-free path from start to goal.

        Args:
            start: Starting grid location.
            goal: Goal grid location.
            robot: Optional Robot instance (supplies robot_type and speed_mps).
            robot_type: Optional RobotType archetype if robot is not provided.
            speed_mps: Optional override transit speed in m/s.

        Returns:
            PathResult with waypoints, path length, travel time, and search stats.
        """
        # Determine robot type and speed
        if robot is not None:
            r_type = robot.robot_type
            nominal_speed = speed_mps if speed_mps is not None else robot.traits.speed_mps
        elif robot_type is not None:
            r_type = robot_type
            if speed_mps is not None:
                nominal_speed = speed_mps
            elif r_type == RobotType.DRONE:
                nominal_speed = 5.0
            else:
                nominal_speed = 2.0
        else:
            r_type = None
            nominal_speed = speed_mps if speed_mps is not None else 2.0

        # Trivially at goal
        if start == goal:
            return PathResult(
                path=[start],
                distance=0.0,
                travel_time=0.0,
                cells_explored=0,
                success=True,
                robot_type=r_type,
                start=start,
                goal=goal,
            )

        # Check cache
        cache_key = ((start.x, start.y), (goal.x, goal.y), r_type)
        if cache_key in self._path_cache:
            cached = self._path_cache[cache_key]
            # Adjust travel time if speed differs
            adj_time = cached.distance / nominal_speed if nominal_speed > 0 else 0.0
            return PathResult(
                path=cached.path,
                distance=cached.distance,
                travel_time=adj_time,
                cells_explored=cached.cells_explored,
                success=cached.success,
                robot_type=r_type,
                start=start,
                goal=goal,
                reason=cached.reason,
            )

        # 1. Drones fly overhead in 3D airspace (direct flight)
        if r_type == RobotType.DRONE:
            dist = start.euclidean_distance_to(goal)
            flight_time = dist / nominal_speed
            # Generate interpolated straight-line waypoints for visualization
            path = self._interpolate_line(start, goal)
            res = PathResult(
                path=path,
                distance=dist,
                travel_time=flight_time,
                cells_explored=1,
                success=True,
                robot_type=r_type,
                start=start,
                goal=goal,
            )
            self._path_cache[cache_key] = res
            return res

        # 2. Ground Robots: Validate start and goal accessibility
        start_zone = self.warehouse.get_zone_at(start)
        goal_zone = self.warehouse.get_zone_at(goal)

        if r_type is not None:
            allowed = ZONE_ACCESS.get(r_type, frozenset())
            if start_zone and start_zone not in allowed:
                return PathResult(
                    path=[], distance=float("inf"), travel_time=float("inf"),
                    cells_explored=0, success=False, robot_type=r_type, start=start, goal=goal,
                    reason=f"Start zone '{start_zone}' is physically inaccessible to {r_type.value}"
                )
            if goal_zone and goal_zone not in allowed:
                return PathResult(
                    path=[], distance=float("inf"), travel_time=float("inf"),
                    cells_explored=0, success=False, robot_type=r_type, start=start, goal=goal,
                    reason=f"Goal zone '{goal_zone}' is physically inaccessible to {r_type.value}"
                )

        if not self.warehouse.is_valid_coord(start.x, start.y):
            return PathResult(
                path=[], distance=float("inf"), travel_time=float("inf"),
                cells_explored=0, success=False, robot_type=r_type, start=start, goal=goal,
                reason=f"Start location {start} is inside an obstacle or out of bounds"
            )

        if not self.warehouse.is_valid_coord(goal.x, goal.y):
            return PathResult(
                path=[], distance=float("inf"), travel_time=float("inf"),
                cells_explored=0, success=False, robot_type=r_type, start=start, goal=goal,
                reason=f"Goal location {goal} is inside an obstacle or out of bounds"
            )

        # 3. A* Search on Grid
        res = self._run_astar(start, goal, r_type, nominal_speed)
        self._path_cache[cache_key] = res
        return res

    def _is_cell_traversable(self, x: int, y: int, robot_type: Optional[RobotType]) -> bool:
        """Check if a cell can be traversed by the given robot type."""
        # Must be in grid bounds and not an obstacle
        if not self.warehouse.is_valid_coord(x, y):
            return False

        # Zone access constraint check
        if robot_type is not None:
            zone_name = self.warehouse.get_zone_at((x, y))
            if zone_name is not None:
                allowed = ZONE_ACCESS.get(robot_type, frozenset())
                if zone_name not in allowed:
                    return False

        return True

    def _run_astar(
        self,
        start: Location,
        goal: Location,
        robot_type: Optional[RobotType],
        speed_mps: float,
    ) -> PathResult:
        """Execute A* search on the warehouse grid graph."""
        start_tuple = (start.x, start.y)
        goal_tuple = (goal.x, goal.y)

        # Priority Queue holds (f_score, tie_breaker, (x, y))
        counter = 0
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (start.euclidean_distance_to(goal), counter, start_tuple))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_tuple: 0.0}
        closed_set: Set[Tuple[int, int]] = set()

        # Motion directions: 4-connected + optional diagonals
        if self.allow_diagonal:
            # (dx, dy, step_cost)
            moves = [
                (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
                (1, -1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
            ]
        else:
            moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]

        cells_explored = 0

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)
            cells_explored += 1

            if current == goal_tuple:
                # Reconstruct path
                path_tuples = [current]
                while current in came_from:
                    current = came_from[current]
                    path_tuples.append(current)
                path_tuples.reverse()

                path = [Location(x, y) for x, y in path_tuples]
                dist = g_score[goal_tuple]
                time_sec = dist / speed_mps if speed_mps > 0 else 0.0

                return PathResult(
                    path=path,
                    distance=dist,
                    travel_time=time_sec,
                    cells_explored=cells_explored,
                    success=True,
                    robot_type=robot_type,
                    start=start,
                    goal=goal,
                )

            cx, cy = current
            current_g = g_score[current]

            for dx, dy, step_cost in moves:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)

                if neighbor in closed_set:
                    continue

                if not self._is_cell_traversable(nx, ny, robot_type):
                    continue

                # Diagonal corner-cutting check: both orthogonal cells must be clear
                if dx != 0 and dy != 0:
                    if not self._is_cell_traversable(cx + dx, cy, robot_type) or \
                       not self._is_cell_traversable(cx, cy + dy, robot_type):
                        continue

                tentative_g = current_g + step_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = Location(nx, ny).euclidean_distance_to(goal)
                    f_score = tentative_g + h_score
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        # Goal was not reached
        return PathResult(
            path=[],
            distance=float("inf"),
            travel_time=float("inf"),
            cells_explored=cells_explored,
            success=False,
            robot_type=robot_type,
            start=start,
            goal=goal,
            reason="No collision-free path found to goal",
        )

    def _interpolate_line(self, start: Location, goal: Location) -> List[Location]:
        """Interpolate direct line of locations between start and goal (for drones)."""
        dist = int(math.ceil(start.euclidean_distance_to(goal)))
        if dist == 0:
            return [start]
        pts: List[Location] = []
        for i in range(dist + 1):
            t = i / dist
            x = int(round(start.x + t * (goal.x - start.x)))
            y = int(round(start.y + t * (goal.y - start.y)))
            if not pts or pts[-1] != Location(x, y):
                pts.append(Location(x, y))
        return pts

    def compute_travel_time(self, from_zone: str, to_zone: str, robot: Robot) -> float:
        """Compute travel time in seconds between two warehouse zone centers for a specific robot."""
        if from_zone == to_zone:
            return 0.0

        start_loc = self.warehouse.get_zone_center(from_zone)
        goal_loc = self.warehouse.get_zone_center(to_zone)

        res = self.plan_path(start_loc, goal_loc, robot=robot)
        return res.travel_time

    def compute_coalition_travel_time(
        self,
        coalition: Coalition,
        to_zone: str,
        from_zone: Optional[str] = None,
        robot_locations: Optional[Dict[str, str]] = None,
    ) -> float:
        """Compute the total travel duration for a coalition to arrive at a task zone.

        In multi-robot synchronization, all robots travel to the designated task zone.
        The coalition rendezvous is complete when the last robot arrives:
            travel_time = max_{r in Coalition} ( travel_time(r.location -> to_zone) )

        Args:
            coalition: The team of robots.
            to_zone: Destination zone where the task is located.
            from_zone: Default departure zone for all members (e.g. 'staging').
            robot_locations: Optional per-robot current zone dict {robot_id: zone_name}.

        Returns:
            Travel duration in seconds until all members have arrived.
        """
        if not coalition.members:
            return 0.0

        travel_times: List[float] = []
        for robot in coalition.members:
            curr_zone = robot_locations.get(robot.id, from_zone) if robot_locations else from_zone
            if curr_zone is None:
                curr_zone = robot.initial_zone

            t = self.compute_travel_time(curr_zone, to_zone, robot)
            travel_times.append(t)

        return max(travel_times) if travel_times else 0.0

    def compute_travel_time_matrix(
        self, robots: List[Robot]
    ) -> Dict[str, Dict[Tuple[str, str], float]]:
        """Precompute the complete zone-to-zone travel time matrix for all robots.

        Returns:
            Dict mapping robot_id -> {(from_zone, to_zone): travel_time_seconds}
        """
        zones = list(self.warehouse.zones.keys())
        matrix: Dict[str, Dict[Tuple[str, str], float]] = {}

        for robot in robots:
            matrix[robot.id] = {}
            for z1 in zones:
                for z2 in zones:
                    if z1 == z2:
                        matrix[robot.id][(z1, z2)] = 0.0
                    else:
                        t = self.compute_travel_time(z1, z2, robot)
                        matrix[robot.id][(z1, z2)] = t

        return matrix
