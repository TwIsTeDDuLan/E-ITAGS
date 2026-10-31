"""Warehouse spatial representation and zones."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Union, Any, Set
import numpy as np


@dataclass(frozen=True)
class Location:
    x: int
    y: int

    def manhattan_distance_to(self, other: Location) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def euclidean_distance_to(self, other: Location) -> float:
        return float(np.hypot(self.x - other.x, self.y - other.y))


@dataclass
class WarehouseZone:
    name: str
    description: str
    center: Location
    cells: List[Location] = field(default_factory=list)
    narrow_access_only: bool = False


class WarehouseGrid:
    """A 2D warehouse floor map with zones, obstacles, and navigation metrics."""

    def __init__(self, width: int = 30, height: int = 20):
        self.width = width
        self.height = height
        self.obstacles: set[Tuple[int, int]] = set()
        self.zones: Dict[str, WarehouseZone] = {}
        self._setup_default_zones()

    def _setup_default_zones(self) -> None:
        """Create standard realistic warehouse sorting zones."""
        # Receiving Zone (Top-Left): Trucks unload raw pallets
        self.zones["receiving"] = WarehouseZone(
            name="receiving",
            description="Truck unloading dock for inbound pallets",
            center=Location(4, 15),
            cells=[Location(x, y) for x in range(1, 8) for y in range(12, 18)],
        )

        # Sorting Zone (Top-Right): Pallet identification & sorting
        self.zones["sorting"] = WarehouseZone(
            name="sorting",
            description="Central automated visual sorting area",
            center=Location(22, 15),
            cells=[Location(x, y) for x in range(18, 27) for y in range(12, 18)],
        )

        # Staging Zone (Mid-Left): Temporary buffer for pallets awaiting transport
        self.zones["staging"] = WarehouseZone(
            name="staging",
            description="Buffer staging floor for sorted pallets",
            center=Location(5, 7),
            cells=[Location(x, y) for x in range(1, 9) for y in range(5, 10)],
        )

        # Narrow Aisle Stacks (Bottom-Left): Floor stacks accessible only by low-profile IW Hub
        self.zones["narrow_aisle"] = WarehouseZone(
            name="narrow_aisle",
            description="High-density tight floor stacks (IW Hub exclusive access)",
            center=Location(5, 2),
            cells=[Location(x, y) for x in range(1, 9) for y in range(1, 4)],
            narrow_access_only=True,
        )

        # Shipping Zone (Bottom-Right): Outbound loading bays
        self.zones["shipping"] = WarehouseZone(
            name="shipping",
            description="Outbound outbound dispatch and truck bays",
            center=Location(22, 5),
            cells=[Location(x, y) for x in range(18, 27) for y in range(2, 9)],
        )

        self._build_cell_lookup()

    def _build_cell_lookup(self) -> None:
        """Cache (x, y) -> zone_name mapping for fast spatial queries."""
        self._cell_to_zone: Dict[Tuple[int, int], str] = {}
        for z_name, zone in self.zones.items():
            for cell in zone.cells:
                self._cell_to_zone[(cell.x, cell.y)] = z_name

    def get_zone_at(self, loc: Location | Tuple[int, int]) -> Optional[str]:
        """Return the zone name containing the given location, or None if in transit corridor."""
        if isinstance(loc, Location):
            coord = (loc.x, loc.y)
        else:
            coord = loc
        return self._cell_to_zone.get(coord)

    def add_obstacle(self, x: int, y: int) -> None:
        """Add an impassable cell obstacle."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.obstacles.add((x, y))

    def add_obstacles(self, coords: list[Tuple[int, int]]) -> None:
        """Add multiple obstacle coordinates."""
        for x, y in coords:
            self.add_obstacle(x, y)

    def add_default_warehouse_racks(self) -> None:
        """Add standard central warehouse storage racks that robots navigate around.
        
        Creates two central rack blocks with wide transit aisles:
        - Upper rack: x in [12, 15], y in [12, 17]
        - Lower rack: x in [12, 15], y in [3, 8]
        Leaves open aisles along y=0..2, y=9..11, y=18..19 and central corridor at x=10..11, 16..17.
        """
        for x in range(12, 16):
            for y in range(12, 18):
                self.add_obstacle(x, y)
            for y in range(3, 9):
                self.add_obstacle(x, y)

    def get_zone_center(self, zone_name: str) -> Location:
        if zone_name not in self.zones:
            raise KeyError(f"Unknown warehouse zone: '{zone_name}'. Available: {list(self.zones.keys())}")
        return self.zones[zone_name].center

    def is_valid_coord(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles

    def get_distance(self, from_zone: str, to_zone: str) -> float:
        """Nominal distance in meters between zone centers."""
        loc1 = self.get_zone_center(from_zone)
        loc2 = self.get_zone_center(to_zone)
        # Using 1 cell = 1 meter
        return loc1.euclidean_distance_to(loc2)

    def render_grid(
        self,
        robot_positions: Optional[Union[Dict[str, Any], List[Any]]] = None,
        task_positions: Optional[Union[Dict[str, Any], List[Any]]] = None,
        path: Optional[List[Any]] = None,
        show_legend: bool = True,
        show_coordinates: bool = True,
        use_color: bool = True,
    ) -> str:
        """Render the 2D floor grid as a formatted ASCII/Unicode string."""
        return render_warehouse_grid(
            warehouse=self,
            robot_positions=robot_positions,
            task_positions=task_positions,
            path=path,
            show_legend=show_legend,
            show_coordinates=show_coordinates,
            use_color=use_color,
        )

    def print_grid(
        self,
        robot_positions: Optional[Union[Dict[str, Any], List[Any]]] = None,
        task_positions: Optional[Union[Dict[str, Any], List[Any]]] = None,
        path: Optional[List[Any]] = None,
        show_legend: bool = True,
        show_coordinates: bool = True,
        use_color: bool = True,
    ) -> str:
        """Print the 2D floor grid to stdout and return the rendered string."""
        return print_warehouse_grid(
            warehouse=self,
            robot_positions=robot_positions,
            task_positions=task_positions,
            path=path,
            show_legend=show_legend,
            show_coordinates=show_coordinates,
            use_color=use_color,
        )


def render_warehouse_grid(
    warehouse: WarehouseGrid,
    robot_positions: Optional[Union[Dict[str, Any], List[Any]]] = None,
    task_positions: Optional[Union[Dict[str, Any], List[Any]]] = None,
    path: Optional[List[Any]] = None,
    show_legend: bool = True,
    show_coordinates: bool = True,
    use_color: bool = True,
) -> str:
    """Render a 2D ASCII/Unicode representation of the warehouse floor map.

    Args:
        warehouse: WarehouseGrid instance to visualize.
        robot_positions: Optional mapping {robot_id: loc} or list of Robot models.
        task_positions: Optional mapping {task_id: loc} or list of Task models.
        path: Optional sequence of (x, y) coordinates or Location objects to overlay.
        show_legend: Whether to include the zone, obstacle, and overlay legend.
        show_coordinates: Whether to display coordinate axes on the perimeter.
        use_color: Whether to use ANSI terminal colors (recommended for rich display).

    Returns:
        Formatted multi-line string representing the warehouse floor map.
    """
    RESET = "\033[0m" if use_color else ""
    BOLD = "\033[1m" if use_color else ""
    DIM = "\033[2m" if use_color else ""

    ZONE_INFO = {
        "receiving": ("\033[96m" if use_color else "", "R", "Receiving Dock"),
        "sorting": ("\033[93m" if use_color else "", "S", "Automated Sorting"),
        "staging": ("\033[95m" if use_color else "", "T", "Staging Floor"),
        "narrow_aisle": ("\033[91m" if use_color else "", "N", "Narrow Aisle Stacks"),
        "shipping": ("\033[92m" if use_color else "", "H", "Shipping Bays"),
    }

    # Normalize robot positions: dict of (x, y) -> (2-char display label, full name)
    norm_robots: Dict[Tuple[int, int], Tuple[str, str]] = {}
    if robot_positions:
        if isinstance(robot_positions, dict):
            for k, v in robot_positions.items():
                coord = (v.x, v.y) if hasattr(v, "x") else tuple(v)
                label = str(k)[:2]
                norm_robots[coord] = (label, str(k))
        elif isinstance(robot_positions, list):
            for r in robot_positions:
                label = getattr(r, "id", str(r))
                if hasattr(r, "initial_zone") and r.initial_zone in warehouse.zones:
                    loc = warehouse.zones[r.initial_zone].center
                    coord = (loc.x, loc.y)
                elif hasattr(r, "location"):
                    loc = r.location
                    coord = (loc.x, loc.y) if hasattr(loc, "x") else tuple(loc)
                else:
                    continue
                norm_robots[coord] = (label[:2], label)

    # Normalize task positions: dict of (x, y) -> (2-char display label, full name)
    norm_tasks: Dict[Tuple[int, int], Tuple[str, str]] = {}
    if task_positions:
        if isinstance(task_positions, dict):
            for k, v in task_positions.items():
                coord = (v.x, v.y) if hasattr(v, "x") else tuple(v)
                label = str(k)[:2]
                norm_tasks[coord] = (label, str(k))
        elif isinstance(task_positions, list):
            for t in task_positions:
                label = getattr(t, "id", str(t))
                if hasattr(t, "zone") and t.zone in warehouse.zones:
                    loc = warehouse.zones[t.zone].center
                    coord = (loc.x, loc.y)
                elif hasattr(t, "location"):
                    loc = t.location
                    coord = (loc.x, loc.y) if hasattr(loc, "x") else tuple(loc)
                else:
                    continue
                norm_tasks[coord] = (label[:2], label)

    # Normalize path waypoints
    norm_path: set[Tuple[int, int]] = set()
    if path:
        for p in path:
            coord = (p.x, p.y) if hasattr(p, "x") else tuple(p)
            norm_path.add(coord)

    center_lookup = { (z.center.x, z.center.y): (name, z) for name, z in warehouse.zones.items() }

    lines: List[str] = []
    lines.append(f"{BOLD}================================================================================{RESET}")
    lines.append(f"{BOLD}  WAREHOUSE GRID FLOOR MAP ({warehouse.width}m × {warehouse.height}m = {warehouse.width * warehouse.height}m²){RESET}")
    lines.append(f"{BOLD}================================================================================{RESET}")

    if show_coordinates:
        tens_header = "       " + "".join(f"{x // 10} " for x in range(warehouse.width))
        units_header = "       " + "".join(f"{x % 10} " for x in range(warehouse.width))
        lines.append(f"{DIM}{tens_header}{RESET}")
        lines.append(f"{DIM}{units_header}{RESET}")

    lines.append("     ┌" + "──" * warehouse.width + "┐")

    for y in range(warehouse.height - 1, -1, -1):
        row_str = f"  {y:>2} │" if show_coordinates else "     │"
        for x in range(warehouse.width):
            coord = (x, y)
            if coord in norm_robots:
                lbl, _ = norm_robots[coord]
                disp = f"{lbl:<2}"[:2]
                cell = f"\033[1;97;44m{disp}{RESET}" if use_color else disp
            elif coord in norm_tasks:
                lbl, _ = norm_tasks[coord]
                disp = f"{lbl:<2}"[:2]
                cell = f"\033[1;93;41m{disp}{RESET}" if use_color else disp
            elif coord in norm_path:
                cell = f"\033[1;92m··{RESET}" if use_color else "··"
            elif coord in warehouse.obstacles:
                cell = f"\033[90m██{RESET}" if use_color else "██"
            elif coord in center_lookup:
                z_name, z = center_lookup[coord]
                color, code, _ = ZONE_INFO.get(z_name, ("", z_name[0].upper(), z_name))
                cell = f"{BOLD}{color}{code}*{RESET}"
            elif coord in warehouse._cell_to_zone:
                z_name = warehouse._cell_to_zone[coord]
                color, code, _ = ZONE_INFO.get(z_name, ("", z_name[0].upper(), z_name))
                cell = f"{color}{code} {RESET}"
            else:
                cell = f"{DIM}· {RESET}" if use_color else "· "
            row_str += cell
        row_str += "│"
        lines.append(row_str)

    lines.append("     └" + "──" * warehouse.width + "┘")

    if show_legend:
        lines.append(f"\n{BOLD}ZONE & FEATURE LEGEND:{RESET}")
        for z_name, (color, code, label) in ZONE_INFO.items():
            if z_name in warehouse.zones:
                z = warehouse.zones[z_name]
                restriction = f" {BOLD}[Restricted: IW Hub Only]{RESET}" if z.narrow_access_only else ""
                lines.append(f"  {color}[{code} ]{RESET} {label:<22} Center: {color}[{code}*]{RESET} at ({z.center.x:>2}, {z.center.y:>2})  Cells: {len(z.cells):>2}{restriction}")
        lines.append(f"  {BOLD}\033[90m[██]{RESET} Impassable Obstacles   Count: {len(warehouse.obstacles):>2} cells (Storage Racks)" if use_color else f"  [██] Impassable Obstacles   Count: {len(warehouse.obstacles):>2} cells (Storage Racks)")
        lines.append(f"  {DIM}[· ]{RESET} Open Transit Corridor  Robots navigate freely" if use_color else "  [· ] Open Transit Corridor  Robots navigate freely")
        if norm_robots:
            r_desc = ", ".join(f"{lbl} ({name}) at {c}" for c, (lbl, name) in norm_robots.items())
            lines.append(f"  \033[1;97;44m[R#]\033[0m Robots Deployed: {r_desc}" if use_color else f"  [R#] Robots Deployed: {r_desc}")
        if norm_tasks:
            t_desc = ", ".join(f"{lbl} ({name}) at {c}" for c, (lbl, name) in norm_tasks.items())
            lines.append(f"  \033[1;93;41m[T#]\033[0m Tasks Located:   {t_desc}" if use_color else f"  [T#] Tasks Located:   {t_desc}")
        if norm_path:
            lines.append(f"  \033[1;92m[··]\033[0m Navigation Path  Waypoints: {len(norm_path)} steps" if use_color else f"  [··] Navigation Path  Waypoints: {len(norm_path)} steps")
        lines.append(f"{BOLD}--------------------------------------------------------------------------------{RESET}")

    return "\n".join(lines)


def print_warehouse_grid(
    warehouse: WarehouseGrid,
    robot_positions: Optional[Union[Dict[str, Any], List[Any]]] = None,
    task_positions: Optional[Union[Dict[str, Any], List[Any]]] = None,
    path: Optional[List[Any]] = None,
    show_legend: bool = True,
    show_coordinates: bool = True,
    use_color: bool = True,
) -> str:
    """Print the 2D warehouse floor map to stdout and return the rendered string."""
    output = render_warehouse_grid(
        warehouse=warehouse,
        robot_positions=robot_positions,
        task_positions=task_positions,
        path=path,
        show_legend=show_legend,
        show_coordinates=show_coordinates,
        use_color=use_color,
    )
    print(output)
    return output


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    # Ensure project root is on sys.path for direct script execution
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    parser = argparse.ArgumentParser(description="E-ITAGS 2D Warehouse Grid Printer")
    parser.add_argument("--no-racks", action="store_true", help="Omit default warehouse storage racks")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color codes")
    parser.add_argument("--no-legend", action="store_true", help="Omit zone and feature legend")
    parser.add_argument("--no-coords", action="store_true", help="Omit coordinate axes")
    parser.add_argument("--no-overlays", action="store_true", help="Omit robot, task, and path overlays")
    args = parser.parse_args()

    grid = WarehouseGrid()
    if not args.no_racks:
        grid.add_default_warehouse_racks()

    sample_robots = None
    sample_tasks = None
    sample_path = None

    if not args.no_overlays:
        sample_robots = {
            "FL": Location(5, 7),
            "DR": Location(4, 15),
            "IW": Location(5, 2),
        }
        sample_tasks = {
            "T1": Location(4, 15),
            "T2": Location(22, 15),
        }
        try:
            from ..core.motion_planner import MotionPlanner
            planner = MotionPlanner(grid)
            res = planner.plan_path(Location(5, 7), Location(22, 15))
            sample_path = res.path
        except (ImportError, ValueError):
            try:
                from src.core.motion_planner import MotionPlanner
                planner = MotionPlanner(grid)
                res = planner.plan_path(Location(5, 7), Location(22, 15))
                sample_path = res.path
            except Exception:
                pass
        except Exception:
            pass

    grid.print_grid(
        robot_positions=sample_robots,
        task_positions=sample_tasks,
        path=sample_path,
        show_legend=not args.no_legend,
        show_coordinates=not args.no_coords,
        use_color=not args.no_color,
    )

