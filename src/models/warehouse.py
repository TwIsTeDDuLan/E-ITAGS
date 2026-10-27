"""Warehouse spatial representation and zones."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
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
