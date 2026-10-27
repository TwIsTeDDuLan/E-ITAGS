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
