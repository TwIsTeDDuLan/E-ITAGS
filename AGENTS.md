# E-ITAGS: AI Agent Project Reference

> **Efficacy-optimized Incremental Task Allocation Graph Search**
> Version: 0.1.0 | Python ≥ 3.10 | Phase 1 (Trait Models & Coalition Reasoning)

---

## 1. Project Purpose & Research Context

E-ITAGS is a **multi-robot task allocation (MRTA)** research system that allocates heterogeneous warehouse robots to tasks based on their physical **trait vectors**. It implements the concepts from the paper *"Learning and Optimizing the Efficacy of Task Allocation"* (included as `2.Learning and Optimizing the Efficacy of.pdf`).

### Core Innovation: Realizability-Aware Allocation
Unlike continuous optimizers that assume arbitrary trait combinations exist, E-ITAGS constrains the search space to **physically realizable** coalitions — only trait vectors that can actually be formed by combining the available robots are considered.

### Domain: Autonomous Warehouse Sorting
The system models an autonomous warehouse with:
- **Forklifts** — heavy-payload mast-lift vehicles with 2D LiDAR
- **Drones** — zero-payload aerial units with high-res RGB cameras
- **IW Hub AMRs** — low-profile 220mm autonomous mobile robots that can enter narrow floor stacks

---

## 2. Directory Structure

```
E-ITAGS/
├── pyproject.toml              # Build config, dependencies, pytest settings
├── requirements.txt            # Pip-installable dependencies
├── conftest.py                 # Pytest root config (adds project root to sys.path)
├── demo_phase1.py              # Phase 0 & 1 demo (trait aggregation & satisfaction)
├── demo_multi_robot.py         # Multi-robot fleet demo (6 robots, 41 coalitions)
├── demo_phase2_3.py            # Phase 2 & 3 demo (A* motion planning & timeline scheduling)
├── configs/
│   ├── warehouse_scenario.yaml       # Base scenario: 3 robots + 5 tasks
│   └── warehouse_scenario_multi.yaml # Multi-robot scenario: 6 robots + 7 tasks
├── src/
│   ├── __init__.py             # Package root (__version__ = "0.1.0")
│   ├── core/
│   │   ├── __init__.py         # Exports TraitEngine, MotionPlanner, Schedule
│   │   ├── trait_engine.py     # Coalition enumeration, satisfaction scoring, realizability
│   │   ├── motion_planner.py   # A* grid motion planning, drone flight, zone routing
│   │   └── scheduler.py        # Timeline scheduling, Gantt entries, rendezvous sync
│   ├── models/
│   │   ├── __init__.py         # Exports all model classes
│   │   ├── robot.py            # Robot, RobotTraits, RobotType, TRAIT_NAMES, ZONE_ACCESS
│   │   ├── task.py             # Task, TaskRequirements
│   │   ├── coalition.py        # Coalition (frozen, hashable, sorted comparison)
│   │   └── warehouse.py        # WarehouseGrid, WarehouseZone, Location, obstacles
│   ├── utils/
│   │   ├── __init__.py         # Exports TerminalDisplay
│   │   └── display.py          # Rich terminal output tables and matrices
│   └── visualization/
│       ├── __init__.py         # Exports WarehouseVisualizer, ScheduleVisualizer
│       ├── warehouse_viz.py    # Matplotlib 2D floor-map and path rendering
│       └── schedule_viz.py     # Matplotlib Gantt chart and fleet utilization rendering
├── tests/
│   ├── test_traits.py          # Unit tests for models, coalitions, trait engine, zones
│   ├── test_motion_planner.py  # Unit tests for A* planner, drone flight, zone routing
│   ├── test_scheduler.py       # Unit tests for timeline scheduler, Gantt, utilization
│   └── test_warehouse_viz.py   # Unit tests for warehouse and dashboard visualizers
└── outputs/                    # Exported visualization maps, charts, and diagrams
```

---

## 3. Dependencies

| Package        | Version  | Purpose                                   |
|:---------------|:---------|:------------------------------------------|
| `numpy`        | ≥1.24.0  | Trait vector arithmetic, distance calcs   |
| `matplotlib`   | ≥3.7.0   | 2D warehouse floor map visualization      |
| `networkx`     | ≥3.0     | Graph structures (future phases)          |
| `scikit-learn` | ≥1.3.0   | ML for efficacy learning (future phases)  |
| `pyyaml`       | ≥6.0     | YAML scenario config parsing              |
| `pytest`       | ≥7.0.0   | Unit testing                              |

Install: `pip install -e .` or `pip install -r requirements.txt`

---

## 4. Architecture & Module Reference

### 4.1 Data Models (`src/models/`)

#### `TRAIT_NAMES` (global constant in `robot.py`)
Canonical ordered list of 6 trait dimensions used throughout the system:
```python
["payload_kg", "has_lidar_2d", "has_camera_rgb", "vertical_lift", "narrow_space_access", "speed_mps"]
```
**CRITICAL**: All vector operations (aggregation, satisfaction, realizability) depend on this ordering. Adding/removing traits requires updating `TRAIT_NAMES`, `RobotTraits`, `TaskRequirements`, `TraitEngine.compute_trait_satisfaction()`, and `TerminalDisplay`.

#### `RobotType` (Enum)
Three heterogeneous robot archetypes:
| Type       | Value        | Payload | LiDAR 2D | Camera RGB | Mast Lift | Narrow Access | Speed |
|:-----------|:-------------|:--------|:----------|:-----------|:----------|:--------------|:------|
| `FORKLIFT` | `"forklift"` | 1000 kg | YES       | 0.0        | YES       | NO            | 2.0 m/s |
| `DRONE`    | `"drone"`    | 0 kg    | NO        | 1.0        | NO        | NO            | 5.0 m/s |
| `IW_HUB`   | `"iw_hub"`   | 1000 kg | YES       | 0.5        | NO        | YES           | 2.0 m/s |

#### `ZONE_ACCESS` (global constant in `robot.py`)
Maps each `RobotType` to its `frozenset` of allowed warehouse zone names. Enforced as a hard constraint during task allocation.

| Robot Type | Allowed Zones                                        | Excluded Zones    |
|:-----------|:-----------------------------------------------------|:------------------|
| `FORKLIFT` | `receiving`, `sorting`, `staging`, `shipping`        | `narrow_aisle`    |
| `DRONE`    | ALL zones (unrestricted aerial access)               | (none)            |
| `IW_HUB`   | `sorting`, `narrow_aisle`, `staging`                 | `receiving`, `shipping` |

**Design Rationale**: Forklifts are too large to enter narrow aisles. Drones fly overhead and are unrestricted. IW Hub AMRs are designed for sorting/narrow operations and don't operate in receiving docks or shipping bays.

#### `RobotTraits` (frozen dataclass)
Immutable 6D trait vector `q_i ∈ ℝ^6`. Key methods:
- `to_vector() → np.ndarray` — returns float64 array in `TRAIT_NAMES` order
- `to_dict() → Dict[str, float]` — keyed by trait name
- `from_dict(data) → RobotTraits` — class method constructor from dict

#### `Robot` (frozen dataclass)
Fields: `id`, `name`, `robot_type: RobotType`, `traits: RobotTraits`, `initial_zone: str` (default: `"staging"`).
Factory methods: `create_forklift()`, `create_drone()`, `create_iw_hub()` — each returns a `Robot` with the canonical trait profile for that type.

#### `TaskRequirements` (frozen dataclass)
5D requirement vector (excludes `speed_mps`). Fields: `min_payload_kg`, `min_has_lidar_2d`, `min_has_camera_rgb`, `min_vertical_lift`, `min_narrow_space_access`.
- `to_vector()` returns 5-element array (no speed dimension)

#### `Task` (frozen dataclass)
Fields: `id`, `name`, `zone`, `duration_seconds`, `requirements: TaskRequirements`, `description`, `deadline_seconds` (optional), `release_time_seconds` (default 0.0).

#### `Coalition` (frozen dataclass, hashable)
Represents a subset `C_k = {r_1, r_2, ...}` of robots. Members are stored sorted by `id` for canonical equality/hashing.

**Trait aggregation rules** (`aggregate_traits` property):
| Trait                | Aggregation Rule                                              |
|:---------------------|:--------------------------------------------------------------|
| `payload_kg`         | **Sum** of all members                                        |
| `has_lidar_2d`       | **Max** (1.0 if any member has it)                            |
| `has_camera_rgb`     | **Max** (best camera quality in coalition)                    |
| `vertical_lift`      | **Max** (1.0 if any member has mast lift)                     |
| `narrow_space_access`| **Min** among payload-bearing members (constraining factor)   |
| `speed_mps`          | **Min** of all members (slowest dictates coalition speed)     |

Key methods: `contains(robot)`, `contains_id(robot_id)`, `overlaps_with(other)`, `to_vector()`.

#### `Location` (frozen dataclass)
Simple `(x, y)` integer coordinate. Methods: `manhattan_distance_to()`, `euclidean_distance_to()`.

#### `WarehouseZone` (dataclass)
Fields: `name`, `description`, `center: Location`, `cells: List[Location]`, `narrow_access_only: bool`.

#### `WarehouseGrid`
2D floor map with hardcoded default zones for a 30×20 grid.

**⚠️ Known Limitation — Scalability**: Zone coordinates in `_setup_default_zones()` are hardcoded absolute positions. Changing `width`/`height` from defaults (30×20) breaks zone placement. See "Known Issues" section.

Default zones:
| Zone           | Grid Region       | Center    | Access       |
|:---------------|:------------------|:----------|:-------------|
| `receiving`    | x∈[1,8), y∈[12,18)  | (4, 15)   | General      |
| `sorting`      | x∈[18,27), y∈[12,18) | (22, 15)  | General      |
| `staging`      | x∈[1,9), y∈[5,10)   | (5, 7)    | General      |
| `narrow_aisle` | x∈[1,9), y∈[1,4)    | (5, 2)    | Narrow Only  |
| `shipping`     | x∈[18,27), y∈[2,9)  | (22, 5)   | General      |

### 4.2 Core Engine (`src/core/`)

#### `TraitEngine` (static methods only)
The central algorithmic class. All methods are `@staticmethod`.

| Method | Signature | Description |
|:-------|:----------|:------------|
| `aggregate_coalition_traits` | `(coalition: Coalition) → RobotTraits` | Delegates to `coalition.aggregate_traits` |
| `enumerate_all_coalitions` | `(robots: List[Robot], max_size=3) → List[Coalition]` | Generates all `2^n - 1` non-empty subsets up to `max_size`. With 3 robots → 7 coalitions |
| `compute_realizable_trait_set` | `(robots, max_size=3) → List[Tuple[Coalition, ndarray]]` | Returns all physically possible (coalition, trait_vector) pairs |
| `is_realizable` | `(target_traits, robots, trait_subset=None, tolerance=1e-3) → Tuple[bool, Optional[Coalition]]` | Checks if a target trait vector matches any realizable coalition. Optional `trait_subset` filters to specific dimensions |
| `compute_trait_satisfaction` | `(coalition, task) → Tuple[float, Dict[str, float]]` | Scores how well a coalition satisfies a task's requirements. Returns `(overall_score, per_dimension_breakdown)` |

**Satisfaction Scoring Logic**:
- Each dimension scored independently: ratio-based for `payload` and `camera_rgb`, binary for `lidar_2d`, `vertical_lift`, `narrow_space_access`
- If any **mandatory** binary trait is missing → `overall_score = 0.0` (hard constraint violation)
- Otherwise → `overall_score = mean(dimension_scores)` (all in [0,1])

**Complexity Warning**: `enumerate_all_coalitions` and `compute_realizable_trait_set` enumerate `O(n^k)` combinations (n=robots, k=max_size). The `is_realizable` method calls `compute_realizable_trait_set` on every invocation with no caching — this is `O(n^k)` per call.

### 4.3 Utils (`src/utils/`)

#### `TerminalDisplay`
Rich terminal output for transparency. Methods: `print_header()`, `print_section()`, `display_robots()`, `display_tasks()`, `display_coalitions()`, `display_satisfaction_matrix()`.

### 4.4 Visualization (`src/visualization/`)

#### `WarehouseVisualizer`
Matplotlib-based 2D floor map renderer. Renders zones (colored rectangles), robot positions (colored dots in their `initial_zone`), and task markers (red squares at zone centers). Saves to PNG at 150 DPI.

---

## 5. Configuration Format (`configs/warehouse_scenario.yaml`)

```yaml
scenario_name: "string"
time_budget_seconds: float

robots:
  - id: "R_XX1"          # Unique robot ID (convention: R_{TYPE_CODE}{N})
    name: "Human Name"
    type: "forklift" | "drone" | "iw_hub"
    initial_zone: "staging"  # Must match a WarehouseGrid zone key

tasks:
  - id: "T1"              # Unique task ID (convention: T{N})
    name: "Task Name"
    zone: "receiving"      # Must match a WarehouseGrid zone key
    duration_seconds: 180.0
    description: "Human description"
    requirements:
      min_payload_kg: 850.0
      min_has_lidar_2d: 1.0        # 0.0 = not required, 1.0 = required
      min_has_camera_rgb: 0.0      # 0.0-1.0 threshold
      min_vertical_lift: 1.0       # 0.0 or 1.0
      min_narrow_space_access: 0.0 # 0.0 or 1.0
```

---

## 6. Running the Project

```bash
# Install (editable mode)
pip install -e .

# Run the Phase 1 demo
python demo_phase1.py

# Run tests
pytest -v

# Tests can also be run directly
python tests/test_traits.py
```

The demo performs in sequence:
1. Loads scenario from YAML config
2. Displays robot fleet and trait vectors
3. Displays warehouse tasks and their requirements
4. Enumerates all coalitions (up to size 3) and their aggregated traits
5. Computes and displays the Coalition × Task satisfaction matrix
6. Runs realizability analysis for various payload queries and multi-trait vectors
7. Prints detailed per-task diagnostics (why coalitions succeed or fail)
8. Renders 2D warehouse map to `outputs/warehouse_map.png`

---

## 7. Key Mathematical Concepts

### Trait Vector Space
- Each robot `r_i` has trait vector `q_i ∈ ℝ^6`
- A coalition `C_k` has aggregate trait `Q(C_k)` computed by domain-specific aggregation (sum, max, min depending on trait)
- The **Realizable Trait Set** `Ω` is the finite set of all `Q(C_k)` formable from the robot pool

### Satisfaction Function
`S(C_k, t_j) ∈ [0, 1]` measures how well coalition `C_k` meets task `t_j`'s requirements:
- Hard constraint violation → `S = 0`
- Full satisfaction on all dimensions → `S = 1`
- Partial match → `S = mean(per_dimension_scores)`

### Realizability Check
Given a target trait vector `q*`, determine if `∃ C_k` such that `Q(C_k) ≈ q*` within tolerance. This is what distinguishes E-ITAGS from standard MRTA approaches.

---

## 8. Development Status & Roadmap

### Current: Phase 0, 1, 2, and 3 (Complete)
- ✅ **Phase 0 & 1: Models & Trait Reasoning**:
  - Robot, Task, Coalition data models with 6D trait vectors
  - Trait aggregation with domain-specific physics rules
  - Physical zone access constraints per robot archetype
  - Coalition enumeration up to configurable max size
  - Trait satisfaction scoring (hard + soft constraints)
  - Realizability verification (discrete physical formability check)
  - Terminal display utilities for inspection
  - 2D warehouse visualization & interactive HTML dashboards
- ✅ **Phase 2: A* Motion Planning**:
  - Grid-based A* graph search with diagonal corner-cutting prevention
  - Physical obstacle navigation (central warehouse storage racks)
  - Zone-constrained routing (forklifts route around narrow aisles)
  - 3D aerial direct flight for drones at 5.0 m/s
  - Precomputed zone-to-zone fleet travel time matrices
  - Path memoization/caching for high-speed graph expansion
- ✅ **Phase 3: Timeline Scheduling & Synchronization**:
  - Multi-robot timeline manager tracking robot availability and locations
  - Coalition synchronized rendezvous arrivals
  - Hard task deadline and overall time budget checking
  - Gantt chart generation with travel and execution intervals
  - Fleet workload & resource utilization breakdown
  - 33/33 Unit tests passing (`pytest -v`)

### Next Phases to Implement:
- **Phase 4**: Dual-Objective Heuristics:
  - Normalized Allocation Cost (NAC) for trait matching efficacy
  - Time Budget Overrun (TBO) for schedule penalty
  - Tunable weighting $w \in [0, 1]$ balancing efficacy vs makespan
- **Phase 5**: Interleaved ITAGS Graph Search:
  - Branching search tree combining allocation and scheduling
  - Most-constrained-task selection heuristic
  - Real-time pruning of infeasible branches
- **Phase 6**: Trait-Efficacy Functions & Quality-Optimized Allocation
- **Phase 7**: Realizability-Aware Active Learning with Gaussian Processes

---

## 9. Known Issues & Limitations

1. **WarehouseGrid is not scalable**: All zone coordinates in `_setup_default_zones()` are hardcoded for a 30×20 grid. Changing `width`/`height` causes zones to be misplaced or out of bounds.

2. **No caching in realizability checks**: `is_realizable()` calls `compute_realizable_trait_set()` every time, re-enumerating all coalitions. For larger robot pools this will be very expensive.

3. **TaskRequirements.to_dict() has a latent bug**: Line 40 in `task.py` references `self.min_camera_rgb` (via `hasattr`) which is never defined — the field is named `min_has_camera_rgb`. The `hasattr` guard prevents a crash but the fallback path is unnecessary dead code.

4. **Trait dimension coupling**: Adding a new trait requires changes in 5+ files: `TRAIT_NAMES`, `RobotTraits`, `TaskRequirements`, `TraitEngine.compute_trait_satisfaction()`, `TerminalDisplay`, and possibly `Coalition.aggregate_traits`.

5. **Coalition enumeration is exponential**: `O(n choose k)` grows rapidly. With >10 robots and `max_size=3`, expect thousands of coalitions. No pruning or heuristics are applied.

6. **`time_budget_seconds` in YAML is unused**: The field is parsed but never referenced in any logic.

7. **Distance model is 1:1 cell-to-meter**: `WarehouseGrid.get_distance()` assumes each cell = 1 meter with no configurable resolution.

---

## 10. Coding Conventions

- **Immutability**: All data models (`Robot`, `RobotTraits`, `Task`, `TaskRequirements`, `Coalition`, `Location`) are `frozen=True` dataclasses
- **Static methods**: `TraitEngine` is entirely static — no instance state
- **Imports**: Use relative imports within `src/` (`from ..models.robot import ...`)
- **Type hints**: All function signatures use Python type hints (`from __future__ import annotations`)
- **Naming**: Snake_case for variables/functions, PascalCase for classes, UPPER_SNAKE for constants
- **ID conventions**: Robot IDs follow `R_{TYPE}{N}` pattern (e.g., `R_FL1`, `R_DR1`, `R_IW1`), Task IDs follow `T{N}` (e.g., `T1`, `T2`)
- **No runtime mutation**: Models are created once and never modified
- **Vector ordering**: Always follows `TRAIT_NAMES` canonical order
