# CatherSim Legacy Documentation

This document consolidates the full contents of the original `CODE_STRUCTURE.md` and `DEVELOPER_GUIDE.md` for historical reference.

---

## Part 1: CODE_STRUCTURE.md

### Core Components

#### 1. Lagrangian Solver Extension
- **[cathetermodule.jl](cathetermodule.jl)**: The primary logic for the simulation.
    - `IrregularBC`: Custom boundary condition handling for complex geometries (polygons).
    - `BD.bd_kernel`: Overridden kernel that performs the particle update loop.
    - **Injection Point**: The `bd_kernel` (Lines 202-275) is where the Lagrangian particle update occurs. This is the ideal location to inject Eulerian chemical grid updates or coupling logic.

#### 2. Simulation Entry Points
- **[channel_test.jl](channel_test.jl)**: Example script for running a single simulation.
- **[channel_sweep*.jl](channel_sweep1.jl)** (Consolidated/Deleted): Redundant parameter sweep variants.
    - `channel_sweep1.jl`: `x2=41:2:43`, `x3=41:2:60`, `h=2:0.5:6.5`
- **[abp/sweep*.jl](abp/sweep1.jl)** (Consolidated/Deleted):
    - `sweep1.jl` to `sweep5.jl` follow the same `x2` partition as above.
- **[levy/sweep*.jl](levy/sweep1.jl)** (Consolidated/Deleted):
    - `sweep1.jl` to `sweep5.jl` and their `-2.jl` variants swept `x2` values (43, 47, 51, 55, 59) across `x3` sub-ranges.

#### 3. Specialized Variants
- **[levy/cathetermodule-levy.jl](levy/cathetermodule-levy.jl)**: Variation implementing Levy flights.
- **[abp/](abp/)**: Active Brownian Particle experimental runs and logs.

### Data Flow: COMSOL Ingestion

The simulation ingests fluid velocity ($\mathbf{u}$) and vorticity ($\boldsymbol{\omega}$) fields exported from COMSOL in `.txt` (delimited) format.

- **File Pattern**: `flow_{x2}_{x3}_{h}.txt`
- **Schema**:
    - **Header**: Skip first 9 lines.
    - **Column 3**: $U_x$ (Velocity X)
    - **Column 4**: $U_y$ (Velocity Y)
    - **Column 5**: $\Omega$ (Vorticity)
- **Resolution**: Typically $1001 \times 201$.
- **Preprocessing**: Data is reshaped to `(Nx, Ny)` and moved to GPU memory (`CUDA.zeros`, `copyto!`).
- **Interpolation**: Bilinear interpolation is performed on-the-fly within `bd_kernel` using `interpolate_u`.

### Rheotactic Parameters (Primary)

The following parameters from the Science Advances paper (Rheotactic parameters) are preserved in `channel_test.jl` and core modules:
- `U0`: Swimming speed (~20.0 μm/s).
- `DT`: Translational diffusion coefficient (~0.1 μm²/s).
- `DR`: Rotational diffusion coefficient (~0.2 rad²/s).
- `uf`: Flow speed (calculated as max of inlet).

---

## Part 2: DEVELOPER_GUIDE.md

### Quick Start & Environment
- **Julia**: Tested with Julia 1.7 on CUDA GPUs.
- **Setup**: `julia --project=.` -> `using Pkg; Pkg.instantiate()`.
- **Primary Driver**: `julia channel_test.jl` (baseline for new simulations).
- **GPU Caches**: External fields (`Ux`, `Uy`, `Omega`) and domain data (grids, vertices) are stored in GPU memory via `CUDA.jl`.

### Solver Logic & Physics

#### 1. Geometry and Boundary Conditions
- **Irregular Domain**: A 2D rectangular domain (`Irregular`) that subclasses `BD.AbstractDomain`.
- **IrregularBC**: Custom boundary condition handling:
    - **Wall Flux**: Applied in `y` via `BD.parallel_flat_walls`.
    - **Polygonal Obstacles**: Defined by contiguous triangles in the `vertices` cache.
    - **Collision Logic**: Particles are reflected across triangle edges when penetration is detected.
    - **Periodicity**: Applied in `x` via `BD.periodicbc`.

#### 2. Flow Interpolation
- **Search**: `findind` uses binary search to locate grid indices on a monotone 1D grid.
- **Bilinear Interpolation**: `interpolate_u` calculates scalar values on-the-fly for velocity and vorticity components. Reused across the entire domain on the GPU.

#### 3. Kernel Override (`BD.bd_kernel`)
The core update loop is overridden to:
1. Interpolate local fluid velocity/vorticity from GPU caches.
2. Combine fluid flow with self-propulsion and Brownian stochastic velocities.
3. Apply custom boundary conditions (walls, triangles, periodicity).
4. Update periodic image counters and record data via callbacks.

### Lévy Reorientation Variant (`cathetermodule-levy.jl`)

This module extends the core solver with orientation reset logic:
- **Lévy Stable Distribution**: `generate_runtime_levy` samples run times using a stable distribution power-law ($\tau \sim U^{-1/\alpha}$).
- **Reset Mechanism**: When a particle's internal clock (`data[i]`) is exceeded, it draws a new run time and a new random orientation.
- **Initialization**: Requires `ParticleData.data` to be initialized with initial reorientation schedules.

### Flow Field Generation (WaterLily.jl vs COMSOL)
The simulation requires 2D velocity and vorticity fields as input. Historically, these were generated using COMSOL, but a Julia-native alternative has been implemented.

- **Legacy COMSOL**: Tab-delimited `.txt` files with 9-line headers and $1001 \times 201$ resolution.
- **Julia Alternative**: [generate_flow.jl](generate_flow.jl) uses `WaterLily.jl` to generate compatible flow fields without requiring a COMSOL license.

#### Data Schema
- **Input**: Tab-delimited `.txt` files (e.g., `flow_41_41_2.txt`).
- **Header**: First 9 lines are ignored.
- **Resolution**: Flexible (WaterLily version uses $1024 \times 256$ for performance).
- **Columns**: $U$ (Col 3), $V$ (Col 4), $\Omega$ (Col 5).
