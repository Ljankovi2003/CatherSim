# CatherSim Developer Guide

This guide provides technical details on the Brownian-dynamics solver and simulation environment for the CatherSim project.

## Quick Start & Environment
- **Julia**: Tested with Julia 1.7 on CUDA GPUs.
- **Setup**: `julia --project=.` -> `using Pkg; Pkg.instantiate()`.
- **Primary Driver**: `julia channel_test.jl` (baseline for new simulations).
- **GPU Caches**: External fields (`Ux`, `Uy`, `Omega`) and domain data (grids, vertices) are stored in GPU memory via `CUDA.jl`.

## Solver Logic & Physics

### 1. Geometry and Boundary Conditions
- **Irregular Domain**: A 2D rectangular domain (`Irregular`) that subclasses `BD.AbstractDomain`.
- **IrregularBC**: Custom boundary condition handling:
    - **Wall Flux**: Applied in `y` via `BD.parallel_flat_walls`.
    - **Polygonal Obstacles**: Defined by contiguous triangles in the `vertices` cache.
    - **Collision Logic**: Particles are reflected across triangle edges when penetration is detected.
    - **Periodicity**: Applied in `x` via `BD.periodicbc`.

### 2. Flow Interpolation
- **Search**: `findind` uses binary search to locate grid indices on a monotone 1D grid.
- **Bilinear Interpolation**: `interpolate_u` calculates scalar values on-the-fly for velocity and vorticity components. Reused across the entire domain on the GPU.

### 3. Kernel Override (`BD.bd_kernel`)
The core update loop is overridden to:
1. Interpolate local fluid velocity/vorticity from GPU caches.
2. Combine fluid flow with self-propulsion and Brownian stochastic velocities.
3. Apply custom boundary conditions (walls, triangles, periodicity).
4. Update periodic image counters and record data via callbacks.

## Lévy Reorientation Variant (`cathetermodule-levy.jl`)

This module extends the core solver with orientation reset logic:
- **Lévy Stable Distribution**: `generate_runtime_levy` samples run times using a stable distribution power-law ($\tau \sim U^{-1/\alpha}$).
- **Reset Mechanism**: When a particle's internal clock (`data[i]`) is exceeded, it draws a new run time and a new random orientation.
- **Initialization**: Requires `ParticleData.data` to be initialized with initial reorientation schedules.

## Flow Field Generation (WaterLily.jl vs COMSOL)
The simulation requires 2D velocity and vorticity fields as input. Historically, these were generated using COMSOL, but a Julia-native alternative has been implemented.

- **Legacy COMSOL**: Tab-delimited `.txt` files with 9-line headers and $1001 \times 201$ resolution.
- **Julia Alternative**: [generate_flow.jl](generate_flow.jl) uses `WaterLily.jl` to generate compatible flow fields without requiring a COMSOL license. This is the recommended troubleshooting path when licenses are unavailable.

### Data Schema
- **Input**: Tab-delimited `.txt` files (e.g., `flow_41_41_2.txt`).
- **Header**: First 9 lines are ignored.
- **Resolution**: Flexible (WaterLily version uses $1024 \times 256$ for performance).
- **Columns**: $U$ (Col 3), $V$ (Col 4), $\Omega$ (Col 5).

For more details, see [CFD_TRANSITION.md](CFD_TRANSITION.md).
