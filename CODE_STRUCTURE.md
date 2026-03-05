# CatherSim Code Structure

This document maps the relationship between files and directories in the `CatherSim` repository.

## Core Components

### 1. Lagrangian Solver Extension
- **[cathetermodule.jl](cathetermodule.jl)**: The primary logic for the simulation.
    - `IrregularBC`: Custom boundary condition handling for complex geometries (polygons).
    - `BD.bd_kernel`: Overridden kernel that performs the particle update loop.
    - **Injection Point**: The `bd_kernel` (Lines 202-275) is where the Lagrangian particle update occurs. This is the ideal location to inject Eulerian chemical grid updates or coupling logic.

### 2. Simulation Entry Points
- **[channel_test.jl](channel_test.jl)**: Example script for running a single simulation.
- **[channel_sweep*.jl](channel_sweep1.jl)** (Consolidated/Deleted): Redundant parameter sweep variants.
    - `channel_sweep1.jl`: `x2=41:2:43`, `x3=41:2:60`, `h=2:0.5:6.5`
    - `channel_sweep2.jl`: `x2=45:2:47`, ...
    - `channel_sweep3.jl`: `x2=49:2:51`, ...
    - `channel_sweep4.jl`: `x2=53:2:55`, ...
    - `channel_sweep5.jl`: `x2=57:2:59`, ...
- **[abp/sweep*.jl](abp/sweep1.jl)** (Consolidated/Deleted):
    - `sweep1.jl` to `sweep5.jl` follow the same `x2` partition as above.
- **[levy/sweep*.jl](levy/sweep1.jl)** (Consolidated/Deleted):
    - `sweep1.jl` to `sweep5.jl` and their `-2.jl` variants swept `x2` values (43, 47, 51, 55, 59) across `x3` sub-ranges.

### 3. Specialized Variants
- **[levy/cathetermodule-levy.jl](levy/cathetermodule-levy.jl)**: Variation implementing Levy flights.
- **[abp/](abp/)**: Active Brownian Particle experimental runs and logs.

## Data Flow: COMSOL Ingestion

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

## Rheotactic Parameters (Primary)

The following parameters from the Science Advances paper (Rheotactic parameters) are preserved in `channel_test.jl` and core modules:
- `U0`: Swimming speed (~20.0 μm/s).
- `DT`: Translational diffusion coefficient (~0.1 μm²/s).
- `DR`: Rotational diffusion coefficient (~0.2 rad²/s).
- `uf`: Flow speed (calculated as max of inlet).

## Artifacts and Redundancies Removed

- **Logs and Outputs**: Removed all `slurm-*.out`, `flow_*.h5`, `*.gif`, and `logdeubg`.
- **Legacy Experiments**: Deleted all dated/named subdirectories in `levy/` (e.g., `0819/`, `0823/`, `polygon/`) which contained redundant snapshots and logs.
- **Duplicate Modules**: Removed `abp/cathetermodule.jl`.
- **Redundant Scripts**: Removed `channel_sweep1.jl` through `channel_sweep5.jl`, `abp/sweep1.jl` through `sweep5.jl`, and `levy/sweep*.jl`.
- **Redundant Workflows**: Removed `channel_paramsweep.jl`, `test.jl`, and interpolation debug scripts (`channel_interpolate*.jl`).
