# CatherSim: CFD Transition (COMSOL to WaterLily.jl)

This document details the migration of the fluid simulation pipeline from proprietary COMSOL to a Julia-native, open-source workflow using `WaterLily.jl`.

## Motivation
- **Workaround for COMSOL**: We added the generated flow files (`flow_*.txt`) as a workaround because a COMSOL license was unavailable.
- **License Independence**: Remove the blocker of requiring a COMSOL license for new simulations.
- **Workflow Integration**: Eliminate manual data export/import steps by scripting the entire pipeline in Julia.
- **Performance**: Leverage GPU acceleration for fast, on-demand steady-state flow generation.

## Technical Implementation

### Legacy COMSOL Workflow
1.  Define geometry (channel + notches) in COMSOL.
2.  Solve 2D Navier-Stokes (steady-state).
3.  Manually export velocity ($U_x, U_y$) and vorticity ($\Omega$) to a `.txt` file.
4.  Ingest `.txt` file into Julia for Lagrangian particle simulation.

### New WaterLily.jl Workflow
1.  **Geometry**: Defined using Signed Distance Functions (SDFs) in [generate_flow.jl](generate_flow.jl).
2.  **Solver**: 2D Navier-Stokes solved on a $1024 \times 256$ grid using `WaterLily.jl`.
3.  **Automation**: Script automatically interpolates results and saves them in the legacy `.txt` format for backwards compatibility.
4.  **Hardware**: Support for GPU execution via `CUDA.jl`.

## Geometry Porting
The triangular notches are ported from the legacy vertex-based description to a composite SDF. The physical domain is mapped to the simulation grid, and the output format is maintained to ensure the Lagrangian solver (`cathetermodule.jl`) remains compatible.

## Usage
To generate a new flow field (local or cluster):
```bash
# Multi-threaded CPU
julia --project=. generate_flow.jl

# GPU (via SLURM)
sbatch submit_cfd.sh
```

To verify the integration with the particle solver:
```bash
julia --project=. verify_integration.jl
```
