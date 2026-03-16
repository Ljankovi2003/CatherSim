# COMSOL Model Documentation: channel-mask5.mph

Extracted from the binary `.mph` file (COMSOL Multiphysics 5.6, Build 401).
We no longer have a COMSOL license, so this documents everything recoverable
from string extraction of the binary, cross-referenced with the paper.

## Software

- **COMSOL Multiphysics 5.6** (Build 401)
- File: `D:\edmond\mask\mask5.DXF` (imported geometry)
- Model name: `channel-mask5.mph`
- Created/modified: 2022-05-14 (timestamp 1652578288308)

## Physics: Creeping Flow (Stokes)

- **Physics interface:** `CreepingFlow` (tag: `spf`)
  - This is Stokes flow (Re=0): no inertial terms, ∇p = μ∇²u
  - NOT Laminar Flow (Navier-Stokes)
- **Solved fields:** velocity (u, v) and pressure (p)
- **Stationary** solver (not time-dependent)

## Fluid Properties

- **Material:** Water, liquid (tag: `mat2`)
- **Dynamic viscosity:** μ = 1e-3 Pa·s (= 1e-3 kg/(m·s))
- **Density:** ρ = 1000 kg/m³
- **Temperature:** 293.15 K (20°C)
- **Pressure reference:** 1 atm

Note: For Creeping Flow, density only matters for body forces (gravity),
which are not enabled. The flow is entirely determined by μ and the BCs.

## Geometry

- **Type:** 2D, imported from DXF file (`mask5.DXF`)
- **Geometry scale:** 481.96 (COMSOL internal units)
- **Bounding box:** approximately 200 × 100 (in geometry units, likely μm)
  - x: [281.96, 481.96] → width ≈ 200 μm (but this is the DXF import frame)
  - y: [279.05, 379.05] → height ≈ 100 μm
- **Channel width (W):** 100 μm (wall-to-wall, from paper)
- **Unit cell length (d):** 62.26 μm (from paper optimization)
- **Number of geometric entities:** 10 edges, 10 vertices, 1 domain

The DXF mask contains the channel walls with triangular obstacle geometry.
The obstacles are the "shark-fin" triangles on top and bottom walls.

### Obstacle Parameters (from paper optimization)

| Parameter | Value | Description |
|-----------|-------|-------------|
| d | 62.26 μm | Inter-obstacle spacing (periodic length) |
| h | 30.0 μm | Obstacle height |
| s | -19.56 μm | Tip offset from center (asymmetry) |
| L | 15.27 μm | Obstacle base length |
| W | 100 μm | Channel width |

Constraints from paper: d > 0.5W (= 50 μm), h < 0.3W (= 30 μm).

## Boundary Conditions

### Walls (tag: `wall1`)
- **No-slip** boundary condition on all wall surfaces
- Applied to edges 2,3,4,5,6,7,8,9 (the obstacle and channel wall edges)

### Inlet (tag: `inl1`)
- Applied to **edge 1** (left boundary)
- **Fully developed flow** (Poiseuille profile)
- **Average velocity:** `Uavfdf = 1e-5` [m/s] = **10 μm/s**
- Also stored: `V0fdf = 1e-5`, `mfr = 1e-5 kg/s`

### Outlet (tag: `out1`)
- Applied to **edge 10** (right boundary)
- **Pressure condition:** `dp = 5e-4 m` (= 500 μm, likely a reference length)
- Average velocity at outlet: `Uavfdf = 0` (zero — pressure-driven outflow)
- Also: `U0out = 1e-5`

### Periodic Flow Condition (tag: `pfc1`)
- A `PeriodicFlowCondition` was **created** in the model history
- However, the active configuration uses **Inlet/Outlet** BCs instead
- The actions log shows: inlet created, outlet created, then periodic created,
  then inlet re-created — suggesting periodic was tried but replaced

**Interpretation:** The COMSOL model solves a single unit cell with:
- Fully developed Poiseuille inlet (left) at U_avg = 10 μm/s
- Pressure outlet (right)
- No-slip walls (top/bottom + obstacle surfaces)

For our WaterLily reproduction, using **periodic BC in x** with a background
velocity is equivalent for Stokes flow, since the flow is the same in every
unit cell.

## Solver Configuration

- **Stationary** study (not time-dependent)
- Direct solver: MUMPS
- Newton nonlinear iteration
- The model solves for: `comp1.p` (pressure), `comp1.u`, `comp1.v` (velocity)

## Post-Processing (from plot configuration strings)

The COMSOL model had two plot groups configured:
1. **Velocity magnitude** (`spf.U`, unit: m/s) — surface plot
2. **Vorticity** (`spf.vorticityz`, unit: 1/s) — surface plot with velocity arrows
   - Arrow plot: 20×20 grid, showing (u, v) velocity field
   - Frame: material frame

## Physical Scales (for reference)

For Poiseuille flow in a 2D channel of width W=100 μm:
- U_avg = 10 μm/s (from inlet BC)
- U_max = 1.5 × U_avg = 15 μm/s (parabolic profile)
- Wall shear rate: γ̇ = 6 U_avg / W = 0.6 1/s
- Max vorticity (smooth): ω_max = 2 U_max / (W/2) = 0.6 1/s (at walls)
- Re = ρ U_avg W / μ = (1000)(1e-5)(1e-4) / (1e-3) = 0.001

The Reynolds number is O(0.001), confirming Creeping Flow (Stokes) is correct.

## Key Takeaways for Reproduction

1. **Use Stokes flow** (Re → 0), not Navier-Stokes
2. **Single unit cell** of length d=62.26 μm, width W=100 μm
3. **Fully developed Poiseuille** inlet at U_avg = 10 μm/s
4. **No-slip** on all walls including obstacle surfaces
5. **Vorticity in 1/s** — the paper plots `spf.vorticityz`
6. The paper tests three flow speeds: 5, 10, 15 μm/s
7. For WaterLily: periodic x-BC is acceptable (equivalent for Stokes)
8. Need sufficient resolution to resolve obstacle tips (dx, dy < 0.5 μm)
