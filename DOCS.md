# CatherSim — Documentation

Reference: Zhou et al., "AI-aided geometric design of anti-infection catheters,"
Science Advances 10(1), eadj1741 (2024).

---

## 1. Paper Physics

### 1.1 Problem

Bacteria swim upstream in catheters via a vorticity-reorientation mechanism.
Triangular obstacles on the channel walls enhance local vorticity at their tips,
redirecting bacteria downstream. The paper optimizes obstacle geometry
(d, h, s, L) using Geo-FNO to minimize upstream contamination.

### 1.2 Fluid Phase — Stokes Flow

The paper solves **Creeping Flow** (Stokes, Re → 0) in COMSOL:

    ∇p = μ∇²u,  ∇·u = 0

- No inertial terms (Re ≈ 0.001 at the physical scale)
- Water: μ = 1e-3 Pa·s, ρ = 1000 kg/m³
- Channel width W = 100 μm
- Inlet: fully developed Poiseuille, U_avg = 10 μm/s (paper tests 5, 10, 15)
- No-slip on all walls and obstacle surfaces
- Single periodic unit cell of length d (inter-obstacle spacing)

### 1.3 Particle Phase — Run-and-Tumble with Lévy Statistics

Bacteria are point-like spheres (Bretherton constant B=0). During a "run":

    dx = [U₀ q + u(x)] dt + √(2 D_T) dW_x
    dθ = ω(x) dt + √(2 D_R) dW_θ

where ω(x) is the **full vorticity** (not half — B=0 for spheres).

At the end of each run, a "tumble" randomizes θ uniformly in [0, 2π).
Runtime τ is drawn from a **Pareto distribution** (Lévy flight):

    φ(τ) = α τ_min^α τ^(-α-1),  τ_min = (α-1)τ_R/α

| Parameter | Symbol | Value | Defined in |
|-----------|--------|-------|------------|
| Swim speed | U₀ | 20.0 μm/s | `Parameters.U0` |
| Translational diffusion | D_T | 0.1 μm²/s | `Parameters.DT` |
| Rotational diffusion | D_R | 0.2 rad²/s | `Parameters.DR` |
| Pareto exponent | α | 1.2 | `Parameters.alpha` |
| Mean runtime | τ_R | 2.0 s | `Parameters.tauR` |
| Timestep | dt | 1e-4 s | `Parameters.dt_default` |

### 1.4 Boundary Conditions

- **Periodic in x**: bacteria wrapping past ±Lx/2 reappear on the other side.
  Image counters track total displacement for upstream distance metrics.
- **No-flux walls in y**: specular reflection at y = ±W/2.
- **Obstacle collision**: reflection off triangle edges (two-segment wedge test).

### 1.5 Optimized Obstacle Geometry

| Parameter | Symbol | Value | Defined in |
|-----------|--------|-------|------------|
| Inter-obstacle spacing | d | 62.26 μm | `Parameters.d_paper` |
| Obstacle height | h | 30.0 μm | `Parameters.h_paper` |
| Tip offset (asymmetry) | s | -19.56 μm | `Parameters.s_paper` |
| Base length | L | 15.27 μm | `Parameters.L_paper` |
| Channel width | W | 100.0 μm | `Parameters.Ly` |

Constraints: d > 0.5W = 50 μm, h < 0.3W = 30 μm.

---

## 2. Reproduction Status & Mechanism

### 2.1 Current Status: ✓ Reproduction Working

After fixing the flow scaling (see §2.2), obstacles correctly **reduce** upstream penetration:

| Metric | Smooth | Obstacles | Paper Expectation |
|--------|--------|-----------|-------------------|
| Upstream % | 2.2% | **0.6%** | Obstacles **reduce** ✓ |
| ⟨x_up⟩ all | -346 μm | -395 μm | Obstacles more negative ✓ |
| x_1% (99th pct) | 51 μm | -15 μm | Obstacles **lower** ✓ |
| max(x_up) | 348 μm | 44 μm | — |

Flow stats (both cases matched):
- mean Ux: ~10 μm/s (smooth 9.99, obstacles 10.0)
- max |ω|: smooth 0.89, obstacles 5.06 (5.7× enhancement at tips)

### 2.2 Same Mechanism as the Paper?

**Yes.** Our implementation uses the same two mechanisms:

1. **Vorticity reorientation**: Orientation evolves as dθ/dt = ω(x) + noise. Enhanced vorticity at obstacle tips rotates bacteria downstream, suppressing upstream swimming.
2. **Geometric rectification**: Obstacle collision reflects particles. The slope of the triangles biases motion and interrupts continuous wall-climbing.

The BD kernel interpolates ω from the flow field and applies obstacle reflection. The flow is scaled to match mean Ux (flow rate), so the comparison is fair.

### 2.3 Critical Fix: Velocity Scaling (Mean vs Max)

**COMSOL (legacy)** uses a **flow-rate-prescribed inlet**: U_avg = 10 μm/s. The solver adjusts pressure to achieve that flow. Both smooth and obstacle geometries get mean Ux ≈ 10 μm/s.

**Our WaterLily** must scale to match **mean velocity**, not max:

```julia
# scripts/generate_flow.jl extract_fields()
U_mean_grid = mean(ux_grid[isfinite.(ux_grid)])
vel_scale = U_avg_phys / U_mean_grid
```

**Lesson**: Always scale flow to match mean Ux (flow rate), not max Ux, when comparing smooth vs obstacle channels.

### 2.4 Vorticity Shape: Round vs Flat

At Re=10, vorticity near obstacle tips can appear **flat/elongated** (shear layer). At **Re=1.0 (Stokes)**, vorticity tends to be more **round** (corner singularity). Use `quick(use_stokes=true)` or `reproduce(use_stokes=true)` for Stokes-like, round vorticity at tips.

---

## 3. Legacy Code

Located in `legacy/`. The original pipeline was:

1. **COMSOL** (`channel-mask5.mph`): Stokes flow on imported DXF mask geometry.
   Exported velocity (u,v) and vorticity (ω) on a regular grid.
   See `legacy/COMSOL_DOCUMENTATION.md` for full extraction from the .mph binary.

2. **BD package** (proprietary Julia package, now lost): GPU Brownian Dynamics.
   `legacy/cathetermodule.jl` extended it with `Irregular` domain, `IrregularBC`,
   `bd_kernel`, `findind`, `interpolate_u`.

3. **channel_test.jl**: Driver script loading flow data and running BD.

### Legacy vs Current

| Aspect | Legacy (COMSOL) | Current (WaterLily) |
|--------|-----------------|---------------------|
| Flow prescription | Inlet U_avg = 10 μm/s | Scale to mean Ux = 10 μm/s |
| Particle model | ABP (Brownian) | Lévy run-and-tumble |
| Domain | Lx=100, Ly=20 μm | Lx≈560, Ly=100 μm |
| Obstacle geometry | x2,x3,h (legacy params) | d,h,s,L (paper params) |

---

## 4. Our Implementation

### 4.1 File Layout

```
reproduce_paper.jl              — Entry point: quick() or reproduce()

src/
  Parameters.jl                 — All physical constants, output_dir
  BD.jl                         — Replacement BD module (GPU CUDA kernels)
  cathetermodule.jl             — Irregular domain, obstacle BC, flow interpolation

scripts/
  generate_flow.jl              — WaterLily CFD (replaces COMSOL)
  simulate_catheter.jl          — Particle BD simulation
  compute_upstream.jl           — Post-processing: ⟨x_up⟩, x₁%
  render_movie.jl               — Animated GIF from HDF5 trajectories
  diagnose_results.jl           — Text-based particle metrics
  diagnose_vorticity.jl         — Text-based flow/vorticity diagnostics

legacy/
  cathetermodule.jl             — Original BD extension code
  channel_test.jl               — Original driver script
  COMSOL_DOCUMENTATION.md       — Extracted COMSOL setup details
```

**All outputs** go to `outputs/` (see `Parameters.output_dir`).

### 4.2 Key Terminology

| Term | Meaning | Value / Location |
|------|---------|------------------|
| **d** (= `d_paper`) | Unit cell period / inter-obstacle spacing | 62.26 μm |
| **W** (= `Ly`) | Channel width | 100 μm |
| **Lx** | Full BD simulation domain length | 500 μm |
| **tiled_Lx** | Actual tiled domain: `ceil(Lx/d) × d` | ~560 μm (9 tiles) |
| **Nx, Ny** (CFD) | Grid resolution of one unit cell | 64×128 (quick) or 256×512 (production) |
| **output_dir** | All outputs directory | `outputs/` |

### 4.3 Two-Stage Domain Strategy

1. **CFD stage** (`generate_flow.jl`): solves flow on ONE periodic unit cell of length **d** = 62.26 μm.
2. **BD stage** (`simulate_catheter.jl`): tiles the unit cell to fill **Lx** ≈ 560 μm. Obstacles at `x = k × d`.

### 4.4 Flow Field Generation (`generate_flow.jl`)

Key functions:
- `run_unit_cell(; h, d, Ly, quick_test, use_stokes, out_file)` — main entry
- `extract_fields(result)` — converts grid units → physical μm/s, 1/s (scales by **mean** Ux)
- `tile_to_full_domain(fields, d, Lx)` — repeats unit cell to fill Lx
- `export_unit_cell(result, filename)` — writes tiled flow to text
- `validate_poiseuille(result)` — checks smooth channel vs analytical solution

**Resolution presets** (`get_resolution`):

| Mode | Nx × Ny | Re | t_end |
|------|---------|-----|-------|
| quick_test | 64 × 128 | 1.0 (stokes) or 10.0 | 5.0 |
| production | 256 × 512 | 1.0 (stokes) or 10.0 | 15.0 |

Use `use_stokes=true` for Re=1.0 (round vorticity at tips, matches COMSOL Stokes).

### 4.5 Particle Simulation (`simulate_catheter.jl`)

- `load_and_tile_flow(flowfile)` — reads flow file, tiles if needed
- `make_obstacle_vertices(tiled_Lx, h)` — places colliders at `x = k × d_paper`
- `run_catheter_simulation(; flowfile, out_file, snapshot_file, np, T_end, h)` — main entry

**Output**: HDF5 in `outputs/` with `data/x`, `data/q`, `data/img`, `tiled_Lx`.

### 4.6 BD.jl & cathetermodule.jl

- `obstacle()` — two-segment wedge collision; reflects particle if inside
- `IrregularBC` — wall reflection + obstacle collision + periodic BC
- `bd_kernel` — CUDA kernel with flow interpolation, Lévy tumble
- `generate_runtime_levy_gpu()` — Pareto-distributed runtime

---

## 5. Coordinate System and Units

- x ∈ [-tiled_Lx/2, +tiled_Lx/2] (flow +x, periodic)
- y ∈ [-W/2, +W/2] (wall-normal, reflecting)
- Upstream displacement: `x_up = x_initial - x_current` (positive = swam upstream)

| Quantity | Unit |
|----------|------|
| Position | μm |
| Velocity | μm/s |
| Vorticity | 1/s |
| Time | s |

---

## 6. Quick Start

```julia
# From Julia REPL in CatherSim/
include("reproduce_paper.jl")

# Fast end-to-end (~minutes)
quick()

# Stokes flow (round vorticity at tips)
quick(use_stokes=true)

# Full reproduction (~hours)
reproduce()

# Options
quick(np=10000, T=100.0, regen_flow=true, make_movie=true, use_stokes=true)
```

### Step by step

```julia
include("scripts/generate_flow.jl")
mkpath(Parameters.output_dir)

res = run_unit_cell(h=0.0, quick_test=true, out_file=joinpath(Parameters.output_dir, "flow_smooth.txt"))
validate_poiseuille(res)

run_unit_cell(h=Parameters.h_paper, quick_test=true, out_file=joinpath(Parameters.output_dir, "flow_obs.txt"))

include("scripts/simulate_catheter.jl")
run_catheter_simulation(flowfile=joinpath(Parameters.output_dir, "flow_obs.txt"),
                        out_file=joinpath(Parameters.output_dir, "results.h5"),
                        snapshot_file=joinpath(Parameters.output_dir, "snapshot.png"),
                        np=5000, T_end=50.0)

include("scripts/compute_upstream.jl")
compute_upstream(joinpath(Parameters.output_dir, "results.h5"))
```

### Diagnostics

```julia
# Particle metrics (text)
include("scripts/diagnose_results.jl")
diagnose("outputs/results_smooth_quick.h5", "outputs/results_obstacles_quick.h5")

# Flow/vorticity (text)
include("scripts/diagnose_vorticity.jl")
diagnose_vorticity("outputs/flow_smooth_quick.txt", "outputs/flow_obstacles_quick.txt")
```

---

## 7. Verification Checklist

When changing flow generation or scaling:

1. **Mean Ux** in both smooth and obstacle flows ≈ 10 μm/s (`diagnose_vorticity`)
2. **Obstacles reduce** upstream % and x_1% (`diagnose_results`)
3. **max |ω|** higher for obstacles than smooth (vorticity enhancement at tips)

---

## 8. Known Limitations

1. **IB smearing**: WaterLily's immersed boundary smooths geometry over ~2 grid cells.
2. **Re=10 vs Stokes**: Default Re=10 for speed. Use `use_stokes=true` for Stokes (Re=1).
3. **2D only**: Paper notes 3D effects differ but qualitative trend persists.
4. **No FNO training**: Geo-FNO optimization pipeline not implemented.



### Current Reproduction Results
  ── SMOOTH CHANNEL ──
  outputs/results_smooth_quick.h5: 5000 particles, 50 snapshots, Lx=560.3 μm
  Snapshot 50 (final):
    Upstream:  106 / 5000 (2.1%)
    <x_up> all:       -345.03 μm
    <x_up> upstream:     78.27 μm
    x_1% (99th pct):    71.89 μm
    max(x_up):         272.49 μm

  ── OBSTACLE CHANNEL ──
  outputs/results_obstacles_quick.h5: 5000 particles, 50 snapshots, Lx=560.3 μm
  Snapshot 50 (final):
    Upstream:  33 / 5000 (0.7%)
    <x_up> all:       -383.62 μm
    <x_up> upstream:     22.66 μm
    x_1% (99th pct):    -9.42 μm
    max(x_up):          54.90 μm
