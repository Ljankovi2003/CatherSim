module Parameters

using StaticArrays

# All simulation outputs go here. Create with mkdir(Parameters.output_dir, exist_ok=true).
const output_dir = "outputs"

# =========================================================================
# 1. CHANNEL & OBSTACLE GEOMETRY
# =========================================================================
# The catheter is a 2D channel of width W (= Ly) with periodic triangular
# obstacles spaced every d μm along the flow direction.
#
# CFD solves ONE unit cell of length d, then tiles it to fill Lx.
# BD simulation runs on the tiled domain of length Lx.

const Ly = 100.0        # Channel width W (μm). Paper: W = 100 μm.
const Lx = 500.0        # Full physics domain length (μm) for BD simulation.

# Optimized obstacle geometry (paper Table / Fig. 2G):
const d_paper = 62.26    # Inter-obstacle spacing / unit cell period (μm)
const h_paper = 30.0     # Obstacle height (μm)
const s_paper = -19.56   # Tip offset from obstacle center (μm), negative = upstream-leaning
const L_paper = 15.27    # Obstacle base length (μm)

# =========================================================================
# 2. BACTERIA PHYSICS (E. coli, paper Materials & Methods)
# =========================================================================
const U0    = 20.0       # Swimming speed (μm/s)
const DT    = 0.1        # Translational diffusivity (μm²/s)
const DR    = 0.2        # Rotational diffusivity (rad²/s)
const alpha = 1.2        # Pareto exponent for Lévy run-time distribution
const tauR  = 2.0        # Mean runtime (s)

# =========================================================================
# 3. SIMULATION DEFAULTS
# =========================================================================
const U_avg_phys = 10.0  # Mean flow velocity (μm/s). Paper tests 5, 10, 15.
const np_default = 20_000
const T_end_default = 200.0   # Evaluation time (s). Paper uses T=500s for Fig. 2F.
const dt_default = 1e-4       # Euler-Maruyama timestep (s)

# =========================================================================
# 4. OBSTACLE VERTEX GENERATION
# =========================================================================
"""
    get_triangle_vertices(; d, h, s, L, n_obs) -> Vector{Float64}

Returns a flat array of vertex coordinates for `n_obs` obstacle pairs
(bottom + top fin), centered around x=0 and spaced by `d`.

Each obstacle pair is 12 floats: [bottom: x_left, y_left, x_tip, y_tip,
x_right, y_right, top: x_right, y_right, x_tip, y_tip, x_left, y_left].

Used by both the CFD SDF (via `make_unit_cell_geometry`) and the BD
collider (via `simulate_catheter.jl` obstacle placement).
"""
function get_triangle_vertices(; d=d_paper, h=h_paper, s=s_paper, L=L_paper, n_obs=1)
    all_verts = Float64[]
    x_starts = [(i - (n_obs+1)/2) * d for i in 1:n_obs]

    for x_base in x_starts
        x_left  = x_base - L/2
        x_right = x_base + L/2
        x_tip   = x_base + s

        push!(all_verts, x_left, -Ly/2.0, x_tip, -Ly/2.0+h, x_right, -Ly/2.0)
        push!(all_verts, x_right, Ly/2.0, x_tip, Ly/2.0-h, x_left, Ly/2.0)
    end
    return all_verts
end

end # module Parameters
