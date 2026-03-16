# scripts/generate_flow.jl
#
# Solves the fluid flow for the catheter paper using WaterLily.jl.
#
# APPROACH:
#   1. Solve ONE periodic unit cell (d × W) with WaterLily (Immersed Boundary).
#   2. Extract velocity & vorticity in physical units (μm/s, 1/s).
#   3. Tile the unit cell to fill the full BD domain (Lx).
#   4. Export a flat text file consumed by simulate_catheter.jl.
#
# PHYSICS (from COMSOL .mph / paper):
#   - Creeping Flow (Stokes, Re→0). WaterLily uses Re=0.1 (production) or
#     Re=10 (quick iteration) — both give correct Poiseuille scaling because
#     we rescale to the analytical U_max = 1.5 * U_avg.
#   - Unit cell: d = 62.26 μm (d_paper) × W = 100 μm (Ly).
#   - U_avg = 10 μm/s (paper tests 5, 10, 15).
#   - No-slip walls, periodic in x.
#
# USAGE:
#   run_unit_cell(quick_test=true)                  # fast iteration
#   run_unit_cell(quick_test=false, out_file="...")  # production
#   compare_smooth_vs_obstacles(quick_test=true)     # paper Fig. 2B vs 2C

using WaterLily
using StaticArrays
using DelimitedFiles
using Printf
using LinearAlgebra
using Statistics
using Plots
using CUDA

include("../src/Parameters.jl")
using .Parameters

# ─────────────────────────────────────────────────────────────────────────────
# SDF Geometry
# ─────────────────────────────────────────────────────────────────────────────
@inline function sdf_triangle(p, v1::SVector{2}, v2::SVector{2}, v3::SVector{2})
    verts = (v1, v2, v3)
    d = dot(p - v1, p - v1)
    s = 1.0
    for i in 1:3
        j = i % 3 + 1
        vi, vj = verts[i], verts[j]
        e = vj - vi
        w = p - vi
        b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0)
        d = min(d, dot(b, b))
        c1 = p[2] >= vi[2]
        c2 = p[2] < vj[2]
        c3 = e[1] * w[2] > e[2] * w[1]
        (c1 && c2 && c3) || (!c1 && !c2 && !c3) && (s *= -1.0)
    end
    return s * sqrt(d)
end

function make_unit_cell_geometry(h, Ly, d)
    walls = (p, t) -> min(p[2] - (-Ly/2), (Ly/2) - p[2])
    if h < 1e-4
        return walls
    end
    v_raw = Parameters.get_triangle_vertices(h=h, d=d, n_obs=1)
    v = tuple([SVector(v_raw[i], v_raw[i+1]) for i in 1:2:length(v_raw)]...)
    triangles = (p, t) -> begin
        px_periodic = mod(p[1] + d/2, d) - d/2
        p_p = SVector(px_periodic, p[2])
        return min(sdf_triangle(p_p, v[1], v[2], v[3]), sdf_triangle(p_p, v[4], v[5], v[6]))
    end
    return (p, t) -> min(walls(p, t), triangles(p, t))
end

# ─────────────────────────────────────────────────────────────────────────────
# Resolution presets
# ─────────────────────────────────────────────────────────────────────────────
function get_resolution(quick_test::Bool; use_stokes=false)
    Re = use_stokes ? 1.0 : 10.0  # Re=0.1 → Stokes (round vorticity at tips); Re=10 → faster
    if quick_test
        return (
            Nx = 64,        # ~0.97 μm/cell in x for d=62.26 μm
            Ny = 128,       # ~0.78 μm/cell in y for W=100 μm (8k cells total)
            Re = Re,
            t_end = 5.0
        )
    else
        return (
            Nx = 256,       # ~0.24 μm/cell in x
            Ny = 512,       # ~0.20 μm/cell in y
            Re = Re,
            t_end = 15.0
        )
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Core CFD: solve ONE periodic unit cell [-d/2, d/2] × [-W/2, W/2]
#
# WaterLily non-dimensionalization:
#   Grid size: (Nx, Ny)
#   Background velocity U_grid = 1.0 (in grid units)
#   Characteristic length L_char = Ny/2 (half-channel in grid units)
#   ν_grid = U_grid * L_char / Re
#   Physical mapping: 1 grid cell in x = d/Nx μm, 1 grid cell in y = W/Ny μm
#
# After solving, convert velocities:
#   u_phys (μm/s) = u_grid * (U_phys / U_grid_max)
# where U_grid_max is the centerline velocity from the simulation,
# and U_phys is the desired physical max velocity.
#
# For Poiseuille flow: U_max = 1.5 * U_avg (2D parabolic profile)
# Paper uses U_avg = 10 μm/s → U_max = 15 μm/s
# ─────────────────────────────────────────────────────────────────────────────
function run_unit_cell(; h=Parameters.h_paper, d=Parameters.d_paper, Ly=Parameters.Ly,
                        U_avg_phys=Parameters.U_avg_phys, quick_test=false,
                        use_stokes=false, use_gpu=CUDA.functional(), out_file=nothing)
    res = get_resolution(quick_test; use_stokes=use_stokes)
    Nx, Ny = res.Nx, res.Ny

    dx_phys = d / Nx    # μm per grid cell in x
    dy_phys = Ly / Ny   # μm per grid cell in y
    s_x = Nx / d        # grid cells per μm in x
    s_y = Ny / Ly       # grid cells per μm in y

    L_char = Ny / 2.0
    U_bg = 1.0
    ν_grid = U_bg * L_char / res.Re

    geom = make_unit_cell_geometry(h, Ly, d)

    body = AutoBody((p, t) -> begin
        px_phys = (p[1] - Nx/2) / s_x
        py_phys = (p[2] - Ny/2) / s_y
        geom(SVector(px_phys, py_phys), t) * s_y
    end)

    mem = use_gpu ? CuArray : Array
    sim = Simulation((Nx, Ny), (U_bg, 0.0), L_char;
                     body=body, ν=ν_grid, mem=mem, perdir=(1,))

    mode_str = quick_test ? "QUICK" : "FULL"
    stokes_str = use_stokes ? " Stokes" : ""
    println(">>> Unit-cell CFD [$mode_str$stokes_str] | grid $(Nx)×$(Ny) | Re=$(res.Re)")
    println("    d=$(d) μm, W=$(Ly) μm | dx=$(round(dx_phys,digits=3)) dy=$(round(dy_phys,digits=3)) μm/cell")
    println("    $(use_gpu ? "GPU" : "CPU") | t_end=$(res.t_end)")
    start_time = time()

    dt_print = max(1.0, res.t_end / 10)
    for t in dt_print:dt_print:res.t_end
        sim_step!(sim, t)
        @printf("    t=%.1f / %.1f (%.1fs)\n", t, res.t_end, time() - start_time)
    end
    elapsed = time() - start_time
    @printf("    Done in %.2f s\n\n", elapsed)

    result = (sim=sim, s_x=s_x, s_y=s_y, dx_phys=dx_phys, dy_phys=dy_phys,
              d=d, Ly=Ly, U_avg_phys=U_avg_phys, h=h, elapsed=elapsed)

    if out_file !== nothing
        export_unit_cell(result, out_file)
    end
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# Extract velocity/vorticity from simulation in PHYSICAL units (μm/s, 1/s)
# ─────────────────────────────────────────────────────────────────────────────
function extract_fields(result)
    sim = result.sim
    s_x, s_y = result.s_x, result.s_y
    d, Ly = result.d, result.Ly
    U_avg_phys = result.U_avg_phys

    u_raw = Array(sim.flow.u)
    Nx_grid, Ny_grid = size(sim.flow.p) .- 2
    nx, ny = Nx_grid + 1, Ny_grid + 1
    dx_phys = d / Nx_grid
    dy_phys = Ly / Ny_grid

    ux_grid = zeros(nx, ny)
    uy_grid = zeros(nx, ny)

    for i in 1:nx, j in 1:ny
        px = (i - 1) * dx_phys - d/2
        py = (j - 1) * dy_phys - Ly/2
        gx = px * s_x + Nx_grid / 2.0
        gy = py * s_y + Ny_grid / 2.0

        ir = clamp(floor(Int, gx + 2.0), 1, size(u_raw, 1) - 1)
        jr = clamp(floor(Int, gy + 1.5), 1, size(u_raw, 2) - 1)
        tx = (gx + 2.0) - ir
        ty = (gy + 1.5) - jr
        ux_val = (1-tx)*(1-ty)*u_raw[ir,jr,1] + tx*(1-ty)*u_raw[ir+1,jr,1] +
                 (1-tx)*ty*u_raw[ir,jr+1,1] + tx*ty*u_raw[ir+1,jr+1,1]

        ir2 = clamp(floor(Int, gx + 1.5), 1, size(u_raw, 1) - 1)
        jr2 = clamp(floor(Int, gy + 2.0), 1, size(u_raw, 2) - 1)
        tx2 = (gx + 1.5) - ir2
        ty2 = (gy + 2.0) - jr2
        uy_val = (1-tx2)*(1-ty2)*u_raw[ir2,jr2,2] + tx2*(1-ty2)*u_raw[ir2+1,jr2,2] +
                 (1-tx2)*ty2*u_raw[ir2,jr2+1,2] + tx2*ty2*u_raw[ir2+1,jr2+1,2]

        ux_grid[i,j] = isfinite(ux_val) ? ux_val : 0.0
        uy_grid[i,j] = isfinite(uy_val) ? uy_val : 0.0
    end

    # Scale to match mean velocity (like COMSOL inlet U_avg = 10 μm/s), not max.
    # Legacy used flow-rate-prescribed inlet; we must match mean Ux for both smooth and obstacles.
    ux_finite = ux_grid[isfinite.(ux_grid)]
    U_mean_grid = length(ux_finite) > 0 ? mean(ux_finite) : maximum(ux_grid)
    if U_mean_grid < 1e-10
        @warn "Mean grid velocity near zero ($U_mean_grid). Flow may not have converged."
        U_mean_grid = 1.0
    end

    vel_scale = U_avg_phys / U_mean_grid

    ux_phys = ux_grid .* vel_scale
    uy_phys = uy_grid .* vel_scale

    for i in 1:nx, j in 1:ny
        py = (j - 1) * dy_phys - Ly/2
        if abs(py) >= Ly/2 - 0.5*dy_phys
            ux_phys[i,j] = 0.0
            uy_phys[i,j] = 0.0
        end
    end

    vort = zeros(nx, ny)
    for i in 2:nx-1, j in 2:ny-1
        vort[i,j] = (uy_phys[i+1,j] - uy_phys[i-1,j]) / (2*dx_phys) -
                     (ux_phys[i,j+1] - ux_phys[i,j-1]) / (2*dy_phys)
    end
    for j in 2:ny-1
        vort[1,j] = (uy_phys[2,j] - uy_phys[nx,j]) / (2*dx_phys) -
                     (ux_phys[1,j+1] - ux_phys[1,j-1]) / (2*dy_phys)
        vort[nx,j] = vort[1,j]
    end
    for i in 1:nx, j in 1:ny
        py = (j - 1) * dy_phys - Ly/2
        if abs(py) >= Ly/2 - 0.5*dy_phys
            vort[i,j] = 0.0
        end
        isfinite(vort[i,j]) || (vort[i,j] = 0.0)
    end

    xgrid = [(i-1)*dx_phys - d/2 for i in 1:nx]
    ygrid = [(j-1)*dy_phys - Ly/2 for j in 1:ny]

    println("    Velocity scale: grid_mean=$(round(U_mean_grid,digits=4)) → phys_mean=$(round(U_avg_phys,digits=2)) μm/s (factor=$(round(vel_scale,digits=4)))")
    println("    Max |ω| = $(round(maximum(abs.(vort)),digits=4)) 1/s")

    return (ux=ux_phys, uy=uy_phys, vort=vort, xgrid=xgrid, ygrid=ygrid,
            dx=dx_phys, dy=dy_phys, vel_scale=vel_scale, nx=nx, ny=ny)
end

# ─────────────────────────────────────────────────────────────────────────────
# Tile unit cell to full domain for export/visualization
# ─────────────────────────────────────────────────────────────────────────────
function tile_to_full_domain(fields, d, Lx)
    n_tiles = Int(ceil(Lx / d))
    Lx_tiled = n_tiles * d
    nx_cell = fields.nx
    ny = fields.ny

    nx_tile = nx_cell - 1
    nx_full = n_tiles * nx_tile + 1
    dx = fields.dx

    ux_full = zeros(nx_full, ny)
    uy_full = zeros(nx_full, ny)
    vort_full = zeros(nx_full, ny)

    for t in 0:n_tiles-1
        i_start = t * nx_tile + 1
        for i in 1:nx_tile, j in 1:ny
            ux_full[i_start + i - 1, j] = fields.ux[i, j]
            uy_full[i_start + i - 1, j] = fields.uy[i, j]
            vort_full[i_start + i - 1, j] = fields.vort[i, j]
        end
    end
    for j in 1:ny
        ux_full[nx_full, j] = fields.ux[1, j]
        uy_full[nx_full, j] = fields.uy[1, j]
        vort_full[nx_full, j] = fields.vort[1, j]
    end

    xgrid_full = [(i-1)*dx - Lx_tiled/2 for i in 1:nx_full]
    return (ux=ux_full, uy=uy_full, vort=vort_full, xgrid=xgrid_full,
            ygrid=fields.ygrid, nx=nx_full, ny=ny, dx=dx, dy=fields.dy, Lx=Lx_tiled)
end

# ─────────────────────────────────────────────────────────────────────────────
# Export flow field
# ─────────────────────────────────────────────────────────────────────────────
function export_unit_cell(result, out_fname; Lx=Parameters.Lx)
    fields = extract_fields(result)
    tiled = tile_to_full_domain(fields, result.d, Lx)

    open(out_fname, "w") do f
        for _ in 1:8 println(f, "# Generated by WaterLily.jl (Unit Cell + Tile)") end
        @printf(f, "# nx=%d, ny=%d, Lx=%.2f, Ly=%.2f\n", tiled.nx, tiled.ny, tiled.Lx, result.Ly)
        for i in 1:tiled.nx
            for j in 1:tiled.ny
                @printf(f, "%.6f %.6f %.6f %.6f %.6f\n",
                        tiled.xgrid[i], tiled.ygrid[j],
                        tiled.ux[i,j], tiled.uy[i,j], tiled.vort[i,j])
            end
        end
    end
    println("    Exported: $out_fname ($(tiled.nx)×$(tiled.ny) = $(tiled.nx*tiled.ny) points)")
end

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
function visualize_flow_field(; result=nothing, flow_file=nothing,
                              Lx=Parameters.Lx, Ly=Parameters.Ly,
                              h=Parameters.h_paper, out_file=joinpath(Parameters.output_dir, "flow_field_catheter.png"))
    if result !== nothing
        fields = extract_fields(result)
        tiled = tile_to_full_domain(fields, result.d, Lx)
        xgrid, ygrid = tiled.xgrid, tiled.ygrid
        vort = tiled.vort
    elseif flow_file !== nothing && isfile(flow_file)
        lines = readlines(flow_file)
        m = match(r"# nx=(\d+), ny=(\d+), Lx=([\d.]+), Ly=([\d.]+)", lines[9])
        nx, ny = parse(Int, m[1]), parse(Int, m[2])
        Lx_f, Ly_f = parse(Float64, m[3]), parse(Float64, m[4])
        data = readdlm(flow_file, skipstart=9)
        vort = reshape(data[:,5], (ny, nx))'
        xgrid = [(i-1) * Lx_f / (nx-1) - Lx_f/2 for i in 1:nx]
        ygrid = LinRange(-Ly_f/2, Ly_f/2, ny)
    else
        error("Provide result or flow_file")
    end

    vort_safe = [isfinite(v) ? v : 0.0 for v in vort]
    vmax = max(0.01, quantile(vec(abs.(vort_safe)), 0.98))
    vort_clip = clamp.(vort_safe, -vmax, vmax)

    p = heatmap(collect(xgrid), collect(ygrid), vort_clip', aspect_ratio=:equal,
                color=:balance, clims=(-vmax, vmax), size=(1200, 280),
                xlabel="x (μm)", ylabel="y (μm)",
                title="Flow vorticity (1/s) — full catheter",
                colorbar_title="ω (1/s)")

    if h > 1e-4
        n_obs = Int(ceil(Lx / Parameters.d_paper))
        v_raw = Parameters.get_triangle_vertices(h=h, d=Parameters.d_paper, n_obs=n_obs)
        for k in 1:(length(v_raw)÷6)
            i0 = 6k - 5
            xs = [v_raw[i0], v_raw[i0+2], v_raw[i0+4], v_raw[i0]]
            ys = [v_raw[i0+1], v_raw[i0+3], v_raw[i0+5], v_raw[i0+1]]
            plot!(p, xs, ys, fillrange=ys[1], fillalpha=0.25, fillcolor=:gray, color=:black, lw=1, label="")
        end
    end
    hline!(p, [-Ly/2, Ly/2], color=:black, lw=2, label="")
    mkpath(Parameters.output_dir)
    savefig(p, out_file)
    println("    Saved: $out_file")
    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# Compare smooth vs obstacles (paper Fig. 2B vs 2C)
# ─────────────────────────────────────────────────────────────────────────────
function compare_smooth_vs_obstacles(; quick_test=true,
                                      out_png=joinpath(Parameters.output_dir, "flow_comparison_smooth_vs_obstacles.png"),
                                      shared_scale=true)
    Ly = Parameters.Ly
    d = Parameters.d_paper
    Lx = Parameters.Lx
    mkpath(Parameters.output_dir)

    println(">>> Running SMOOTH channel (h=0)...")
    res_smooth = run_unit_cell(h=0.0, quick_test=quick_test, out_file=joinpath(Parameters.output_dir, "flow_smooth_whole.txt"))
    println(">>> Running OBSTACLES channel (h=$(Parameters.h_paper))...")
    res_obs = run_unit_cell(h=Parameters.h_paper, quick_test=quick_test, out_file=joinpath(Parameters.output_dir, "flow_obstacles_whole.txt"))

    f_smooth = extract_fields(res_smooth)
    f_obs = extract_fields(res_obs)

    t_smooth = tile_to_full_domain(f_smooth, d, Lx)
    t_obs = tile_to_full_domain(f_obs, d, Lx)

    vort_s = [isfinite(v) ? v : 0.0 for v in t_smooth.vort]
    vort_o = [isfinite(v) ? v : 0.0 for v in t_obs.vort]

    vmax_s = max(0.01, quantile(vec(abs.(vort_s)), 0.98))
    vmax_o = max(0.01, quantile(vec(abs.(vort_o)), 0.98))
    vmax_all = max(vmax_s, vmax_o)
    v1, v2 = shared_scale ? (vmax_all, vmax_all) : (vmax_s, vmax_o)

    p1 = heatmap(collect(t_smooth.xgrid), collect(t_smooth.ygrid), clamp.(vort_s', -v1, v1),
                 aspect_ratio=:equal, color=:balance, clims=(-v1, v1), size=(600, 280),
                 xlabel="x (μm)", ylabel="y (μm)",
                 title="(B) Smooth — Poiseuille (max |ω|=$(round(vmax_s, digits=2)) 1/s)")
    hline!(p1, [-Ly/2, Ly/2], color=:black, lw=2, label="")

    p2 = heatmap(collect(t_obs.xgrid), collect(t_obs.ygrid), clamp.(vort_o', -v2, v2),
                 aspect_ratio=:equal, color=:balance, clims=(-v2, v2), size=(600, 280),
                 xlabel="x (μm)", ylabel="y (μm)",
                 title="(C) Obstacles (max |ω|=$(round(vmax_o, digits=2)) 1/s)")
    v_raw = Parameters.get_triangle_vertices(h=Parameters.h_paper, d=d, n_obs=Int(ceil(Lx/d)))
    for k in 1:(length(v_raw)÷6)
        i0 = 6k - 5
        xs = [v_raw[i0], v_raw[i0+2], v_raw[i0+4], v_raw[i0]]
        ys = [v_raw[i0+1], v_raw[i0+3], v_raw[i0+5], v_raw[i0+1]]
        plot!(p2, xs, ys, fillrange=ys[1], fillalpha=0.25, fillcolor=:gray, color=:black, lw=1, label="")
    end
    hline!(p2, [-Ly/2, Ly/2], color=:black, lw=2, label="")

    p_combined = plot(p1, p2, layout=(1,2), size=(1200, 300))
    savefig(p_combined, out_png)
    println("    Saved: $out_png")
    println("\n--- Comparison with paper (Fig. 2B, 2C) ---")
    println("  SMOOTH:    max |ω| = $(round(vmax_s, digits=2)) 1/s")
    println("  OBSTACLES: max |ω| = $(round(vmax_o, digits=2)) 1/s")
    println("  Enhancement ratio: $(round(vmax_o / max(vmax_s, 1e-10), digits=2))×")
    println("  Paper expects: vorticity greatly enhanced near obstacle tips")
    return p_combined
end

# ─────────────────────────────────────────────────────────────────────────────
# Analytical Poiseuille validation (smooth channel, h=0)
# ─────────────────────────────────────────────────────────────────────────────
function validate_poiseuille(result)
    fields = extract_fields(result)
    Ly = result.Ly
    mid_x = fields.nx ÷ 2

    ux_profile = fields.ux[mid_x, :]
    y_profile = fields.ygrid

    U_max_sim = maximum(ux_profile)
    U_max_theory = 1.5 * result.U_avg_phys

    ux_theory = [U_max_theory * (1 - (y / (Ly/2))^2) for y in y_profile]
    ω_theory = [-2 * U_max_theory * y / (Ly/2)^2 for y in y_profile]

    err_vel = sqrt(sum((ux_profile .- ux_theory).^2) / sum(ux_theory.^2))

    println("\n--- Poiseuille Validation ---")
    println("  U_max (sim):    $(round(U_max_sim, digits=4)) μm/s")
    println("  U_max (theory): $(round(U_max_theory, digits=4)) μm/s")
    println("  Relative L2 error: $(round(err_vel*100, digits=2))%")
    println("  ω at wall (theory): $(round(ω_theory[1], digits=4)) 1/s")
    println("  ω at wall (sim):    $(round(fields.vort[mid_x, 2], digits=4)) 1/s")

    p = plot(layout=(1,2), size=(900, 350))
    plot!(p[1], ux_profile, y_profile, label="WaterLily", lw=2)
    plot!(p[1], ux_theory, y_profile, label="Theory", ls=:dash, lw=2)
    xlabel!(p[1], "u_x (μm/s)")
    ylabel!(p[1], "y (μm)")
    title!(p[1], "Velocity profile")

    vort_profile = fields.vort[mid_x, :]
    plot!(p[2], vort_profile, y_profile, label="WaterLily", lw=2)
    plot!(p[2], ω_theory, y_profile, label="Theory", ls=:dash, lw=2)
    xlabel!(p[2], "ω (1/s)")
    ylabel!(p[2], "y (μm)")
    title!(p[2], "Vorticity profile")

    out_path = joinpath(Parameters.output_dir, "poiseuille_validation.png")
    mkpath(Parameters.output_dir)
    savefig(p, out_path)
    println("  Saved: $out_path")
    return p, err_vel
end

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    quick = "--quick" in ARGS
    smooth = "--smooth" in ARGS
    out = joinpath(Parameters.output_dir, "flow_whole_domain.txt")
    for a in ARGS
        startswith(a, "--out=") && (out = split(a, "=")[2])
    end
    mkpath(Parameters.output_dir)
    h = smooth ? 0.0 : Parameters.h_paper
    run_unit_cell(h=h, quick_test=quick, out_file=out)
end
