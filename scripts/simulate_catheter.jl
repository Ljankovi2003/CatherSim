# scripts/simulate_catheter.jl
#
# Brownian Dynamics simulation of bacteria in a catheter with flow + obstacles.
# Reads a pre-computed flow field (from generate_flow.jl) and runs
# Lévy run-and-tumble particles on the GPU.
#
# The flow file may be a single unit cell or pre-tiled to the full domain.
# If pre-tiled (Lx_flow >> d_paper), it is used directly.
# If a single cell (Lx_flow ≈ d_paper), it is tiled to fill Parameters.Lx.

include("../src/BD.jl")
using .BD
include("../src/Parameters.jl")
using .Parameters
include("../src/cathetermodule.jl")

using StaticArrays
using LinearAlgebra
using Statistics
using CUDA
using Printf
using HDF5
using Plots
using DelimitedFiles

# ─────────────────────────────────────────────────────────────────────────────
# Flow file loading + tiling
# ─────────────────────────────────────────────────────────────────────────────
function load_and_tile_flow(flowfile; Lx_target=Parameters.Lx, Ly=Parameters.Ly)
    all_lines = readlines(flowfile)
    header = all_lines[9]
    flow_Lx = parse(Float64, match(r"Lx=([0-9.]+)", header).captures[1])
    nx      = parse(Int, match(r"nx=([0-9]+)", header).captures[1])
    ny      = parse(Int, match(r"ny=([0-9]+)", header).captures[1])

    flowdata = readdlm(flowfile, skipstart=9)
    Ux_cell    = reshape(flowdata[:,3], (ny, nx))'
    Uy_cell    = reshape(flowdata[:,4], (ny, nx))'
    Omega_cell = reshape(flowdata[:,5], (ny, nx))'

    grid_dx = flow_Lx / (nx - 1)

    n_tiles = max(1, Int(ceil(Lx_target / flow_Lx)))
    tiled_Lx = n_tiles * flow_Lx

    if n_tiles == 1
        tiled_Ux, tiled_Uy, tiled_O = Ux_cell, Uy_cell, Omega_cell
        tiled_nx = nx
    else
        interior = nx - 1
        tiled_nx = n_tiles * interior + 1
        tiled_Ux = zeros(tiled_nx, ny)
        tiled_Uy = zeros(tiled_nx, ny)
        tiled_O  = zeros(tiled_nx, ny)
        for k in 0:n_tiles-1
            dst = k * interior .+ (1:interior)
            tiled_Ux[dst, :] .= Ux_cell[1:interior, :]
            tiled_Uy[dst, :] .= Uy_cell[1:interior, :]
            tiled_O[dst, :]  .= Omega_cell[1:interior, :]
        end
        tiled_Ux[end, :] .= Ux_cell[1, :]
        tiled_Uy[end, :] .= Uy_cell[1, :]
        tiled_O[end, :]  .= Omega_cell[1, :]
    end

    xgrid = [(i-1) * grid_dx - tiled_Lx/2 for i in 1:tiled_nx]
    ygrid = collect(LinRange(-Ly/2, Ly/2, ny))

    u_peak = maximum(tiled_Ux)
    omega_peak = maximum(abs.(tiled_O))
    println("  Flow: $(tiled_nx)x$(ny) grid, Lx=$(round(tiled_Lx,digits=1)) μm ($(n_tiles) tile(s))")
    println("  Peak u_x = $(round(u_peak, digits=2)) μm/s, max |ω| = $(round(omega_peak, digits=2)) 1/s")

    return (Ux=CuArray(tiled_Ux), Uy=CuArray(tiled_Uy), Omega=CuArray(tiled_O),
            xgrid=CuArray(xgrid), ygrid=CuArray(ygrid),
            xgrid_cpu=xgrid, ygrid_cpu=ygrid,
            tiled_Lx=tiled_Lx, tiled_nx=tiled_nx, ny=ny)
end

# ─────────────────────────────────────────────────────────────────────────────
# Obstacle collider vertices (placed at x = k * d_paper across the domain)
# ─────────────────────────────────────────────────────────────────────────────
function make_obstacle_vertices(tiled_Lx, h; Ly=Parameters.Ly, d=Parameters.d_paper)
    all_verts = Float64[]
    if h < 1e-4
        return CuArray{Float64}(all_verts)
    end
    n_half = Int(ceil(tiled_Lx / (2 * d)))
    for k in -n_half:n_half
        xc = k * d
        abs(xc) > tiled_Lx/2 + d/2 && continue
        xl = xc - Parameters.L_paper/2
        xr = xc + Parameters.L_paper/2
        xt = xc + Parameters.s_paper
        push!(all_verts, xl, -Ly/2, xt, -Ly/2+h, xr, -Ly/2)
        push!(all_verts, xr,  Ly/2, xt,  Ly/2-h, xl,  Ly/2)
    end
    n_obs = length(all_verts) ÷ 12
    println("  Obstacles: $n_obs pairs at d=$(d) μm spacing")
    return CuArray{Float64}(all_verts)
end

# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────
function run_catheter_simulation(; flowfile="flow_obstacles.txt",
                                   out_file="results.h5",
                                   snapshot_file="snapshot.png",
                                   np=Parameters.np_default,
                                   T_end=Parameters.T_end_default,
                                   dt=Parameters.dt_default,
                                   h=Parameters.h_paper)
    println("=== Catheter BD Simulation ===")
    println("  np=$np, T=$(T_end)s, dt=$dt, h=$h")

    flow = load_and_tile_flow(flowfile)
    vertices = make_obstacle_vertices(flow.tiled_Lx, h)
    Ly = Parameters.Ly

    particle = BD.Particle(BD.Point(2), BD.Brownian(Parameters.DT), BD.Brownian(Parameters.DR))
    domain = Irregular(flow.tiled_Lx, Ly)
    bc = IrregularBC(domain, BD.Point(2))

    cache = (U=flow.Ux, V=flow.Uy, Omega=flow.Omega,
             xgrid=flow.xgrid, ygrid=flow.ygrid,
             Nx=flow.tiled_nx, Ny=flow.ny, vertices=vertices,
             alpha=Parameters.alpha, tauR=Parameters.tauR)

    model = BD.IdealModel(particle, domain, bc, np=np,
                          U=linear_velocity_swimonly(domain, BD.Point(2), Parameters.U0),
                          Omega=zero_angular_velocity(domain, BD.Point(2)),
                          cache=cache)

    data = BD.ParticleData(model)
    x0_val = flow.tiled_Lx/2 - 1.0
    y0_dist = (rand(np) .- 0.5) .* Ly * 0.8
    copyto!(data.x, [SVector(x0_val, y0_dist[i]) for i in 1:np])
    copyto!(data.q, rand(np) .* 2π)
    fill!(data.data, 0.0)

    Nt = Int(T_end / dt)
    BD.safecreate_h5file(out_file)
    h5open(out_file, "r+") do file
        write(file, "tiled_Lx", flow.tiled_Lx)
    end
    sample_steps = Int(1/dt)
    writer = BD.PositionWriter(BD.Counter(0, Nt, sample_steps, Nt ÷ sample_steps),
                               out_file, nothing, (true, false))

    sim = BD.Simulation(model, data, BD.EulerMaruyama(particle.linear, particle.angular, dt), Nt, nothing)
    println("  Running $(T_end)s on Lx=$(round(flow.tiled_Lx,digits=1)) μm ...")
    BD.run!(sim, callbacks=(BD.ETA(BD.IterationInterval(50_000), Nt), writer))

    _save_snapshot(data, flow, vertices, Ly, T_end, snapshot_file)
end

function _save_snapshot(data, flow, vertices, Ly, T_end, snapshot_file)
    pos = Array(data.x)
    omega_cpu = Array(flow.Omega)
    omega_max = max(0.1, Float64(maximum(abs.(omega_cpu))))
    display_omega = clamp.(omega_cpu, -omega_max, omega_max)

    p = heatmap(flow.xgrid_cpu, flow.ygrid_cpu, display_omega',
                aspect_ratio=:equal, alpha=0.9, color=:balance, clims=(-omega_max, omega_max),
                title="Bacteria (t=$(T_end)s)", size=(1500, 400),
                xlabel="x (μm)", ylabel="y (μm)")

    v = Array(vertices)
    for j in 1:12:length(v)
        plot!(p, [v[j], v[j+2], v[j+4], v[j]], [v[j+1], v[j+3], v[j+5], v[j+1]],
              fillrange=v[j+1], fillalpha=0.3, fillcolor=:gray, color=:black, lw=1.5, label="")
        plot!(p, [v[j+6], v[j+8], v[j+10], v[j+6]], [v[j+7], v[j+9], v[j+11], v[j+7]],
              fillrange=v[j+7], fillalpha=0.3, fillcolor=:gray, color=:black, lw=1.5, label="")
    end
    hline!(p, [-Ly/2, Ly/2], color=:black, lw=2, label="")
    scatter!(p, getindex.(pos, 1), getindex.(pos, 2), markersize=0.8, color=:black, alpha=0.3, legend=nothing)
    savefig(p, snapshot_file)
    println("  Saved: $snapshot_file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_catheter_simulation()
end
