# scripts/render_movie.jl
#
# Renders an animated GIF of bacteria trajectories overlaid on the vorticity
# field and obstacle geometry.
#
# Usage:
#   include("scripts/render_movie.jl")
#   render_movie("results.h5", "flow_obstacles.txt", out_movie="movie.gif")

using HDF5
using Plots
using DelimitedFiles
using Printf
include("../src/Parameters.jl")
using .Parameters

function render_movie(h5file, flowfile; out_movie="simulation.gif", n_frames=50)
    println("=== Rendering Movie ===")
    println("  HDF5: $h5file | Flow: $flowfile")

    Ly = Parameters.Ly
    header = readlines(flowfile)[9]
    flow_Lx = parse(Float64, match(r"Lx=([0-9.]+)", header).captures[1])
    nx = parse(Int, match(r"nx=([0-9]+)", header).captures[1])
    ny = parse(Int, match(r"ny=([0-9]+)", header).captures[1])

    flowdata = readdlm(flowfile, skipstart=9)
    Omega = reshape(flowdata[:,5], (ny, nx))'

    grid_dx = flow_Lx / (nx - 1)
    xgrid = [(i-1) * grid_dx - flow_Lx/2 for i in 1:nx]
    ygrid = collect(LinRange(-Ly/2, Ly/2, ny))

    omega_max = max(0.1, maximum(abs.(Omega)))
    display_omega = clamp.(Omega, -omega_max, omega_max)

    h_val = occursin("smooth", h5file) ? 0.0 : Parameters.h_paper
    d = Parameters.d_paper
    n_half = Int(ceil(flow_Lx / (2 * d)))
    obs_x = [k * d for k in -n_half:n_half]
    filter!(c -> abs(c) < flow_Lx/2 + d/2, obs_x)
    vertices = Float64[]
    for xc in obs_x
        xl = xc - Parameters.L_paper/2
        xr = xc + Parameters.L_paper/2
        xt = xc + Parameters.s_paper
        push!(vertices, xl, -Ly/2, xt, -Ly/2+h_val, xr, -Ly/2)
        push!(vertices, xr,  Ly/2, xt,  Ly/2-h_val, xl,  Ly/2)
    end

    fid = h5open(h5file, "r")
    x_all = read(fid["data/x"])
    img_all = read(fid["data/img"])
    n_samples = size(x_all, 3)
    actual_Lx = haskey(fid, "tiled_Lx") ? read(fid["tiled_Lx"]) : flow_Lx

    step_size = max(1, n_samples ÷ n_frames)
    frame_indices = 1:step_size:n_samples

    anim = @animate for (i, idx) in enumerate(frame_indices)
        if i % 10 == 0 println("  Frame $i / $(length(frame_indices))") end

        x_raw = x_all[1, :, idx]
        img_x = img_all[1, :, idx]
        x_true = x_raw .+ img_x .* actual_Lx
        y_true = x_all[2, :, idx]

        p = heatmap(xgrid, ygrid, display_omega',
                    aspect_ratio=:equal, alpha=0.9, color=:balance, clims=(-omega_max, omega_max),
                    title="t = $(idx) s", size=(1500, 400),
                    xlabel="x (μm)", ylabel="y (μm)", legend=nothing)

        for j in 1:12:length(vertices)
            plot!(p, [vertices[j], vertices[j+2], vertices[j+4], vertices[j]],
                     [vertices[j+1], vertices[j+3], vertices[j+5], vertices[j+1]],
                     color=:black, lw=1.2, label="")
            plot!(p, [vertices[j+6], vertices[j+8], vertices[j+10], vertices[j+6]],
                     [vertices[j+7], vertices[j+9], vertices[j+11], vertices[j+7]],
                     color=:black, lw=1.2, label="")
        end
        hline!(p, [-Ly/2, Ly/2], color=:black, lw=2)
        scatter!(p, x_true, y_true, markersize=0.6, color=:black, alpha=0.3)
        xlims!(p, -flow_Lx/2, flow_Lx/2)
        ylims!(p, -Ly/2 - 5, Ly/2 + 5)
    end

    gif(anim, out_movie, fps=15)
    close(fid)
    println("  Saved: $out_movie")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 || error("Usage: julia render_movie.jl <results.h5> <flow.txt> [out.gif]")
    render_movie(ARGS[1], ARGS[2], out_movie=(length(ARGS) > 2 ? ARGS[3] : "simulation.gif"))
end
