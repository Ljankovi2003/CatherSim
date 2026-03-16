# scripts/diagnose_vorticity.jl
#
# Text-based vorticity field diagnostic. Dumps flow/vorticity stats and
# field structure so you can understand the flow without reading PNGs.
#
# Usage (from Julia REPL in CatherSim/):
#   include("scripts/diagnose_vorticity.jl")
#   diagnose_vorticity("flow_smooth_quick.txt", "flow_obstacles_quick.txt")
#   diagnose_vorticity("flow_smooth_quick.txt")  # single file
#
# Or from CLI:
#   julia scripts/diagnose_vorticity.jl flow_smooth_quick.txt flow_obstacles_quick.txt

using DelimitedFiles
using Printf
using Statistics
include("../src/Parameters.jl")
using .Parameters

function load_flow(flowfile::String)
    isfile(flowfile) || error("File not found: $flowfile")
    lines = readlines(flowfile)
    m = match(r"# nx=(\d+), ny=(\d+), Lx=([\d.]+), Ly=([\d.]+)", lines[9])
    nx, ny = parse(Int, m[1]), parse(Int, m[2])
    Lx_f, Ly_f = parse(Float64, m[3]), parse(Float64, m[4])
    data = readdlm(flowfile, skipstart=9)
    # Flow file: x, y, Ux, Uy, Omega per row; row order matches (i,j) over grid
    Ux   = reshape(data[:, 3], (ny, nx))'
    Uy   = reshape(data[:, 4], (ny, nx))'
    vort = reshape(data[:, 5], (ny, nx))'
    xgrid = [(i-1) * Lx_f / (nx-1) - Lx_f/2 for i in 1:nx]
    ygrid = collect(LinRange(-Ly_f/2, Ly_f/2, ny))
    return (; Ux, Uy, vort, xgrid, ygrid, nx, ny, Lx=Lx_f, Ly=Ly_f)
end

function vorticity_summary(flow; label="")
    v = flow.vort
    v_finite = v[isfinite.(v)]
    out = String[]
    push!(out, "  VORTICITY STATS $(label):")
    push!(out, "    min(ω):     $(round(minimum(v_finite), digits=4)) 1/s")
    push!(out, "    max(ω):     $(round(maximum(v_finite), digits=4)) 1/s")
    push!(out, "    max|ω|:     $(round(maximum(abs.(v_finite)), digits=4)) 1/s")
    push!(out, "    mean(ω):    $(round(mean(v_finite), digits=4)) 1/s")
    push!(out, "    std(ω):     $(round(std(v_finite), digits=4)) 1/s")
    # Where is max |ω|?
    idx = argmax(abs.(v))
    i, j = Tuple(CartesianIndices(v)[idx])
    push!(out, "    argmax|ω|:  x=$(round(flow.xgrid[i], digits=1)), y=$(round(flow.ygrid[j], digits=1)) μm")
    push!(out, "")
    push!(out, "  VELOCITY STATS:")
    ux_f = flow.Ux[isfinite.(flow.Ux)]
    push!(out, "    max Ux:     $(round(maximum(ux_f), digits=4)) μm/s")
    push!(out, "    mean Ux:    $(round(mean(ux_f), digits=4)) μm/s")
    return join(out, "\n")
end

function vorticity_slices(flow; y_slices=[-45.0, -40.0, 0.0, 40.0, 45.0], stride_x=1)
    out = String[]
    push!(out, "  VORTICITY SLICES ω(x) at fixed y (1/s):")
    for y_tgt in y_slices
        j = argmin(abs.(flow.ygrid .- y_tgt))
        y_actual = flow.ygrid[j]
        ω_vals = flow.vort[:, j]
        # Sample every stride_x for readability
        xs = flow.xgrid[1:stride_x:end]
        ωs = ω_vals[1:stride_x:end]
        push!(out, "    y=$(round(y_actual, digits=1)) μm:")
        push!(out, "      x:    " * join([@sprintf("%7.1f", x) for x in xs[1:min(20, end)]], " "))
        push!(out, "      ω:    " * join([@sprintf("%7.3f", ω) for ω in ωs[1:min(20, end)]], " "))
        if length(xs) > 20
            push!(out, "      ... (nx=$(flow.nx) total)")
        end
        push!(out, "")
    end
    return join(out, "\n")
end

function vorticity_ascii_map(flow; ncols=80, nrows=24, char_scale=" .:-=+*#%@")
    # Coarse ASCII representation: darker = higher |ω|
    out = String[]
    v = flow.vort
    v_abs = abs.(v)
    vmax = max(1e-6, quantile(vec(v_abs[isfinite.(v_abs)]), 0.98))
    push!(out, "  ASCII VORTICITY MAP (|ω|, max≈$(round(vmax, digits=2)) 1/s):")
    push!(out, "  y→  " * "x→ " * "-"^(ncols-5))
    # Sample grid
    ix = round.(Int, LinRange(1, flow.nx, ncols))
    iy = round.(Int, LinRange(flow.ny, 1, nrows))  # y from top (+Ly/2) to bottom (-Ly/2)
    for (ri, j) in enumerate(iy)
        row = Char[]
        for i in ix
            val = v[i, j]
            lvl = isfinite(val) ? clamp(abs(val) / vmax, 0.0, 1.0) : 0.0
            idx = clamp(round(Int, lvl * (length(char_scale) - 1)) + 1, 1, length(char_scale))
            push!(row, char_scale[idx])
        end
        y_val = flow.ygrid[j]
        push!(out, "  $(rpad(round(y_val, digits=0), 4)) |" * String(row))
    end
    push!(out, "      " * "+" * "-"^(ncols-2) * "+")
    push!(out, "      $(round(flow.xgrid[1], digits=0))" * " "^max(0, ncols-20) * "$(round(flow.xgrid[end], digits=0)) μm")
    return join(out, "\n")
end

function diagnose_one_vorticity(flowfile::String; out_txt=nothing)
    flow = load_flow(flowfile)
    buf = String[]
    push!(buf, "="^70)
    push!(buf, "  $flowfile")
    push!(buf, "  Grid: $(flow.nx)×$(flow.ny), Lx=$(round(flow.Lx, digits=1)), Ly=$(round(flow.Ly, digits=1)) μm")
    push!(buf, "="^70)
    push!(buf, "")
    push!(buf, vorticity_summary(flow))
    push!(buf, vorticity_slices(flow; stride_x=max(1, flow.nx ÷ 30)))
    push!(buf, "")
    push!(buf, vorticity_ascii_map(flow))
    result = join(buf, "\n")
    if out_txt !== nothing
        out_dir = dirname(out_txt)
        isempty(out_dir) || mkpath(out_dir)
        open(out_txt, "w") do f
            println(f, result)
        end
        println("  Wrote: $out_txt")
    end
    return result
end

function diagnose_vorticity(smooth_file::String, obstacle_file::String;
                           out_smooth=joinpath(Parameters.output_dir, "vorticity_smooth.txt"),
                           out_obstacles=joinpath(Parameters.output_dir, "vorticity_obstacles.txt"))
    println(diagnose_one_vorticity(smooth_file, out_txt=out_smooth))
    println()
    println(diagnose_one_vorticity(obstacle_file, out_txt=out_obstacles))
    println()
    # Direct comparison
    fs = load_flow(smooth_file)
    fo = load_flow(obstacle_file)
    println("="^70)
    println("  COMPARISON (paper: obstacles should ENHANCE |ω| at tips)")
    println("="^70)
    println("  Smooth    max|ω| = $(round(maximum(abs.(fs.vort)), digits=4)) 1/s")
    println("  Obstacles max|ω| = $(round(maximum(abs.(fo.vort)), digits=4)) 1/s")
    ratio = maximum(abs.(fo.vort)) / max(1e-10, maximum(abs.(fs.vort)))
    println("  Ratio (obs/smooth) = $(round(ratio, digits=2))")
    if ratio > 1.1
        println("  ✓ Obstacles enhance vorticity (expected)")
    else
        println("  ⚠ Obstacles do NOT enhance vorticity — check flow/geometry")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 2
        diagnose_vorticity(ARGS[1], ARGS[2])
    elseif length(ARGS) == 1
        println(diagnose_one_vorticity(ARGS[1], out_txt=joinpath(Parameters.output_dir, "vorticity_diagnostic.txt")))
    else
        println("Usage: julia diagnose_vorticity.jl <smooth_flow.txt> [obstacles_flow.txt]")
        println("  Or: diagnose_vorticity(\"outputs/flow_smooth_quick.txt\", \"outputs/flow_obstacles_quick.txt\")")
    end
end
