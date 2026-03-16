# scripts/diagnose_results.jl
#
# Text-based diagnostic output for comparing simulation results.
# Easier to parse than PNG snapshots for debugging divergence from paper/legacy.
#
# Usage (from Julia REPL in CatherSim/):
#   include("scripts/diagnose_results.jl")
#   diagnose("outputs/results_smooth_quick.h5", "outputs/results_obstacles_quick.h5")
#
# Or from CLI:
#   julia scripts/diagnose_results.jl outputs/results_smooth_quick.h5 outputs/results_obstacles_quick.h5

using HDF5
using Statistics
using Printf
include("../src/Parameters.jl")
using .Parameters

function diagnose_one(h5file::String; nbins=20)
    isfile(h5file) || error("File not found: $h5file")
    out = String[]
    push!(out, "="^60)
    push!(out, "  $h5file")
    push!(out, "="^60)

    h5open(h5file, "r") do file
        x_data = read(file["data/x"])
        img_data = read(file["data/img"])
        _, np, n_samples = size(x_data)
        tiled_Lx = haskey(file, "tiled_Lx") ? read(file["tiled_Lx"]) : Parameters.Lx

        x0_unwrapped = x_data[1, :, 1] .+ img_data[1, :, 1] .* tiled_Lx
        x_final = x_data[1, :, n_samples] .+ img_data[1, :, n_samples] .* tiled_Lx
        x_up = x0_unwrapped .- x_final

        upstream_mask = x_up .> 0
        n_upstream = sum(upstream_mask)

        push!(out, "")
        push!(out, "  np=$np, n_samples=$n_samples, Lx=$(round(tiled_Lx,digits=1)) μm")
        push!(out, "")
        push!(out, "  METRICS (paper Fig 2F):")
        push!(out, "    Upstream count:     $n_upstream / $np ($(round(100*n_upstream/np, digits=2))%)")
        push!(out, "    <x_up> all:         $(round(mean(x_up), digits=2)) μm")
        push!(out, "    <x_up> upstream:    $(n_upstream > 0 ? round(mean(x_up[upstream_mask]), digits=2) : "N/A") μm")
        push!(out, "    x_1% (99th pct):    $(n_upstream > 0 ? round(quantile(x_up, 0.99), digits=2) : "N/A") μm")
        push!(out, "    max(x_up):          $(round(maximum(x_up), digits=2)) μm")
        push!(out, "    min(x_up):          $(round(minimum(x_up), digits=2)) μm")
        push!(out, "")

        # Histogram of x_up (text bars)
        lo, hi = extrema(x_up)
        edges = range(lo, hi, length=nbins+1)
        counts = zeros(Int, nbins)
        edges_vec = collect(edges)
        for v in x_up
            k = clamp(searchsortedfirst(edges_vec, v) - 1, 1, nbins)
            counts[k] += 1
        end
        push!(out, "  x_up HISTOGRAM (μm):")
        max_count = maximum(counts)
        for k in 1:nbins
            bar_len = max_count > 0 ? Int(round(40 * counts[k] / max_count)) : 0
            push!(out, "    [$(rpad(round(edges[k], digits=1), 8)) .. $(rpad(round(edges[k+1], digits=1), 8))]  $(counts[k])  $(repeat("█", bar_len))")
        end
        push!(out, "")

        # Sample of upstream swimmers (first 10)
        up_idx = findall(upstream_mask)
        if length(up_idx) >= 1
            push!(out, "  SAMPLE upstream swimmers (first 10):")
            for (j, i) in enumerate(up_idx[1:min(10, length(up_idx))])
                push!(out, "    p$i: x0=$(round(x0_unwrapped[i], digits=1)) → x=$(round(x_final[i], digits=1))  x_up=$(round(x_up[i], digits=1)) μm")
            end
        end
        push!(out, "")
    end
    return join(out, "\n")
end

function diagnose(smooth_file::String, obstacle_file::String; nbins=20)
    println(diagnose_one(smooth_file, nbins=nbins))
    println(diagnose_one(obstacle_file, nbins=nbins))

    # Comparison summary
    println("="^60)
    println("  PAPER EXPECTATION: obstacles REDUCE upstream penetration")
    println("  (fewer upstream swimmers, lower <x_up>, lower x_1%)")
    println("="^60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 2
        diagnose(ARGS[1], ARGS[2])
    elseif length(ARGS) == 1
        println(diagnose_one(ARGS[1]))
    else
        println("Usage: julia diagnose_results.jl <smooth.h5> [obstacles.h5]")
        println("  Or: diagnose(\"outputs/results_smooth_quick.h5\", \"outputs/results_obstacles_quick.h5\")")
    end
end
