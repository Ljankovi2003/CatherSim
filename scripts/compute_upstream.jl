# scripts/compute_upstream.jl
#
# Computes upstream swimming metrics from HDF5 trajectory data.
# This is the key observable from the paper: how far bacteria penetrate
# against the flow, comparing smooth vs. obstacle-decorated channels.
#
# Metrics:
#   - Mean upstream displacement <x_up> (all particles)
#   - Mean upstream displacement (upstream swimmers only)
#   - 99th percentile upstream penetration (x_1%)
#   - Fraction of particles that swam upstream
#
# Usage:
#   include("scripts/compute_upstream.jl")
#   compute_upstream("results_smooth.h5")

using HDF5
using Statistics
using Printf
include("../src/Parameters.jl")
using .Parameters

function compute_upstream(h5file::String)
    isfile(h5file) || error("File not found: $h5file")

    h5open(h5file, "r") do file
        x_data = read(file["data/x"])       # (2, np, n_samples)
        img_data = read(file["data/img"])    # (2, np, n_samples)
        _, np, n_samples = size(x_data)

        tiled_Lx = haskey(file, "tiled_Lx") ? read(file["tiled_Lx"]) : Parameters.Lx

        println("  $h5file: $np particles, $n_samples snapshots, Lx=$(round(tiled_Lx,digits=1)) μm")

        x0_unwrapped = x_data[1, :, 1] .+ img_data[1, :, 1] .* tiled_Lx

        for s in [n_samples]
            x_unwrapped = x_data[1, :, s] .+ img_data[1, :, s] .* tiled_Lx
            x_up = x0_unwrapped .- x_unwrapped

            upstream_mask = x_up .> 0
            n_upstream = sum(upstream_mask)

            mean_all = mean(x_up)
            mean_up  = n_upstream > 0 ? mean(x_up[upstream_mask]) : 0.0
            x1_pct   = n_upstream > 0 ? quantile(x_up, 0.99) : 0.0
            max_up   = maximum(x_up)

            @printf("  Snapshot %d (final):\n", s)
            @printf("    Upstream:  %d / %d (%.1f%%)\n", n_upstream, np, 100.0*n_upstream/np)
            @printf("    <x_up> all:      %8.2f μm\n", mean_all)
            @printf("    <x_up> upstream:  %8.2f μm\n", mean_up)
            @printf("    x_1%% (99th pct): %8.2f μm\n", x1_pct)
            @printf("    max(x_up):       %8.2f μm\n", max_up)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 1 || error("Usage: julia compute_upstream.jl <results.h5>")
    compute_upstream(ARGS[1])
end
