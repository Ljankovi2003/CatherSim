# reproduce_paper.jl
#
# Reproduces the core result of Zhou et al., Science Advances 10(1) (2024):
# upstream bacterial swimming is suppressed by optimized triangular obstacles.
#
# Pipeline:
#   1. CFD — Solve Stokes flow on one periodic unit cell (d × W) via WaterLily,
#      then tile to fill the BD domain (Lx). Two cases: smooth (h=0) and
#      obstacles (h=h_paper).
#   2. BD  — Run Lévy run-and-tumble particles in the flow+obstacle field.
#      Particles start at the downstream end; we measure how far they swim
#      upstream after T seconds.
#   3. Analysis — Compare upstream penetration: smooth vs. obstacles.
#      Paper's key finding: obstacles reduce upstream penetration by >50%.
#
# Usage (from Julia REPL in CatherSim/):
#   include("reproduce_paper.jl")
#   quick()                          # fast iteration (~minutes)
#   reproduce()                      # full reproduction (~hours)

using Revise

include("scripts/generate_flow.jl")
include("scripts/simulate_catheter.jl")
include("scripts/compute_upstream.jl")

# ─────────────────────────────────────────────────────────────────────────────
# Quick mode: low resolution CFD (Re=10, 64×128), fewer particles, shorter time.
# Useful for verifying the pipeline works end-to-end in a few minutes.
# ─────────────────────────────────────────────────────────────────────────────
function quick(; np=5000, T=50.0, regen_flow=false, make_movie=false, use_stokes=false)
    println("="^70)
    println("  QUICK REPRODUCTION (np=$np, T=$(T)s, quick_test CFD$(use_stokes ? ", Stokes" : ""))")
    println("="^70)

    _run_pipeline(quick_test=true, np=np, T=T, regen_flow=regen_flow,
                  make_movie=make_movie, use_stokes=use_stokes, tag="quick")
end

# ─────────────────────────────────────────────────────────────────────────────
# Full reproduction: high resolution CFD (Re=10, 256×512), production particles.
# Matches the paper's simulation parameters as closely as possible.
# ─────────────────────────────────────────────────────────────────────────────
function reproduce(; np=Parameters.np_default, T=Parameters.T_end_default,
                     regen_flow=false, make_movie=false, use_stokes=false)
    println("="^70)
    println("  FULL REPRODUCTION (np=$np, T=$(T)s, production CFD$(use_stokes ? ", Stokes" : ""))")
    println("="^70)

    _run_pipeline(quick_test=false, np=np, T=T, regen_flow=regen_flow,
                  make_movie=make_movie, use_stokes=use_stokes, tag="full")
end

# ─────────────────────────────────────────────────────────────────────────────
# Shared pipeline
# ─────────────────────────────────────────────────────────────────────────────
function _run_pipeline(; quick_test, np, T, regen_flow, make_movie, use_stokes, tag)
    out = Parameters.output_dir
    mkpath(out)

    flow_smooth = joinpath(out, "flow_smooth_$(tag).txt")
    flow_obs    = joinpath(out, "flow_obstacles_$(tag).txt")
    res_smooth  = joinpath(out, "results_smooth_$(tag).h5")
    res_obs     = joinpath(out, "results_obstacles_$(tag).h5")
    snap_smooth = joinpath(out, "snapshot_smooth_$(tag).png")
    snap_obs    = joinpath(out, "snapshot_obstacles_$(tag).png")

    # ── Step 1: CFD ──────────────────────────────────────────────────────
    println("\n>>> Step 1: Flow Fields")
    if regen_flow || !isfile(flow_smooth)
        println("  [SMOOTH] h=0 ...")
        run_unit_cell(h=0.0, quick_test=quick_test, use_stokes=use_stokes, out_file=flow_smooth)
    else
        println("  [SMOOTH] reusing $flow_smooth")
    end
    if regen_flow || !isfile(flow_obs)
        println("  [OBSTACLES] h=$(Parameters.h_paper) ...")
        run_unit_cell(h=Parameters.h_paper, quick_test=quick_test, use_stokes=use_stokes, out_file=flow_obs)
    else
        println("  [OBSTACLES] reusing $flow_obs")
    end

    # ── Step 2: BD Simulation ────────────────────────────────────────────
    println("\n>>> Step 2: Particle Simulations (np=$np, T=$(T)s)")
    println("  [SMOOTH] ...")
    run_catheter_simulation(flowfile=flow_smooth, out_file=res_smooth,
                            snapshot_file=snap_smooth, np=np, T_end=T, h=0.0)
    println("  [OBSTACLES] ...")
    run_catheter_simulation(flowfile=flow_obs, out_file=res_obs,
                            snapshot_file=snap_obs, np=np, T_end=T, h=Parameters.h_paper)

    # ── Step 3: Analysis ─────────────────────────────────────────────────
    println("\n>>> Step 3: Upstream Penetration Comparison")
    println("\n  ── SMOOTH CHANNEL ──")
    compute_upstream(res_smooth)
    println("\n  ── OBSTACLE CHANNEL ──")
    compute_upstream(res_obs)

    # ── Step 4 (optional): Movie ─────────────────────────────────────────
    if make_movie
        include("scripts/render_movie.jl")
        println("\n>>> Step 4: Rendering Movies")
        Base.invokelatest(render_movie, res_smooth, flow_smooth;
                          out_movie=joinpath(out, "movie_smooth_$(tag).gif"))
        Base.invokelatest(render_movie, res_obs, flow_obs;
                          out_movie=joinpath(out, "movie_obstacles_$(tag).gif"))
    end

    println("\n" * "="^70)
    println("  DONE. Outputs in $out/")
    println("  Compare: $snap_smooth vs $snap_obs")
    println("  Paper expects: obstacles reduce upstream penetration by >50%.")
    println("="^70)
end

println("Reproduction script loaded.")
println("  quick()      — fast pipeline (~minutes, 5k particles, 50s)")
println("  reproduce()  — full pipeline (~hours, 20k particles, 200s)")
println("  Options: regen_flow=true, make_movie=true, use_stokes=true, np=..., T=...")
