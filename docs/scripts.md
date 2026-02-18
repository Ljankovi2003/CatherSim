# Driver scripts and jobs

## Top-level scripts
- `channel_interpolate.jl`: Demonstration run with an irregular domain, two triangular obstacles (hard-coded points), no imposed flow (Ux/ Uy/ Omega zero arrays). Uses `linear_velocity_swimonly` so motion comes from self-propulsion + Brownian kicks.
- `channel_interpolate_noobstacle.jl`: Baseline Poiseuille channel without obstacles (`BD.Box` + `ChannelBC`). Initializes one particle near `x=80, y=-10` and writes positions to HDF5 (`smooth_persistent...`).
- `channel_paramsweep.jl`: Sweeps geometry parameters (x2, x3, h) and reads precomputed flow fields from `flowdatapath` (`flow_{x2}_{x3}_{h}.txt`). Saves HDF5 outputs to `outpath`. Configure `np`, `Nt`, and sampling via `sample_steps`.
- `channel_sweep1.jl` .. `channel_sweep5.jl`: Same pipeline as `channel_paramsweep` but partition the x2 search space for batching on a cluster.
  - sweep1: x2 ? {41,43}, sweep2: {45,47}, sweep3: {49,51}, sweep4: {53,55}, sweep5: {57,59}; each scans x3 = 41:2:60 and h = 2:0.5:6.5.
- `channel_test.jl`: Small debug run with a single geometry (x2=41, x3=41, h=2) writing to `outpath` under `debug/`.
- `analysis.jl`: Reads an HDF5 trajectory (expects groups under `config/<frame>/`) and builds a GIF of a chosen particle path. Update `file`, `pid`, and frame counts to match your output.

## Batch job
- `jobcatheter.gpu`: SLURM script requesting one V100 GPU for 10 hours; loads CUDA 11.3 and Julia 1.7.2 before calling `julia --project=.. sweep1.jl`. Duplicate/adjust for other sweeps.

## Subfolders
- `abp/`: Active Brownian particle runs mirroring the top-level sweep scripts; includes SLURM outputs and data under `traindata/`.
- `levy/`: Lévy-flight variant runs that rely on `cathetermodule-levy.jl`; contains multiple sweep batches and result folders.
