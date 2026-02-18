# Repository structure (quick map)

- `Project.toml`, `Manifest.toml`: Julia environment pinning BD.jl, CUDA.jl, HDF5, Plots, etc.
- `cathetermodule.jl`: core solver extensions.
- `cathetermodule-levy.jl`: solver + Lévy reorientation.
- `channel_*.jl`: simulation entry points (see `docs/scripts.md`).
- `analysis.jl`: trajectory plotting/GIF exporter.
- `jobcatheter.gpu`: SLURM GPU job template.
- `abp/`: active-Brownian sweeps + logs/data.
- `levy/`: Lévy sweeps + logs/data; contains multiple dated experiment folders.
- `docs/`: this documentation set.
