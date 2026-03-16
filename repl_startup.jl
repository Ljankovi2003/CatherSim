# repl_startup.jl - Streamlined for fast iteration
import Pkg
Pkg.activate(".")

# Set CUDA runtime BEFORE anything else if needed
# Pkg.instantiate()
using CUDA
if CUDA.runtime_version() != v"12.8"
    CUDA.set_runtime_version!(v"12.8")
end
using Revise
using WaterLily
using Plots

println("✓ Revise, CUDA ($(CUDA.functional() ? CUDA.name(CUDA.device()) : "CPU")), WaterLily, Plots loaded.")