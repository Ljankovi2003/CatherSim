#!/bin/bash
# start_repl.sh

# Move to the project directory
cd /home/rwhan/CatherSim

# Run julia using the project environment and the startup script
# -L loads a file before starting the REPL
JULIA_NUM_THREADS=auto
/central/software9/external/julia/1.12.2/bin/julia --project -i -L repl_startup.jl
