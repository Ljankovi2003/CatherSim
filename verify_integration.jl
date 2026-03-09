using BD
using StaticArrays
using LinearAlgebra
using Statistics
using CUDA
using Printf
using DelimitedFiles
using Plots

# Include the core module
include("cathetermodule.jl")

function run_integrated_test(flowfile="flow_41_41_2.txt")
    println("Loading flow data from $flowfile...")
    Lx, Ly = 100.0, 20.0
    Nx, Ny = 1024, 256
    
    flowdata = readdlm(flowfile, skipstart=9)
    tmpU = reshape(flowdata[:,3], (Nx, Ny))
    tmpV = reshape(flowdata[:,4], (Nx, Ny))
    tmpOmega = reshape(flowdata[:,5], (Nx, Ny))
    
    # Environment Setup
    xgrid = CuArray(collect(LinRange(-Lx/2, Lx/2, Nx)))
    ygrid = CuArray(collect(LinRange(-Ly/2, Ly/2, Ny)))
    Ux = CuArray(tmpU)
    Uy = CuArray(tmpV)
    Omegatest = CuArray(tmpOmega)
    
    # Notch vertices (matching flow generation)
    # v1...v6 from generate_flow.jl
    # For IrregularBC in cathetermodule.jl
    # Nvertices = 6. 
    # vertices = [p1...,p2...,p3...,p4...,p5...,p6...]
    # Let's simplify and just use the box for now or replicate the vertices exactly
    x10 = 40.0 - Lx/2.0; x2 = 41.0; x3 = 41.0; h = 2.0
    x20 = x2 - Lx/2.0; x30 = x3 - Lx/2.0; x60 = x10; x50 = x20; x40 = x30
    y1 = -Ly/2.0; y2 = h - Ly/2.0; y3 = -Ly/2.0
    y4 = Ly/2.0; y5 = Ly/2.0 - h; y6 = Ly/2.0
    tmpv = [x10, y1, x20, y2, x30, y3, x40, y4, x50, y5, x60, y6]
    vertices = CuArray{Float64}(tmpv)

    # Particle setup
    U0 = 20.0; DT = 0.1; DR = 0.2
    particleshape = BD.Point(2)
    linear = BD.Brownian(DT); angular = BD.Brownian(DR)
    particle = BD.Particle(particleshape, linear, angular)
    
    domain = Irregular(Lx, Ly)
    bc = IrregularBC(domain, particleshape)
    
    mycache = (U=Ux, V=Uy, Omega=Omegatest, xgrid=xgrid, ygrid=ygrid, Nx=Nx, Ny=Ny, vertices=vertices)
    
    model = BD.IdealModel(particle, domain, bc, np=1000, 
                         U=linear_velocity_swimonly(domain, particleshape, U0),
                         Omega=zero_angular_velocity(domain, particleshape),
                         cache=mycache)
    
    data = BD.ParticleData(model)
    BD.initialize!(data, SVector(Lx/2-2.0, 0.0), 1.0*pi) # Start near right exit, swim left
    
    dt = 1e-4; Nt = 10000; sample_steps = 100
    integrator = BD.EulerMaruyama(linear, angular, dt)
    kernel = BD.BDKernel(model, data, integrator, 1)
    simulation = BD.Simulation(model, data, integrator, Nt, kernel)
    
    println("Running integration test for $Nt steps...")
    # Track positions manually for plotting
    trajectories = [Vector{SVector{2,Float64}}() for _ in 1:10] # Track first 10 particles
    
    BD.run!(simulation, callbacks=(BD.ETA(BD.IterationInterval(1000), Nt),))
    
    # Final positions
    pos = Array(data.x)
    p1 = scatter(pos[1,:], pos[2,:], aspect_ratio=:equal, xlims=(-Lx/2, Lx/2), ylims=(-Ly/2, Ly/2),
                title="Integrated Particle Positions (t=$(Nt*dt))", label="Particles")
    
    # Overlay flow field (vorticity heatmap)
    p2 = heatmap(collect(LinRange(-Lx/2, Lx/2, Nx)), collect(LinRange(-Ly/2, Ly/2, Ny)), tmpOmega', 
                 aspect_ratio=:equal, alpha=0.5, color=:balance, label="Vorticity")
    scatter!(p2, pos[1,1:100], pos[2,1:100], label="Particles (first 100)", markersize=2)
    
    savefig(p2, "integration_proof.png")
    println("Saved integration proof to integration_proof.png")
end

if isfile("flow_41_41_2.txt")
    run_integrated_test()
else
    println("flow_41_41_2.txt not found. Please run CFD first.")
end
