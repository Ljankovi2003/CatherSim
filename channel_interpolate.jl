using BD
using StaticArrays
using LinearAlgebra
using Statistics
using CUDA
using Printf
# using IdealBD

include("cathetermodule.jl")

path = "/home/tingtao/cathetersim/"


############ simulation

T = Float64
dim = 2
particleshape = BD.Point(dim) #BD.Sphere(dim, a)
U0 = T(5.0)
DT = T(1)
DR = T(1)
linear = BD.Brownian(DT)
angular = BD.Brownian(DR)
particle = BD.Particle(particleshape, linear, angular)

Lx = T(1)
Ly = T(1)
# domain = BD.Box(Lx, Ly)
Nv = 6
# the 3 points of the triangle should go around s.t. the inner of 
# triangle is on the right of the curve
# flow test1
p1 = (-0.1*Lx, -0.5*Ly)
p2 = (-0.02*Lx, -0.25*Ly)
p3 = (-0.05*Lx, -0.5*Ly)
p4 = (-0.05*Lx, 0.5*Ly)
p5 = (-0.02*Lx, 0.25*Ly)
p6 = (-0.1*Lx, 0.5*Ly)
# # flow test 2
# p1 = (-0.1*Lx, -0.5*Ly)
# p2 = (-0.05*Lx, -0.25*Ly)
# p3 = (-0.02*Lx, -0.5*Ly)
# p4 = (-0.1*Lx, 0.5*Ly)
# p5 = (-0.05*Lx, 0.25*Ly)
# p6 = (-0.02*Lx, 0.5*Ly)
vertices = CuArray{Float64}(undef, Nv*2)
tmpv = [p1...,p2...,p3...,p4...,p5...,p6...]
copyto!(vertices, tmpv)
domain = Irregular(Lx, Ly)
# domain = BD.Box(Lx, Ly)

# bc = BD.ChannelBC(domain, particleshape)
bc = IrregularBC(domain, particleshape)

np = 100

uf = 0.1
# U = poiseuille_linear_velocity_and_swim(domain, particleshape, uf, U0)
U = linear_velocity_swimonly(domain, particleshape, U0)
# Omega = poiseuille_angular_velocity(domain, particleshape, uf)
Omega = zero_angular_velocity(domain, particleshape)

Nx = 1_000
Ny = 1_000
xgrid = CuArray{Float64}(undef, Nx)
ygrid = CuArray{Float64}(undef, Ny)
xx = collect(LinRange(-Lx / 2.0, Lx / 2.0, Nx))
yy = collect(LinRange(-Ly / 2.0, Ly / 2.0, Ny))
copyto!(xgrid, xx)
copyto!(ygrid, yy)
#Utest = CuArray{Float64}(undef, 1_000,1_000)
Ux = CUDA.zeros(Float64,Nx,Ny)
Uy = CUDA.zeros(Float64,Nx, Ny)
Omegatest = CUDA.zeros(Float64,Nx, Ny)
mycache = (U=Ux, V=Uy, Omega=Omegatest, xgrid=xgrid,
    ygrid=ygrid, Nx=Nx, Ny=Ny, vertices=vertices)

model = BD.IdealModel(particle, domain, bc,
    np=np,
    U=U,
    Omega=Omega,
    cache=mycache)

data = BD.ParticleData(model)

BD.initialize!(data,
    zero(SVector{dim,Float64}),
    pi / 8
)
# BD.default(BD.orientation_type(dim, T)),

dt = 1e-4
integrator = BD.EulerMaruyama(linear, angular, dt)

# set kernel batch_size and total time steps
batch_size = 1
Nt = 20_000

kernel = BD.BDKernel(model, data, integrator, batch_size)

simulation = BD.Simulation(model, data, integrator, Nt, kernel)

# store particle positions
file = @sprintf("irregularU0%guf%g.h5",U0, uf)
BD.safecreate_h5file(file)
counter = BD.Counter(0,Nt,100,2000)
positionsaver = BD.PositionWriter(counter,
    file, simulation,(false, false))

eta = BD.ETA(BD.IterationInterval(batch_size * 500), Nt)
#x = BD.MSDCalculator(BD.IterationInterval(batch_size), simulation, 1, true)
callbacks = (eta, positionsaver)

BD.run!(simulation, callbacks=callbacks)
