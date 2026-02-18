# print("now import modules...\n")
using BD
# print("imported BD\n")
using StaticArrays
using LinearAlgebra
using Statistics
using CUDA
using Printf
using DelimitedFiles
# using IdealBD
# print("now include cathetermodule\n")
include("cathetermodule.jl")

path = "/home/tingtao/cathetersim/"

############ simulation

T = Float64
dim = 2
particleshape = BD.Point(dim) #BD.Sphere(dim, a)
U0 = T(20.0)
DT = T(0.001)
DR = T(0.001)
linear = BD.Brownian(DT)
angular = BD.Brownian(DR)
particle = BD.Particle(particleshape, linear, angular)

Lx = T(200)
Ly = T(40)

domain = BD.Box(Lx, Ly)

bc = BD.ChannelBC(domain, particleshape)

np = 10

uf = 18.0
U = poiseuille_linear_velocity_and_swim(domain, particleshape, uf, U0)
Omega = poiseuille_angular_velocity(domain, particleshape, uf)

model = BD.IdealModel(particle, domain, bc,
    np=np,
    U=U,
    Omega=Omega,)

data = BD.ParticleData(model)

x0 = 80.0
y0 = -10.0
BD.initialize!(data,
    # zero(SVector{dim,Float64}),
    SVector(x0, y0),
    0.5*pi
)
# BD.default(BD.orientation_type(dim, T)),

dt = 1e-4
integrator = BD.EulerMaruyama(linear, angular, dt)

# set kernel batch_size and total time steps
batch_size = 1
Nt = Int64(500000)

kernel = BD.BDKernel(model, data, integrator, batch_size)

simulation = BD.Simulation(model, data, integrator, Nt, kernel)

# store particle positions
# file = @sprintf("irregularU0%guf%g.h5",U0, uf)

fprefix = @sprintf("smooth_persistent_U0%guf%g0.5pi",U0, uf)
file = string(path, fprefix, ".h5")

BD.safecreate_h5file(file)
counter = BD.Counter(0,Nt,1000,500)
positionsaver = BD.PositionWriter(counter,
    file, simulation,(true, false))

eta = BD.ETA(BD.IterationInterval(batch_size * 500), Nt)
#x = BD.MSDCalculator(BD.IterationInterval(batch_size), simulation, 1, true)
callbacks = (eta, positionsaver)

BD.run!(simulation, callbacks=callbacks)


# endoffile
print("finished")