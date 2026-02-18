using BD
using StaticArrays
using LinearAlgebra
using Statistics
using CUDA
using Printf
using DelimitedFiles
using Random
# using IdealBD
# print("now include cathetermodule\n")

include("cathetermodule-levy-polygon.jl")

flowdatapath = outpath = "/home/tingtaoz/data/storage/catheter/cathetersim/levy/polygon/"

############ simulation

T = Float64
dim = 2
particleshape = BD.Point(dim) #BD.Sphere(dim, a)
U0 = T(20.0)
DT = T(0.1)
DR = T(0.0)
alpha = T(1.5)
tauR = T(2.0)
linear = BD.Brownian(DT)
angular = BD.Brownian(DR)
particle = BD.Particle(particleshape, linear, angular)

Lx = T(100)
Ly = T(20)
# domain = BD.Box(Lx, Ly)
Nv = 6
Nt = Int64(500000)
dt = 1e-4
np = 1000
sample_steps = 1000
Nx = 1_000
Ny = 100


x1 = 40; x3 = 71.5; x5 = 47.4
x2 = 62.6; x4 = 71.4; h=2.616

xs1 = xs10 = x1-Lx/2;
xs2 = xs9 = x2-Lx/2;
xs3 = xs8 = x3-Lx/2;
xs4 = xs7 = x4-Lx/2;
xs5 = xs6 = x5-Lx/2;
y1 = 0-Ly/2; y10 = -y1;
y2 = h/2-Ly/2; y9 = -y2;
y3 = h-Ly/2; y8 = -y3;
y4 = h/2-Ly/2; y7 = -y4;
y5 = 0-Ly/2; y6 = -y5

p1 = (xs1, y1);p2 = (xs2, y2);p3 = (xs3, y3)
p4 = (xs4, y4);p5 = (xs5, y5);p6 = (xs6, y6)
p7 = (xs7, y7);p8 = (xs8, y8);p9 = (xs9, y9);p10=(xs10,y10)
vertices = CuArray{Float64}(undef, Nv*2)
# print("defined vertices\n")
tmpv = [p1...,p3...,p5...,p6...,p8...,p10...]
#tmpv = [p1...,p2...,p3...,p4...,p5...,p6...,p7...,p8...,p9...,p10...]
copyto!(vertices, tmpv)
domain = Irregular(Lx, Ly)
# domain = BD.Box(Lx, Ly)
# print("defined domain\n")
# bc = BD.ChannelBC(domain, particleshape)
bc = IrregularBC(domain, particleshape)

# U = poiseuille_linear_velocity_and_swim(domain, particleshape, uf, U0)
# Omega = poiseuille_angular_velocity(domain, particleshape, uf)
U = linear_velocity_swimonly(domain, particleshape, U0)
Omega = zero_angular_velocity(domain, particleshape)
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

# flowfname = @sprintf("flow_%g_%g_%g.txt",x2,x3,h)
flowfname = "0815.txt"
print("read from ",flowfname)
flowfile = string(flowdatapath, flowfname)
flowdata = readdlm(flowfile, skipstart=9)
tmpU = flowdata[:,3]
tmpU[isnan.(tmpU)].=0.0;
print(size(tmpU))
tmpV = flowdata[:,4]
tmpV[isnan.(tmpV)].=0.0;
tmpOmega = flowdata[:,5]
tmpOmega[isnan.(tmpOmega)].=0.0;
tmpU = reshape(tmpU, (Nx, Ny))
tmpV = reshape(tmpV, (Nx, Ny))
tmpOmega = reshape(tmpOmega, (Nx, Ny))
copyto!(Ux, tmpU)
copyto!(Uy, tmpV)
copyto!(Omegatest, tmpOmega)
mycache = (U=Ux, V=Uy, Omega=Omegatest, xgrid=xgrid,
    ygrid=ygrid, Nx=Nx, Ny=Ny, vertices=vertices, alpha=alpha, tauR=tauR)
# mycache = (U=Ux, V=Uy, Omega=Omegatest, xgrid=xgrid,
#     ygrid=ygrid, Nx=Nx, Ny=Ny, vertices=())
uf = maximum(tmpU[1,:])
model = BD.IdealModel(particle, domain, bc,
    np=np,
    U=U,
    Omega=Omega,
    cache=mycache)

data = BD.ParticleData(model,Float64)

x0 = Lx/2-1.0
y0 = 0.0
q0 = rand(np) .* 2 * pi
BD.initialize!(data,
    # zero(SVector{dim,Float64}),
    SVector(x0, y0),
    q0,
    0.0
)
# BD.default(BD.orientation_type(dim, T)),
integrator = BD.EulerMaruyama(linear, angular, dt)

# set kernel batch_size and total time steps
batch_size = 1
kernel = BD.BDKernel(model, data, integrator, batch_size)
simulation = BD.Simulation(model, data, integrator, Nt, kernel)

# store particle positions
fprefix = @sprintf("U0%guf%galpha%gtauR%g_%g_%g_%g_%g_%g",U0, uf,alpha, tauR,x2,x3,x4,x5,h)
file = string(outpath, fprefix, ".h5")
BD.safecreate_h5file(file)
counter = BD.Counter(0,Nt,sample_steps,Int64(Nt/sample_steps))
positionsaver = BD.PositionWriter(counter,
    file, simulation,(true, false))
eta = BD.ETA(BD.IterationInterval(batch_size * 500), Nt)
callbacks = (eta, positionsaver)
BD.run!(simulation, callbacks=callbacks)
# endoffile
print("finished ",fprefix)
