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

flowdatapath = outpath = "/home/tingtao/cathetersim/0925/"
#flowdatapath = "/home/tingtaoz/data/storage/catheter/opt0925/"
#outpath = "/home/tingtaoz/storage/traindatalevy/allparams/0925/"

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


lxarr=[61.24855804,65.82542419]
harr=[29.93624878,25.18229103]
d2arr = [21.72837067,28.63066483]
d3arr = [16.53041077,17.85698509]
d4arr = [-10.21968842,-22.04808807]
d5arr = [-2.816635132,-4.428844452]

for indsample=1:2

Lx = T(lxarr[indsample])
Ly = T(100)
Nv = 10
Nt = Int64(500000)
dt = 1e-4
np = 100_000
sample_steps = 50000
Nxlong = 1_000
Nx = 200
Ny = 100


x1 = 40; 
x2 = x1+d4arr[indsample]; 
x3 = x1+d2arr[indsample]; 
x4 = x1+d5arr[indsample];
x5 = x1+d3arr[indsample]
h = harr[indsample]

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
tmpv = [p1...,p2...,p3...,p4...,p5...,p6...,p7...,p8...,p9...,p10...]
copyto!(vertices, tmpv)
domain = Irregular(Lx, Ly)
bc = IrregularBC(domain, particleshape)

U = linear_velocity_swimonly(domain, particleshape, U0)
Omega = zero_angular_velocity(domain, particleshape)
xgrid = CuArray{Float64}(undef, Nx)
ygrid = CuArray{Float64}(undef, Ny)
xx = collect(LinRange(-Lx / 2.0, Lx / 2.0, Nx))
yy = collect(LinRange(-Ly / 2.0, Ly / 2.0, Ny))
copyto!(xgrid, xx)
copyto!(ygrid, yy)
Ux = CUDA.zeros(Float64,Nx,Ny)
Uy = CUDA.zeros(Float64,Nx, Ny)
Omegatest = CUDA.zeros(Float64,Nx, Ny)

flowfname = @sprintf("0925-5gon-%d.txt",indsample)
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

tmpU = reshape(tmpU, (Nxlong, Ny))
tmpV = reshape(tmpV, (Nxlong, Ny))
tmpOmega = reshape(tmpOmega, (Nxlong, Ny))
uftmp = maximum(tmpU[1,:])

for uf=5:5:15
    rescale_factor = uf/uftmp
    copyto!(Ux, rescale_factor*tmpU[401:600,:])
    copyto!(Uy, rescale_factor*tmpV[401:600,:])
    copyto!(Omegatest, rescale_factor*tmpOmega[401:600,:])
mycache = (U=Ux, V=Uy, Omega=Omegatest, xgrid=xgrid,
    ygrid=ygrid, Nx=Nx, Ny=Ny, vertices=vertices, alpha=alpha, tauR=tauR)
model = BD.IdealModel(particle, domain, bc,
    np=np,
    U=U,
    Omega=Omega,
    cache=mycache)

data = BD.ParticleData(model,Float64)

x0 = Lx/2
y0 = 0.0
q0 = rand(np) .* 2 * pi
BD.initialize!(data,
    SVector(x0, y0),
    q0,
    0.0
)
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

end
end