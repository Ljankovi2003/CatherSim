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

#flowdatapath="/home/tingtao/cathetersim/flowdata/"
flowdatapath="/home/tingtaoz/data/storage/catheter/flowdata/"
outpath = "/home/tingtaoz/data/storage/catheter/traindata/"

############ simulation

T = Float64
dim = 2
particleshape = BD.Point(dim) #BD.Sphere(dim, a)
U0 = T(20.0)
DT = T(0.1)
DR = T(0.2)
linear = BD.Brownian(DT)
angular = BD.Brownian(DR)
particle = BD.Particle(particleshape, linear, angular)

Lx = T(100)
Ly = T(20)
# domain = BD.Box(Lx, Ly)
Nv = 6
Nt = Int64(500000)
dt = 1e-4
np = 100000
sample_steps = 1000
Nx = 1_000+1
Ny = 200+1

x10 = x60 = 40-Lx/2
y1 = 0-Ly/2
y3 = 0-Ly/2
y6 = Ly/2
y4 = Ly/2
# the 3 points of the triangle should go around s.t. the inner of 
# triangle is on the right of the curve
# for x2=41:60
for x2=45:2:47
    x50 = x20 = x2 - Lx/2
    # for x3=41:60
    for x3=41:2:60
        x40 = x30 = x3 - Lx/2
        for h=2:0.5:6.5
            y2 = h-Ly/2
            y5 = Ly/2-h
            p1 = (x10, y1)
            p2 = (x20, y2)
            p3 = (x30, y3)
            p4 = (x40, y4)
            p5 = (x50, y5)
            p6 = (x60, y6)
            vertices = CuArray{Float64}(undef, Nv*2)
            # print("defined vertices\n")
            tmpv = [p1...,p2...,p3...,p4...,p5...,p6...]
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

            flowfname = @sprintf("flow_%g_%g_%g.txt",x2,x3,h)
            print("read from ",flowfname)
            flowfile = string(flowdatapath, flowfname)
            flowdata = readdlm(flowfile, skipstart=9)
            tmpU = flowdata[:,3]
            print(size(tmpU))
            tmpV = flowdata[:,4]
            tmpOmega = flowdata[:,5]
            tmpU = reshape(tmpU, (Nx, Ny))
            tmpV = reshape(tmpV, (Nx, Ny))
            tmpOmega = reshape(tmpOmega, (Nx, Ny))
            copyto!(Ux, tmpU)
            copyto!(Uy, tmpV)
            copyto!(Omegatest, tmpOmega)
            mycache = (U=Ux, V=Uy, Omega=Omegatest, xgrid=xgrid,
                ygrid=ygrid, Nx=Nx, Ny=Ny, vertices=vertices)
            # mycache = (U=Ux, V=Uy, Omega=Omegatest, xgrid=xgrid,
            #     ygrid=ygrid, Nx=Nx, Ny=Ny, vertices=())
            uf = maximum(tmpU[1,:])
            model = BD.IdealModel(particle, domain, bc,
                np=np,
                U=U,
                Omega=Omega,
                cache=mycache)

            data = BD.ParticleData(model)

            x0 = Lx/2-1.0
            y0 = 0.0
            BD.initialize!(data,
                # zero(SVector{dim,Float64}),
                SVector(x0, y0),
                0.6*pi
            )
            # BD.default(BD.orientation_type(dim, T)),
            integrator = BD.EulerMaruyama(linear, angular, dt)

            # set kernel batch_size and total time steps
            batch_size = 1
            kernel = BD.BDKernel(model, data, integrator, batch_size)
            simulation = BD.Simulation(model, data, integrator, Nt, kernel)

            # store particle positions
            fprefix = @sprintf("U0%guf%g_%g_%g_%g",U0, uf,x2,x3,h)
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
end
