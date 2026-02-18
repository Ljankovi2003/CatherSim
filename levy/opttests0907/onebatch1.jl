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

include("cathetermodule-levy.jl")

flowdatapath="/home/tingtaoz/data/storage/catheter/optimizedtests0906/"
outpath = "/home/tingtaoz/storage/traindatalevy/optimizedtests0906/"

#outpath = "/home/tingtaoz/data/storage/catheter/cathetersim/levy/traindata/"

############ simulation

T = Float64
dim = 2
particleshape = BD.Point(dim) #BD.Sphere(dim, a)
U0 = T(20.0)
DT = T(0.1)
DR = T(0.0)
tauR = T(2.0)
linear = BD.Brownian(DT)
angular = BD.Brownian(DR)
particle = BD.Particle(particleshape, linear, angular)

# Lx = T(100)
Ly = T(100)
# domain = BD.Box(Lx, Ly)
Nv = 6
Nt = Int64(5000000)
dt = 1e-4
np = 100000
sample_steps = 50000
Nxlong = 1_000
Nx = 200
Ny = 100

x10 = x60 = 0
y1 = 0-Ly/2
y3 = 0-Ly/2
y6 = Ly/2
y4 = Ly/2
# the 3 points of the triangle should go around s.t. the inner of 
# triangle is on the right of the curve

for x2 = [5]
    ell = 60
    Lx = ell
    
    x3 = 15
    h = 30
    x50 = x20 = x2
    x40 = x30 = x3
    for flowfactor = 0.33333:0.33333:1
        for alpha = [1.5]
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

            flowfname = @sprintf("x2_%d.txt",x2)
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
            copyto!(Ux, flowfactor*tmpU[401:600,:])
            copyto!(Uy, flowfactor*tmpV[401:600,:])
            copyto!(Omegatest, flowfactor*tmpOmega[401:600,:])
            mycache = (U=Ux, V=Uy, Omega=Omegatest, xgrid=xgrid,
                ygrid=ygrid, Nx=Nx, Ny=Ny, vertices=vertices, alpha=alpha, tauR=tauR)
            # mycache = (U=Ux, V=Uy, Omega=Omegatest, xgrid=xgrid,
            #     ygrid=ygrid, Nx=Nx, Ny=Ny, vertices=())
            uf = maximum(tmpU[1,:])*flowfactor
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
            fprefix = @sprintf("x2_%d_U0%guf%galpha%gtauR%g",x2,U0, uf,alpha, tauR)
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

