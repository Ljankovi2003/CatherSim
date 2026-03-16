module BD

using CUDA
using StaticArrays
using HDF5
using Printf

# Abstract Types
abstract type AbstractDomain{N} end
abstract type AbstractBC{D,S} end
abstract type AbstractShape{N} end
abstract type SphereLike{N,T} <: AbstractShape{N} end

# Geometry Types
struct Point{N,T} <: SphereLike{N,T} end
Point(dim::Int) = Point{dim,Float64}()

struct Box{N,T} <: AbstractDomain{N}
    widths::NTuple{N,T}
end
Box(Lx::T, Ly::T) where {T} = Box{2,T}((Lx, Ly))

# Stochastic Types
struct Brownian{T}
    D::T
end

struct Levy{T}
    alpha::T
    tauR::T
end

# Simple RNG for GPU kernels
@inline function xorshift(state::UInt64)
    state ⊻= state << 13
    state ⊻= state >> 7
    state ⊻= state << 17
    return state
end
@inline function uniform_rand(state::UInt64)
    # Standard UInt64 to Float64 in [0, 1)
    return (state >> 11) * 1.1102230246251565e-16 # 1/2^53
end

# Particle Type
struct Particle{S<:AbstractShape, L, A}
    shape::S
    linear::L
    angular::A
end

# Model & Data Types
struct IdealModel{P, D, B, F1, F2, C}
    particle::P
    domain::D
    bc::B
    np::Int
    U::F1
    Omega::F2
    cache::C
end

function IdealModel(particle::P, domain::D, bc::B; np::Int, U::F1, Omega::F2, cache::C) where {P, D, B, F1, F2, C}
    return IdealModel{P, D, B, F1, F2, C}(particle, domain, bc, np, U, Omega, cache)
end

struct ParticleData{T}
    x::CuVector{SVector{2,T}}
    q::CuVector{T}
    img::CuVector{SVector{2,Int}}
    data::CuVector{Float64} # time of next tumble or extra data
    rng_states::CuVector{UInt64} # per-particle RNG states
end
function ParticleData(model::IdealModel)
    np = model.np
    T = Float64
    x = CUDA.zeros(SVector{2,T}, np)
    q = CUDA.zeros(T, np)
    img = CUDA.zeros(SVector{2,Int}, np)
    data = CUDA.zeros(Float64, np)
    rng_states = CuArray{UInt64}(collect(1:UInt64(np)) .+ UInt64(123456789)) # Seeded
    ParticleData(x, q, img, data, rng_states)
end

function initialize!(data::ParticleData, x0::SVector{2,T}, q0::T) where {T}
    np = length(data.x)
    x_init = CUDA.fill(x0, np)
    q_init = CUDA.fill(q0, np)
    copyto!(data.x, x_init)
    copyto!(data.q, q_init)
    # Initialize tumble times to 0 so they tumble immediately
    fill!(data.data, 0.0)
end

# Integrator Type
struct EulerMaruyama{L, A, T}
    linear::L
    angular::A
    dt::T
end

# Kernel Wrapper
struct BDKernel{M, D, I}
    model::M
    data::D
    integrator::I
    batch_size::Int
end

# Simulation Wrapper
struct Simulation{M, D, I, K}
    model::M
    data::D
    integrator::I
    Nt::Int
    kernel::K
end

# Boundary Condition Solver
struct SimpleBCSolver end

# Callbacks
struct IterationInterval
    interval::Int
end

struct ETA
    interval::IterationInterval
    Nt::Int
end

struct Counter
    current::Int
    total::Int
    sample_steps::Int
    expected_samples::Int
end

struct PositionWriter{S}
    counter::Counter
    filename::String
    simulation::S
    flags::Tuple{Bool, Bool}
end

# Utilities
@inline q2c(q::Real) = SVector(cos(q), sin(q))
@inline update_orientation(q::Real, omega::Real, dt::Real) = q + omega * dt

@inline function parallel_flat_walls(shape::SphereLike, yy::Real, y1::Real, y2::Real)
    # Simple reflection dummy, assumes point particle
    if yy < y1
        return 2*(y1 - yy)
    elseif yy > y2
        return 2*(y2 - yy)
    end
    return 0.0
end

@inline function periodicbc(xx::Real, x1::Real, x2::Real, Lx::Real)
    dx = 0.0
    px = 0
    if xx > x2
        dx = -Lx
        px = 1
    elseif xx < x1
        dx = Lx
        px = -1
    end
    return dx, px
end

@inline function box_muller(state::UInt64)
    # Returns two independent N(0,1) variables
    u1 = uniform_rand(state)
    state = xorshift(state)
    u2 = uniform_rand(state)
    state = xorshift(state)
    r = sqrt(-2.0 * log(max(u1, 1e-10)))
    theta = 2.0 * pi * u2
    return r * cos(theta), r * sin(theta), state
end

@inline function linear_stochastic_velocity(integrator, t, x, q, linear, state::UInt64)
    T = typeof(integrator.dt)
    v1, v2, next_state = box_muller(state)
    return SVector{2,T}(v1, v2) * sqrt(2 * linear.D / integrator.dt), next_state
end

@inline function angular_stochastic_velocity(integrator, t, x, q, angular, state::UInt64)
    T = typeof(integrator.dt)
    v1, v2, next_state = box_muller(state)
    return v1 * sqrt(2 * angular.D / integrator.dt), next_state
end

# Dummy operation for record
@inline default_op(data, i, record, batch_id, batch_size) = nothing

function safecreate_h5file(filename::String)
    h5open(filename, "w") do file
        # create valid empty hdf5
        g = create_group(file, "data")
    end
end

# Functions to be replaced/extended in user code
function bd_kernel end

# Runtime
function run!(sim::Simulation; callbacks=())
    model = sim.model
    data = sim.data
    integ = sim.integrator
    np = model.np
    
    threads = 256
    blocks = ceil(Int, np / threads)
    bcsolver = SimpleBCSolver()
    
    # Extract callbacks
    eta_cb = nothing
    pos_writer = nothing
    for cb in callbacks
        if cb isa ETA
            eta_cb = cb
        elseif cb isa PositionWriter
            pos_writer = cb
        end
    end
    
    op = default_op
    eta_val = 1.0 # arbitrary
    
    println("Running Simulation: Nt = $(sim.Nt), np = $(np)")
    start_time = time()
    t = 0.0

    # Initialize HDF5 if pos_writer is present
    if pos_writer !== nothing
        # Calculate exactly how many samples will be written
        n_samples = sim.Nt ÷ pos_writer.counter.sample_steps
        h5open(pos_writer.filename, "r+") do file
            if haskey(file, "data")
                g = file["data"]
                # 3D datasets: [dim, np, n_samples]
                # x: position [2, np, n_samples]
                # q: orientation [1, np, n_samples]
                # img: periodic image counters [2, np, n_samples]
                create_dataset(g, "x", datatype(Float64), dataspace(2, np, n_samples))
                create_dataset(g, "q", datatype(Float64), dataspace(1, np, n_samples))
                create_dataset(g, "img", datatype(Int), dataspace(2, np, n_samples))
            end
        end
    end

    sample_index = 1
    # Main loop (simplified, batch_size = 1 for now)
    for step in 1:sim.Nt
        if typeof(data.x) <: Array
            # CPU fallback (omitted for brevity in this update, focusing on GPU)
            error("CPU fallback not updated for RNG states yet.")
        else
            @cuda threads=threads blocks=blocks bd_kernel(
                integ, model.particle, model.domain, model.bc, bcsolver,
                data.x, data.q, data.img, data.data, data.rng_states,
                model.U, model.Omega, t, eta_val, 1, op, model.cache
            )
        end
        t += integ.dt

        if eta_cb !== nothing && step % eta_cb.interval.interval == 0
            # Print ETA / info
            elapsed = time() - start_time
            rate = step / elapsed
            rem_steps = sim.Nt - step
            eta_sec = rem_steps / rate
            @printf("Step %d / %d | Rate: %.2f steps/s | ETA: %.2f s\n", step, sim.Nt, rate, eta_sec)
        end
        
        if pos_writer !== nothing && step % pos_writer.counter.sample_steps == 0
            # Simple HDF5 dump of positions and image counters
            x_cpu = Array(data.x)
            q_cpu = Array(data.q)
            img_cpu = Array(data.img)
            h5open(pos_writer.filename, "r+") do file
                dset_x = file["data/x"]
                dset_q = file["data/q"]
                dset_img = file["data/img"]
                
                # Reshape SVectors to Matrices for HDF5 storage
                x_mat = zeros(Float64, 2, np)
                img_mat = zeros(Int, 2, np)
                for i in 1:np
                    x_mat[1, i] = x_cpu[i][1]
                    x_mat[2, i] = x_cpu[i][2]
                    img_mat[1, i] = img_cpu[i][1]
                    img_mat[2, i] = img_cpu[i][2]
                end
                
                dset_x[:, :, sample_index] = x_mat
                dset_q[1, :, sample_index] = q_cpu
                dset_img[:, :, sample_index] = img_mat
            end
            sample_index += 1
        end
    end
    println("Simulation finished in ", round(time() - start_time, digits=2), " seconds.")
end

end # module BD
