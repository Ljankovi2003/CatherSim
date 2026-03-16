using .BD
using StaticArrays
using LinearAlgebra
using Statistics
using CUDA

"""
Irregular domain definition.
"""
struct Irregular{N,T} <: BD.AbstractDomain{N}
    coordinates::NTuple{N,Tuple{T,T}}
    widths::NTuple{N,T}
    function Irregular(coordinates::NTuple{N,Tuple{T,T}}) where {N,T<:Real}
        N == 2 || throw(ArgumentError("Only 2D supported."))
        widths = tuple([x[2] - x[1] for x in coordinates]...)
        return new{N,T}(coordinates, widths)
    end
end
Irregular(Lx::Real, Ly::Real) = Irregular(((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

struct IrregularBC{D,S} <: BD.AbstractBC{D,S}
    domain::D
    shape::S
end

@inline function obstacle(sphere::BD.SphereLike, x::Real, y::Real,
    x0::Real, y0::Real, x1::Real, y1::Real, x2::Real, y2::Real)
    
    # Legacy logic: Two segments forming the wedge
    ell0 = SVector(x1-x0, y1-y0)
    n0 = SVector(-y1+y0, x1-x0)
    n0 = n0 / norm(ell0)
    
    ell1 = SVector(x2-x1, y2-y1)
    n1 = SVector(-y2+y1, x2-x1)
    n1 = n1 / norm(ell1)
    
    # Distance behind the segments (positive = inside/behind)
    d0 = -((x-x0)*n0[1] + (y-y0)*n0[2])
    d1 = -((x-x1)*n1[1] + (y-y1)*n1[2])
    
    if (d0 > 0 && d1 > 0)
        if d0 < d1
            return 2 * d0 * n0[1], 2 * d0 * n0[2]
        else
            return 2 * d1 * n1[1], 2 * d1 * n1[2]
        end
    end
    return 0.0, 0.0
end

@inline function (bc::IrregularBC)(bcsolver::BD.SimpleBCSolver, x::SVector{2,T}, q::Real, u, omega, eta, dt, vertices) where {T}
    x1, x2 = bc.domain.coordinates[1]
    y1, y2 = bc.domain.coordinates[2]
    Lx = bc.domain.widths[1]
    
    x_tent = x + u * dt
    q_tent = q + omega * dt # update_orientation
    
    xx, yy = x_tent
    dy = BD.parallel_flat_walls(bc.shape, yy, y1, y2)
    dxwall = SVector{2,T}(0, dy)
    
    dxpolygon = SVector{2,T}(0, 0)
    Nvertices = length(vertices)
    # Each obstacle is 3 vertices = 6 floats
    for itri in 0:Int(floor(Nvertices/6)-1)
        p1x, p1y = vertices[itri*6+1], vertices[itri*6+2]
        p2x, p2y = vertices[itri*6+3], vertices[itri*6+4]
        p3x, p3y = vertices[itri*6+5], vertices[itri*6+6]
        
        tmpdx, tmpdy = obstacle(bc.shape, xx, yy, p1x, p1y, p2x, p2y, p3x, p3y)
        dxpolygon += SVector{2,T}(tmpdx, tmpdy)
    end
    
    dx_per, px = BD.periodicbc(xx, x1, x2, Lx)
    xnew = x_tent + dxwall + dxpolygon + SVector{2,T}(dx_per, 0)
    
    return dxwall, xnew, q_tent, SVector{2,Int}(px, 0)
end

@inline function generate_runtime_levy_gpu(alpha::Float64, tauR::Float64, state::UInt64)
    taum = (alpha - 1.0) * tauR / alpha
    state = BD.xorshift(state)
    U = BD.uniform_rand(state)
    tau = taum / (max(U, 1e-15) ^ (1.0 / alpha))
    return tau, state
end

@inline function random_orientation_gpu(state::UInt64)
    state = BD.xorshift(state)
    return BD.uniform_rand(state) * 2.0 * π, state
end

@inline function findind(Ngrid, grid, x)
    low, high = 1, Ngrid
    while (high > low + 1)
        i = Int(floor((low + high) / 2))
        if (grid[i] <= x) low = i else high = i end
    end
    return low
end

@inline function interpolate_u(xgrid, ygrid, indx, indy, x, y, ugrid)
    x1, x2 = xgrid[indx], xgrid[indx+1]
    y1, y2 = ygrid[indy], ygrid[indy+1]
    f11, f12 = ugrid[indx, indy], ugrid[indx, indy+1]
    f21, f22 = ugrid[indx+1, indy], ugrid[indx+1, indy+1]
    
    dx, dy = x2 - x1, y2 - y1
    t, s = (x - x1) / dx, (y - y1) / dy
    
    return (1-t)*(1-s)*f11 + t*(1-s)*f21 + (1-t)*s*f12 + t*s*f22
end

function BD.bd_kernel(integrator::BD.EulerMaruyama, particle, domain::Irregular{N}, bc, bcsolver, x, q, img, data, rng_states, U_func, Omega_func, t::Real, eta::Real, batch_size::Integer, op, cache) where {N}
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    
    @inbounds for i = idx:stride:length(x)
        xi, qi, state = x[i], q[i], rng_states[i]
        
        # Grid lookups
        indx = findind(cache.Nx, cache.xgrid, xi[1])
        indy = findind(cache.Ny, cache.ygrid, xi[2])
        # Safety clamp
        indx = clamp(indx, 1, cache.Nx-1)
        indy = clamp(indy, 1, cache.Ny-1)
        
        ux = interpolate_u(cache.xgrid, cache.ygrid, indx, indy, xi[1], xi[2], cache.U)
        uy = interpolate_u(cache.xgrid, cache.ygrid, indx, indy, xi[1], xi[2], cache.V)
        wz = interpolate_u(cache.xgrid, cache.ygrid, indx, indy, xi[1], xi[2], cache.Omega)
        
        flow_v = SVector(ux, uy)
        
        # Stochastic velocity contributions
        lin_noise, state = BD.linear_stochastic_velocity(integrator, t, xi, qi, particle.linear, state)
        ang_noise, state = BD.angular_stochastic_velocity(integrator, t, xi, qi, particle.angular, state)
        
        # Combined velocities
        u_total = flow_v + U_func(xi, qi, t) + lin_noise
        omega_total = wz + Omega_func(xi, qi, t) + ang_noise
        
        # Step and BCs
        record, x[i], q[i], dimg = bc(bcsolver, xi, qi, u_total, omega_total, eta, integrator.dt, cache.vertices)
        img[i] += dimg
        
        # Lévy tumble check
        if hasfield(typeof(cache), :alpha)
            if data[i] < t
                dtau, state = generate_runtime_levy_gpu(cache.alpha, cache.tauR, state)
                data[i] = t + dtau
                q[i], state = random_orientation_gpu(state)
            end
        end
        
        rng_states[i] = state
    end
    return nothing
end

@inline function linear_velocity_swimonly(domain::Irregular{2,T}, shape::BD.SphereLike, U0::Real) where {T}
    return (x, q, t) -> U0 * BD.q2c(q)
end

@inline function zero_angular_velocity(domain::Irregular{2,T}, shape::BD.SphereLike) where {T}
    return (x, q, t) -> 0.0
end
