using BD
using StaticArrays
using LinearAlgebra
using Statistics
using CUDA

"""
still a rectangular box, but to notify the boundary is rough later
also store the info of triangles here
every 3 vertices makes a triangle
"""
struct Irregular{N,T} <: BD.AbstractDomain{N}
    # coordinates in each Cartesian direction.
    coordinates::NTuple{N,Tuple{T,T}}
    widths::NTuple{N,T}
    function Irregular(coordinates::NTuple{N,Tuple{T,T}}, 
        ) where {N} where {T<:Real}
        N in (2, ) || throw(ArgumentError("invalid dimension. irregular only support 2D right now."))
        # M%3 == 0 || throw(ArgumentError("#vertices should be multiples of 3 for irregular"))
        widths_vector = [x[2] - x[1] for x in coordinates]
        widths = tuple(widths_vector...)
        return new{N,T}(coordinates, widths)
    end
end

function Base.show(io::IO, box::Irregular)
    print(io, typeof(box))
    print(io, ": ")
    print(io, box.coordinates)
    # print(io, box.vertices)
end


"""
    Irregular(Lx::Real, Ly::Real)
Create a rectangular simulation box in 2D centered at the origin.
"""
function Irregular(Lx::Real, Ly::Real, 
    )
    # centered at zero
    coordinates = ((-Lx / 2.0, Lx / 2.0), (-Ly / 2.0, Ly / 2.0))
    return Irregular(coordinates)
end

# ### irregular domain defined
# ### now define bc for irregular domain
# ### IrregularBC 
struct IrregularBC{D,S} <: BD.AbstractBC{D,S}
    domain::D
    shape::S
    function IrregularBC(domain::Irregular{N}, shape::BD.AbstractShape{N}) where {N}
        return new{typeof(domain),typeof(shape)}(domain, shape)
    end
end

@inline function obstacle(sphere::BD.SphereLike, x::Real, y::Real,
    xx::AbstractVector{<:Real},yy::AbstractVector{<:Real})
    # the line segment is signed
    # with normal vector n = ez x (x1-x0, y1-y0, 0)
    # the aux vectors:
    npoints = length(xx)
    shiftoff = [npoints,collect(1:npoints-1)...]
    deltax = xx[shiftoff].-xx
    deltay = yy[shiftoff].-yy
    ell = SMatrix{npoints,2,Float64}(deltax...,deltay...)
    ell02 = sum(ell.*ell,dims=2)
    ell0norm = sqrt.(ell02)
    n0 = SMatrix{npoints,2,Float64}((-deltay./ell0norm)...,(deltax./ell0norm)...)
    dd = SVector((x .- xx).*n0[:,1] .+ (y .- yy).*n0[:,2])
    dx = dy = 0.0
    if (all(dd.>0))
        dmin, indmin = findmin(dd)
        dx = 2*dmin*nn[indmin,1]
        dy = 2*dmin*nn[indmin,2]
    end
    return dx, dy
end

@inline function (bc::IrregularBC{Irregular{2,S1},<:BD.SphereLike{2,S2}})(bcsolver::BD.SimpleBCSolver,
    x::SVector{2,T},
    q::Real,
    u,
    omega,
    eta,
    dt, vertices) where {S1<:Real,S2<:Real,T<:Real}
    # 2D channel, `y` is the wall
    # extract box coordinates
    x1, x2 = bc.domain.coordinates[1]
    y1, y2 = bc.domain.coordinates[2]
    Lx = bc.domain.widths[1]
    # compute tentative position and orientation
    x̃ = x + u * dt
    q̃ = BD.update_orientation(q, omega, dt)

    # extract scalar locations from vector
    xx, yy = x̃

    # no-flux in y
    dy = BD.parallel_flat_walls(bc.shape, yy, y1, y2)
    # wall displacement
    dxwall = SVector{2,T}(0, dy)
    # polygon displacement
    dxpolygon = SVector{2,T}(0, 0)
    # vertices = bc.domain.vertices
    Nvertices1side = length(vertices)/2 # two sides
    # for itri in 0:Int64(floor(Nvertices/6)-1)
    Npoly1side = (Nvertices1side-1)/2
    xv = yv = SVector{4,T}(0,0,0,0)
    for ipoly in 1:Npoly1side-1
        pind1 = ipoly-1; pind2 = pind1+1;
        pind3 = Nvertices1side-ipoly-1; pind4 = pind3+1;
        for iside in 0:1
            offset = iside * Nvertices1side
            xv[1] = vertices[offset+pind1*2+1]; yv[1] = vertices[offset+pind1*2+2]
            xv[2] = vertices[offset+pind2*2+1]; yv[2] = vertices[offset+pind2*2+2]
            xv[3] = vertices[offset+pind3*2+1]; yv[3] = vertices[offset+pind3*2+2]
            xv[4] = vertices[offset+pind4*2+1]; yv[4] = vertices[offset+pind4*2+2]
            tmpdx, tmpdy = obstacle(bc.shape,xx,yy,xv,yv)
            dxpolygon += SVector{2,T}(tmpdx, tmpdy)
            # @cuprintln("p1x=$p1x,p1y=$p1y")
        end
    end
    ipoly = Npoly1side
    xv = yv = SVector{3,T}(0,0,0)
    pind1 = ipoly-1; pind2 = pind1+1; pind3 = pind2+1;
    for iside in 0:1
        offset = iside * Nvertices1side
        xv[1] = vertices[offset+pind1*2+1]; yv[1] = vertices[offset+pind1*2+2]
        xv[2] = vertices[offset+pind2*2+1]; yv[2] = vertices[offset+pind2*2+2]
        xv[3] = vertices[offset+pind3*2+1]; yv[3] = vertices[offset+pind3*2+2]
        tmpdx, tmpdy = obstacle(bc.shape,xx,yy,xv,yv)
        dxpolygon += SVector{2,T}(tmpdx, tmpdy)
    end
    dx, px = BD.periodicbc(xx, x1, x2, Lx)
    dxperio = SVector{2,T}(dx, 0)
    dimg = SVector{2,Int}(px, 0)
    # update position first due to wall in `y` and obstacles
    xnew = x̃ + dxwall + dxpolygon
    # periodic wrapping
    xnew += dxperio
    return dxwall, xnew, q̃, dimg
end

# ### overload the kernel for external flow field

@inline function findind(Ngrid, grid, x)
    i = Int64(floor(Ngrid / 2))
    if (grid[i] <= x)
        low = i
        high = Ngrid
    else
        low = 1
        high = i
    end
    while (high > low + 1)
        i = Int64(floor((low + high) / 2))
        if (grid[i] <= x)
            low = i
        else
            high = i
        end
    end
    return i
end

# return the binlinear interpolated scalar value
@inline function interpolate_u(xgrid, ygrid,
    indx, indy, x, y, ugrid)
    x1 = xgrid[indx]
    x2 = xgrid[indx+1]
    y1 = ygrid[indy]
    y2 = ygrid[indy+1]
    tmp = (x2 - x1) * (y2 - y1)
    f11 = ugrid[indx,indy]
    f12 = ugrid[indx,indy+1]
    f21 = ugrid[indx+1,indy]
    f22 = ugrid[indx+1,indy+1]
    a00 = x2 * y2 * f11 - x2 * y1 * f12 -x1 * y2 * f21 + x1 * y1 * f22
    a10 = -y2 * f11 + y1 * f12 + y2 * f21 - y1 * f22
    a01 = -x2 * f11 + x2 * f12 + x1 * f21 - x1 * f22
    a11 = f11 - f12 - f21 + f22
    a00 /= tmp
    a10 /= tmp
    a01 /= tmp
    a11 /= tmp
    u = a00 + a10 * x + a01 * y + a11 * x * y
    # @cuprintln("x=$x,y=$y,x1=$x1,y1=$y1,x2=$x2,y2=$y2,f11=$f11,f12=$f12,f21=$f21,f22=$f22,u=$u")
    return u
end

@inline function generate_runtime_levy(alpha::Real, tauR::Real)
    taum = (alpha - 1.0) * tauR / alpha;
    # curand_uniform_double is uniformly distributed in (0,1)
    U = rand();
    tau = taum / (U^(1.0 / alpha));
    return tau
end

@inline function random_orientation(qi::Real)
    return rand() .* 2 * pi
end

function BD.bd_kernel(integrator::BD.EulerMaruyama,
    particle,
    domain::Irregular{N},
    bc,
    bcsolver,
    x,
    q,
    img,
    data,
    U, # external velocity
    Omega, # external rotary velocity
    t::Real, # current time
    eta::Real,
    batch_size::Integer,
    op,
    cache
) where {N}

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    # the for loop allows more particles than the kernel config
    @inbounds for i = index:stride:length(x)
        for batch_id in 1:batch_size
            # get my own position and orientation
            xi = x[i]
            qi = q[i]
            # total velocity: external, translational Brownian and active swimming
            Nx = cache.Nx
            Ny = cache.Ny
            xgrid = cache.xgrid
            ygrid = cache.ygrid
            Ux = cache.U
            Uy = cache.V
            Omegaz = cache.Omega
            indx = findind(Nx, xgrid, xi[1])
            indy = findind(Ny, ygrid, xi[2])
            ux = interpolate_u(xgrid, ygrid,
                indx, indy, xi[1], xi[2], Ux)
            uy = interpolate_u(xgrid, ygrid,
                indx, indy, xi[1], xi[2], Uy)
            tmpu = SVector{2,Float64}(ux, uy)
            # if abs(xi[2])<5
            #     @cuprintln("(x,y)=,$(xi[1]),$(xi[2]),tmpu=$ux,$uy") 
            # end
            tmpomega = interpolate_u(xgrid, ygrid,
                indx, indy, xi[1], xi[2], Omegaz)
            # @cuprintln("tmpomega=$tmpomega")
            u = tmpu + U(xi, qi, t) + BD.linear_stochastic_velocity(integrator, t, xi, qi, particle.linear)
            # angular velocity: external and Brownian
            omega = tmpomega + Omega(xi, qi, t) + BD.angular_stochastic_velocity(integrator, t, xi, qi, particle.angular)
            ### remove after debug
            x[i] += u * integrator.dt
            q[i] = BD.update_orientation(qi, omega, integrator.dt)
            ### remove after debug
            # apply boundary conditions
            record, x[i], q[i], dimg = bc(bcsolver, xi, qi, u, omega, eta, integrator.dt, cache.vertices)
            # update periodic images
            img[i] += dimg
            # store additional quantities to data,  preprocessed with `op`
            op(data, i, record, batch_id, batch_size)
            taui = data[i]
            if taui<t
                dtau = generate_runtime_levy(cache.alpha, cache.tauR)
                data[i] += dtau
                q[i] = random_orientation(qi)
            end
            # update time
            t += integrator.dt
        end
    end
    return nothing
end


"""
    _initialize!(data::BD.ParticleData, x, q, tauR)
Initialize both positions and orientations, runtime with arrays.
"""
function BD._initialize!(data::BD.ParticleData,
    x::AbstractVector{<:SVector{N,<:Real}},
    q::AbstractVector{Q},
    tauR::AbstractVector{<:Real}) where {N,Q<:Union{<:Real,<:BD.Quaternion{<:Real}}}
    copyto!(data.x0, x)
    copyto!(data.x, x)
    copyto!(data.q0, q)
    copyto!(data.q, q)
    copyto!(data.data, tauR)
    return nothing
end

"""
    _initialize!(data::BD.ParticleData, x, q)
Initialize orientations with an array of orientations but positions of all particles are the same.
"""
function BD._initialize!(data::BD.ParticleData,
    x::SVector{N,<:Real},
    q::AbstractVector{Q},
    tauR::Real) where {N,Q<:Union{<:Real,<:BD.Quaternion{<:Real}}}
    fill!(data.x0, x)
    fill!(data.x, x)
    copyto!(data.q0, q)
    copyto!(data.q, q)
    fill!(data.data, tauR)
    return nothing
end

function BD.initialize!(data::BD.ParticleData, x, q, tauR)
    BD._initialize!(data, x, q, tauR)
end


@inline function poiseuille_linear_velocity_and_swim(domain::BD.Box{2,T}, shape::BD.SphereLike, uf::Real, U0::Real) where {T}
    function U(x::SVector{2,T}, q, t) where {T}
        H = domain.widths[2]
        ux = uf * (1 - 4 * x[2]^2 / H^2)
        return SVector{2,T}(ux, 0.0) + U0 * BD.q2c(q)
    end
end

@inline function linear_velocity_swimonly(domain :: Irregular{2,T}, shape::BD.SphereLike, U0::Real) where{T}
    function U(x::SVector{2,T}, q, t) where{T}
        return U0 * BD.q2c(q)
    end
end

@inline function poiseuille_angular_velocity(domain::BD.Box{2,T}, shape::BD.SphereLike, uf::Real) where {T}
    function Omega(x::SVector{2,T}, q, t) where {T}
        H = domain.widths[2]
        # half of vorticity
        omega = 4 * uf * x[2] / H^2
        return omega
    end
end

@inline function zero_angular_velocity(domain::Irregular{2,T}, shape::BD.SphereLike) where {T}
    function Omega(x::SVector{2,T}, q, t) where {T}
        # H = domain.widths[2]
        # half of vorticity
        # omega = 4 * uf * x[2] / H^2
        return 0.0
    end
end

