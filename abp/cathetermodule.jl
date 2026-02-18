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
    x0::Real, y0::Real, x1::Real, y1::Real, x2::Real, y2::Real)
    # the line segment is signed
    # with normal vector n = ez x (x1-x0, y1-y0, 0)
    # the aux vectors:
    ell0 = SVector{2,Float64}(x1-x0,y1-y0)
    n0 = SVector{2,Float64}(-y1+y0,x1-x0)
    ell0norm = norm(ell0)
    n0 = n0/ell0norm
    ell02 = sum(ell0 .* ell0)
    ell1 = SVector{2,Float64}(x2-x1,y2-y1)
    n1 = SVector{2,Float64}(-y2+y1,x2-x1)
    ell1norm = norm(ell1)
    n1 = n1/ell1norm
    ell12 = sum(ell1 .* ell1)
    # the criterion of crossing segment happen:
    # (1) r0.n<0
    # (2) 0<r0.ell<ell2
    # a bottom wall.
    # @cuprintln("x0=$x0,x1=$x1,y0=$y0,y1=$y1,n0=$(n0[1]),$(n0[2])")
    d0 = -((x-x0)*n0[1]+(y-y0)*n0[2])
    d1 = -((x-x1)*n1[1]+(y-y1)*n1[2])
    dx = dy = 0.0
    # if x<-10 && x>-15 && y0>10
    # if x>-10 && y0>10
    #     @cuprintln("x=$x,y=$y,y0=$y0, d0=$d0, d1=$d1")
    # end
    if (d0>0 && d1>0)
        if (d0<d1)
            dx = 2*d0*n0[1];
            dy = 2*d0*n0[2];
        else
            dx = 2*d1*n1[1];
            dy = 2*d1*n1[2];
            #@cuprintln("(x,y)=($x,$y), (x0,y0)=($x0,$y0), (dx,dy)=($dx,$dy), d0=$d0, d1=$d1")
        end
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
    xx, yy = x
    if any(isnan.(x))
        @cuprintln("just in BC: x=$xx,$yy")
    end
    # compute tentative position and orientation
    x̃ = x + u * dt
    q̃ = BD.update_orientation(q, omega, dt)

    # extract scalar locations from vector
    xx, yy = x̃
    if any(isnan.(x̃))
        ux, uy = u
        @cuprintln("in BC integed: x̃=$xx,$yy, u=$ux,$uy, dt=$dt")
    end
    # no-flux in y
    dy = BD.parallel_flat_walls(bc.shape, yy, y1, y2)
    # wall displacement
    dxwall = SVector{2,T}(0, dy)
    # polygon displacement
    dxpolygon = SVector{2,T}(0, 0)
    # vertices = bc.domain.vertices
    Nvertices = length(vertices)
    for itri in 0:Int64(floor(Nvertices/6)-1)
        p1x = vertices[itri*6+1]; p1y = vertices[itri*6+2]
        p2x = vertices[itri*6+3]; p2y = vertices[itri*6+4]
        p3x = vertices[itri*6+5]; p3y = vertices[itri*6+6]
        tmpdx, tmpdy = obstacle(bc.shape,xx,yy,p1x,p1y,p2x,p2y,p3x,p3y)
        # @cuprintln("tmpdx=$tmpdx, tmpdy=$tmpdy")
        dxpolygon += SVector{2,T}(tmpdx, tmpdy)
        # @cuprintln("p1x=$p1x,p1y=$p1y")
    end
    # @cuprintln("dxwall=$dy, dxpolygon=$(dxpolygon[1]),$(dxpolygon[2])")
    # periodic in x
    dx, px = BD.periodicbc(xx, x1, x2, Lx)
    dxperio = SVector{2,T}(dx, 0)
    dimg = SVector{2,Int}(px, 0)
    # update position first due to wall in `y` and obstacles
    xnew = x̃ + dxwall + dxpolygon
    # periodic wrapping
    xnew += dxperio
    if any(isnan.(xnew))
        @cuprintln("x̃=$xx,$yy,dxperiod=$dx,dxwall=$dy, dxpolygon=$(dxpolygon[1]),$(dxpolygon[2])")
        error("nan!")
    end
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
            if any(isnan.(tmpu))
                @cuprintln("fluid u = $ux,$uy")
                error("u is nan")
            end
            # if abs(xi[2])<5
            #     @cuprintln("(x,y)=,$(xi[1]),$(xi[2]),tmpu=$ux,$uy") 
            # end
            tmpomega = interpolate_u(xgrid, ygrid,
                indx, indy, xi[1], xi[2], Omegaz)
            # @cuprintln("tmpomega=$tmpomega")
            u = tmpu + U(xi, qi, t) + BD.linear_stochastic_velocity(integrator, t, xi, qi, particle.linear)
            # angular velocity: external and Brownian
            omega = tmpomega + Omega(xi, qi, t) + BD.angular_stochastic_velocity(integrator, t, xi, qi, particle.angular)
            # ### remove after debug
            # x[i] += u * integrator.dt
            # q[i] = BD.update_orientation(qi, omega, integrator.dt)
            # ### remove after debug
            # apply boundary conditions
            xx, yy = xi
            if any(isnan.(xi))
                @cuprintln("before BC: xi=$xx,$yy")
            end
            record, x[i], q[i], dimg = bc(bcsolver, xi, qi, u, omega, eta, integrator.dt, cache.vertices)
            # update periodic images
            img[i] += dimg
            # store additional quantities to data,  preprocessed with `op`
            op(data, i, record, batch_id, batch_size)
            # update time
            t += integrator.dt
        end
    end
    return nothing
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

