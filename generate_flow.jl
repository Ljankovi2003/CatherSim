"""
Stokes flow generation using Gridap.jl
This script replaces the old WaterLily.jl explicitly-stepped flow generation,
solving the steady Stokes equations directly. This is much faster and more 
accurate for creeping flow (Re = 0) than explicit projection methods,
and the body-fitted mesh prevents artificial smearing of the vorticity at the tip.

Install dependencies before running:
    ] add Gridap GridapGmsh
"""

using Gridap
using GridapGmsh
using Printf
using DelimitedFiles

# Geometry Parameters (matching BD simulation)
Lx = 100.0
Ly = 20.0
x2 = 41.0
x3 = 41.0
h  = 2.0

# Calculate obstacle vertices
x10 = 40.0 - Lx/2
x20 = x2 - Lx/2
x30 = x3 - Lx/2
y1  = -Ly/2
y2  = h - Ly/2
y3  = -Ly/2

x40 = x3 - Lx/2
x50 = x2 - Lx/2
x60 = 40.0 - Lx/2
y4  = Ly/2
y5  = Ly/2 - h
y6  = Ly/2

# Output Grid Parameters (matching test.jl)
Nx = 1001
Ny = 201
xgrid = collect(LinRange(-Lx/2.0, Lx/2.0, Nx))
ygrid = collect(LinRange(-Ly/2.0, Ly/2.0, Ny))

# 1. Generate Gmsh mesh
# We use Gmsh to generate a body-fitted mesh. This is crucial for capturing
# the exact shape of the sharp obstacle tip, yielding correct localized vorticity.
function create_mesh()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("channel")

    # Define characteristic lengths for mesh refinement
    lc_bulk = 0.5
    lc_tip = 0.05 # fine resolution at the sharp tips to capture vorticity

    # Corner points of the domain
    gmsh.model.geo.addPoint(-Lx/2, -Ly/2, 0, lc_bulk, 1)
    gmsh.model.geo.addPoint( Lx/2, -Ly/2, 0, lc_bulk, 2)
    gmsh.model.geo.addPoint( Lx/2,  Ly/2, 0, lc_bulk, 3)
    gmsh.model.geo.addPoint(-Lx/2,  Ly/2, 0, lc_bulk, 4)

    # Bottom obstacle
    gmsh.model.geo.addPoint(x10, y1, 0, lc_bulk, 5)
    gmsh.model.geo.addPoint(x20, y2, 0, lc_tip,  6) # sharp tip
    gmsh.model.geo.addPoint(x30, y3, 0, lc_bulk, 7)

    # Top obstacle
    gmsh.model.geo.addPoint(x40, y4, 0, lc_bulk, 8)
    gmsh.model.geo.addPoint(x50, y5, 0, lc_tip,  9) # sharp tip
    gmsh.model.geo.addPoint(x60, y6, 0, lc_bulk, 10)

    # Lines for boundaries
    # Bottom wall
    gmsh.model.geo.addLine(1, 5, 1)
    gmsh.model.geo.addLine(5, 6, 2)
    gmsh.model.geo.addLine(6, 7, 3)
    gmsh.model.geo.addLine(7, 2, 4)
    # Right wall (Outlet)
    gmsh.model.geo.addLine(2, 3, 5)
    # Top wall
    gmsh.model.geo.addLine(3, 10, 6)
    gmsh.model.geo.addLine(10, 9, 7)
    gmsh.model.geo.addLine(9, 8, 8)
    gmsh.model.geo.addLine(8, 4, 9)
    # Left wall (Inlet)
    gmsh.model.geo.addLine(4, 1, 10)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [10], 1) # inlet
    gmsh.model.setPhysicalName(1, 1, "inlet")
    
    gmsh.model.addPhysicalGroup(1, [5], 2) # outlet
    gmsh.model.setPhysicalName(1, 2, "outlet")
    
    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 6, 7, 8, 9], 3) # walls and obstacles
    gmsh.model.setPhysicalName(1, 3, "walls")
    
    gmsh.model.addPhysicalGroup(2, [1], 4) # fluid domain
    gmsh.model.setPhysicalName(2, 4, "fluid")

    gmsh.model.mesh.generate(2)
    gmsh.write("channel.msh")
    gmsh.finalize()
end

println("Generating Gmsh mesh...")
create_mesh()

# 2. Setup the Stokes problem with Gridap
model = GmshDiscreteModel("channel.msh")

# Define reference elements (Taylor-Hood elements for stability)
order = 2
refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
refe_p = ReferenceFE(lagrangian, Float64, order - 1)

# Test spaces
V = TestFESpace(model, refe_u, conformity=:H1, dirichlet_tags=["inlet", "walls"])
Q = TestFESpace(model, refe_p, conformity=:H1)

# Inlet velocity profile (Poiseuille-like parabola)
U_max = 1.0 # This can be scaled linearly later due to Stokes equations being linear
u_in(x) = VectorValue(U_max * (1.0 - (2.0*x[2]/Ly)^2), 0.0)
u_wall(x) = VectorValue(0.0, 0.0)

# Trial spaces
U = TrialFESpace(V, [u_in, u_wall])
P = TrialFESpace(Q)

Y = MultiFieldFESpace([V, Q])
X = MultiFieldFESpace([U, P])

# Integration measure
degree = order * 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

# Weak form of the Steady Stokes equations
# ∫ (∇v : ∇u) dΩ - ∫ (∇⋅v) p dΩ + ∫ q (∇⋅u) dΩ = 0
a((u, p), (v, q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ
l((v, q)) = ∫( 0.0*q )dΩ # No body forces

# 3. Solve the system
println("Assembling and solving the linear Stokes system...")
op = AffineFEOperator(a, l, X, Y)
uh, ph = solve(op)

# 4. Compute vorticity
println("Computing vorticity...")
# Vorticity ω = ∂v/∂x - ∂u/∂y
∇u = ∇(uh)
ωh = cell_field(x -> ∇u(x)[2,1] - ∇u(x)[1,2], Ω)

# 5. Interpolate onto the Cartesian Grid for BD.jl
println("Interpolating results onto Cartesian grid...")
Ux = zeros(Nx, Ny)
Uy = zeros(Nx, Ny)
Omega = zeros(Nx, Ny)

# Gridap interpolation over points
points = [Point(x, y) for x in xgrid, y in ygrid]
u_vals = uh(points)
ω_vals = ωh(points)

for i in 1:Nx
    for j in 1:Ny
        Ux[i, j] = u_vals[i, j][1]
        Uy[i, j] = u_vals[i, j][2]
        Omega[i, j] = ω_vals[i, j]
    end
end

# 6. Save to expected format
flowfname = @sprintf("flow_%g_%g_%g.txt", x2, x3, h)
println("Saving flow field to $flowfname")

open(flowfname, "w") do io
    # Write exactly 9 header lines to match skipstart=9 in BD code
    println(io, "Gridap Steady Stokes Solver Output")
    println(io, "Lx=$Lx, Ly=$Ly, x2=$x2, x3=$x3, h=$h")
    println(io, "Nx=$Nx, Ny=$Ny")
    println(io, "x, y, u, v, omega")
    println(io, "--------------------")
    println(io, "--------------------")
    println(io, "--------------------")
    println(io, "--------------------")
    println(io, "--------------------")
    
    # Write the flattened data (Column-major order expected by typical reshape)
    # The BD code does: tmpU = reshape(flowdata[:,3], (Nx, Ny))
    # In Julia, reshape fills column by column (y index varies fastest, then x index varies)
    # But wait, looking at BD code: `tmpU = reshape(tmpU, (Nx, Ny))` means the file must have
    # rows corresponding to (x1, y1), (x2, y1), ..., (xNx, y1), (x1, y2) ...
    # Wait, Julia's `reshape(arr, Nx, Ny)` expects `arr` to vary its FIRST index (Nx) fastest.
    # So the file should loop: `for j=1:Ny, i=1:Nx` or `for i=1:Nx, j=1:Ny`?
    # If `reshape(arr, Nx, Ny)` takes 1D array to 2D array, it goes down rows (i) then columns (j).
    # This means the 1D array should be `arr[i + (j-1)*Nx]`.
    # Let's write it in this exact order: `for j in 1:Ny`, `for i in 1:Nx`.
    for j in 1:Ny
        for i in 1:Nx
            # x, y, u, v, omega
            println(io, "$(xgrid[i])\t$(ygrid[j])\t$(Ux[i,j])\t$(Uy[i,j])\t$(Omega[i,j])")
        end
    end
end

println("Done! Flow data is ready.")
