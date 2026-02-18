import Pda
using Plots
using HDF5
using Statistics

Lx = T(100)
Ly = T(20)
outpath = "/home/tingtao/cathetersim/traindata/"

T = Float64
dim = 2
particleshape = BD.Point(dim) #BD.Sphere(dim, a)
U0 = T(20.0)
DT = T(0.1)
DR = T(0.5)

x10 = x60 = 40-Lx/2
y1 = 0-Ly/2
y3 = 0-Ly/2
y6 = Ly/2
y4 = Ly/2

x2 =41
x50 = x20 = x2 - Lx/2
x3=41
x40 = x30 = x3 - Lx/2
h=2
y2 = h-Ly/2
y5 = Ly/2-h
p1 = (x10, y1)
p2 = (x20, y2)
p3 = (x30, y3)
p4 = (x40, y4)
p5 = (x50, y5)
p6 = (x60, y6)
    
# flow test1
p1 = (-0.1*Lx, -0.5*Ly)
p2 = (-0.02*Lx, -0.25*Ly)
p3 = (-0.05*Lx, -0.5*Ly)
p4 = (-0.1*Lx, 0.5*Ly)
p5 = (-0.02*Lx, 0.25*Ly)
p6 = (-0.05*Lx, 0.5*Ly)
# # flow test 2
# p1 = (-0.1*Lx, -0.5*Ly)
# p2 = (-0.05*Lx, -0.25*Ly)
# p3 = (-0.02*Lx, -0.5*Ly)
# p4 = (-0.1*Lx, 0.5*Ly)
# p5 = (-0.05*Lx, 0.25*Ly)
# p6 = (-0.02*Lx, 0.5*Ly)

xx = [p1...,p2...,p3...]
xx = reshape(xx,2,3)
xhook1 = xx[1,:]
yhook1 = xx[2,:]
xx = [p4...,p5...,p6...]
xx = reshape(xx,2,3)
xhook2 = xx[1,:]
yhook2 = xx[2,:]


path = "/home/tingtao/cathetersim/"
fprefix = @sprintf("smooth_persistent_U0%guf%g0.5pi",U0, uf)
file = string(path, fprefix, ".h5")

fid = h5open(file, "r")

# x = Pda.get_h5data(file, "config/$(frame)/x")
# y = Pda.get_h5data(file, "config/$(frame)/y")
plot()
xp=zeros(0); yp=zeros(0);
Nframe = 500
pid = 8
x0 = 80.0
y0 = -10.0
anim = @animate for frame in 1:Nframe
    x = Pda.get_h5data(file, "config/$(frame)/x")
    y = Pda.get_h5data(file, "config/$(frame)/y")
    data = fid["config"][string(frame)]
    t = read_attribute(data,"t")
    append!(xp,x[pid]+x0)
    append!(yp,y[pid]+y0)
    scatter(xp,yp,c=colormap("Blues",Nframe),labels="")
    plot!(xp,yp,labels="",ylims=(-Ly/2, Ly/2),xlims=(-Lx,Lx/2))
    # plot!(xhook1,yhook1,lw=2,label="hook1")
    # plot!(xhook2,yhook2,lw=2,label="hook2")
    plot!(aspect_ratio=:equal, framestyle=:box)
    # quiver!([x[pid]*H],[y[pid]*H],gradient=10 .*(cos.([theta[pid]]),sin.([theta[pid]])))
    # plot!(H*hooks_arr,-0.5*H*ones(10,1),lw=2,color=:black,labels="")
    title!(string("t=",floor(t*1e4)/1e4))
end
gif(anim, string(path,fprefix,"_fps30.gif"), fps = 30)

# endoffile
print("finished")