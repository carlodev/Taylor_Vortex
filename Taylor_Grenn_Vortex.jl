using Gridap
#using GridapGmsh
#using GridapDistributed
#using PartitionedArrays
#using GridapPETSc
using LineSearches: BackTracking
using Revise

"""
Taylor Green 2D vortex
"""


D = 0.5 #[m] vortex dimension
Vs = 1 #[m/s]swirling speed
Ua = 0.3 #[m/s]convective velocity in x
Va = 0.2 #[m/s]convective velocity in y
nu = 0.001 #m2/s 
N=64; #cells per dimensions

#MESH DEFINITION
domain = (-D,D,-D,D)
partition = (N,N)
model = CartesianDiscreteModel(domain,partition; isperiodic=(true,true))
writevtk(model,"model")

#ANALITICAL SOLUTION, used also for initial condition
Tx(x,t) = pi/D*(x[1]-Ua*t)
Ty(x,t) = pi/D*(x[2]-Va*t)
Et(t) = exp(-(2*nu*t*pi^2)/(D^2))
u(x,t) = Ua - Vs*cos(Tx(x,t))*sin(Ty(x,t))*Et(t)
v(x,t) = Va + Vs*sin(Tx(x,t))*cos(Ty(x,t))*Et(t)
velocity(x,t) = VectorValue(u(x,t),v(x,t))
p(x,t) = -(Vs^2/4)*(cos(2*Tx(x,t))*cos(2*Ty(x,t)))*Et(t)^2
ω(x,t) = 2*Vs*pi/D*cos(Tx(x,t))*cos(Ty(x,t))*Et(t)^2

u(t::Real) = x -> u(x,t)
v(t::Real) = x -> v(x,t)
velocity(t::Real) = x -> velocity(x,t)
p(t::Real) = x -> p(x,t)
ω(t::Real) = x -> ω(x,t)



order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
V = TestFESpace(model,reffeᵤ,conformity=:H1)

reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)
Q = TestFESpace(model,reffeₚ,conformity=:L2, constraint=:zeromean)

U = TransientTrialFESpace(V)
P = TransientTrialFESpace(Q)

Y = TransientMultiFieldFESpace([V, Q])
X = TransientMultiFieldFESpace([U, P])

degree = order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

#Computing Initial condition for checking
u0 = velocity(0)
ω0 = ω(0)
ω1 = ∇ × u0 #checking that ω == ω1, so correct implementation of formula
writevtk(Ω,"Sol_t0",cellfields=["u" => u0,"ω" => ω0, "ω1" => ω1])


conv(u, ∇u) = (∇u') ⋅ u
m(ut, v) = ∫(ut ⋅ v)dΩ
a(t, (u, p), (v, q)) = ∫(nu * ∇(v) ⊙ ∇(u) - (∇ ⋅ v) * p + q * (∇ ⋅ u))dΩ
c(u, v) = ∫(v ⊙ (conv ∘ (u, ∇(u))))dΩ
res(t, (u, p), (v, q)) = a(t, (u, p), (v, q)) + c(u, v) + m(∂t(u), v) #- ∫(v ⋅ hf(t)) * dΩ

op = TransientFEOperator(res,X,Y)

U0 = U(0.0)
P0 = P(0.0)
X0 = X(0.0)

uh0 = interpolate_everywhere(velocity(0), U0)
ph0 = interpolate_everywhere(p(0), P0)
xh0 = interpolate_everywhere([uh0, ph0], X0)


t0 = 0.0
dt = 0.1 # Vs/(2*D)
tF = 2.5

θ = 0.5
print("Nls")

using LineSearches: BackTracking
nls = NLSolver(
  show_trace=true, method=:newton, linesearch=BackTracking())

ode_solver = ThetaMethod(nls,dt,θ)
print("starting solving")

sol_t = solve(ode_solver,op,xh0,t0,tF)


_t_nn = t0
createpvd("Channel2d_td") do pvd
  for (xh_tn, tn) in sol_t
    global _t_nn
    _t_nn += dt
    uh_tn = xh_tn[1]
    ph_tn = xh_tn[2]
    ωh_tn = ∇ × uh_tn
    pvd[tn] = createvtk(Ω,"Results/TV_$_t_nn"*".vtu",cellfields=["uh"=>uh_tn,"ph"=>ph_tn, "ωh" =>ωh_tn])
  end

end