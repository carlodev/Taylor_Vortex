using Gridap
using Gridap.Fields
using Gridap.CellData
using Gridap.Arrays

using FillArrays
#using GridapGmsh
#using GridapDistributed
#using PartitionedArrays
#using GridapPETSc
using LineSearches: BackTracking
using Revise

#include("StabilizeComputationCart.jl")
#using .StabilizeComputationCart: h_compute2D, stabilization_coefficients
"""
Taylor Green 2D vortex
with and without stabilization
"""


D = 0.5 #0.5 [m] vortex dimension
Vs = 1 #1[m/s]swirling speed
Ua = 0.3 #0.3 [m/s]convective velocity in x
Va = 0.2 #0.2 [m/s]convective velocity in y
nu = 0.001 #0.001 m2/s 
N = 64; #cells per dimensions

stabilization = 2
"""
1 complex time dipendant stabilization, not enought
2 very easy stabilization, it WORKS
2.1 improvement
3 no stabilizaztion
"""
initial_condition = false #print model of initial condition



#MESH DEFINITION
domain = (-D, D, -D, D)
partition = (N, N)
model = CartesianDiscreteModel(domain, partition; isperiodic=(true, true))
#writevtk(model,"model")

#ANALITICAL SOLUTION, used also for initial condition
Tx(x, t) = pi / D * (x[1] - Ua * t)
Ty(x, t) = pi / D * (x[2] - Va * t)
Et(t) = exp(-(2 * nu * t * pi^2) / (D^2))
ua(x, t) = Ua - Vs * cos(Tx(x, t)) * sin(Ty(x, t)) * Et(t)
va(x, t) = Va + Vs * sin(Tx(x, t)) * cos(Ty(x, t)) * Et(t)
velocity(x, t) = VectorValue(ua(x, t), va(x, t))
pa(x, t) = -(Vs^2 / 4) * (cos(2 * Tx(x, t)) + cos(2 * Ty(x, t))) * Et(t)^2
ωa(x, t) = 2 * Vs * pi / D * cos(Tx(x, t)) * cos(Ty(x, t)) * Et(t)^2

ua(t::Real) = x -> ua(x, t)
va(t::Real) = x -> va(x, t)
velocity(t::Real) = x -> velocity(x, t)
pa(t::Real) = x -> pa(x, t)
ωa(t::Real) = x -> ωa(x, t)



order = 2
reffeᵤ = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
V = TestFESpace(model, reffeᵤ, conformity=:H1)

reffeₚ = ReferenceFE(lagrangian,Float64,order-1; space=:P)
#reffeₚ = ReferenceFE(lagrangian, Float64, order - 1)
#Q = TestFESpace(model,reffeₚ, conformity=:L2, constraint=:zeromean)
Q = TestFESpace(model,reffeₚ, conformity=:L2, dirichlet_tags="interior")

#Since we impose Dirichlet boundary conditions on the entire boundary ∂Ω, the mean value of the pressure is constrained to zero in order have a well posed problem
#Q = TestFESpace(model, reffeₚ)


#Transient is just for the fact that the boundary conditions change with time
U = TransientTrialFESpace(V)
#P = TrialFESpace(Q) #?transient
P = TransientTrialFESpace(Q, pa) #?transient



Y = MultiFieldFESpace([V, Q]) #?transient
X = TransientMultiFieldFESpace([U, P])

degree = order
Ω = Triangulation(model)
dΩ = Measure(Ω, 2*degree)

if initial_condition
  #Computing Initial condition for checking
  u0 = velocity(0)
  ω0 = ωa(0)
  ω1 = ∇ × u0 #checking that ω == ω1, so correct implementation of formula
  p0 = pa(0)
  writevtk(Ω, "Sol_t0", cellfields=["u" => u0, "p" => p0, "ω" => ω0, "ω1" => ω1])
end



conv(u, ∇u) = (∇u') ⋅ u
m(ut, v) = ∫(ut ⋅ v)dΩ
a(t, (u, p), (v, q)) = ∫(nu * ∇(v) ⊙ ∇(u) - (∇ ⋅ v) * p + q * (∇ ⋅ u))dΩ
c(u, v) = ∫(v ⊙ (conv ∘ (u, ∇(u))))dΩ


#Stabilization
if stabilization == 1
  include("StabilizeComputationFF.jl")
  
  #stab_su(t, (u, p), (v, q)) = ∫(tau_su(t) ⋅ ((u ⋅ ∇(v))' ⊙ (conv ∘ (u, ∇(u))) + (u ⋅ ∇(v))' ⊙ ∂t(u) + (u ⋅ ∇(v))' ⊙ ∇(p)))dΩ
  stab_su(t, (u, p), (v, q)) = ∫(tau_su(t) ⋅ ((u ⋅ ∇(v))' ⊙ (conv ∘ (u, ∇(u))) + (u ⋅ ∇(v))' ⊙ ∂t(u) + (u ⋅ ∇(v))' ⊙ ∇(p)))dΩ
  stab_bk(t, u, v) = ∫(tau_bk(t) ⋅ ((∇⋅ v) ⊙ (∇ ⋅ u)))dΩ
  stab_ps(t, (u, p), (v, q)) = ∫(tau_ps(t) ⋅ ((∇(q)) ⊙ ∂t(u) + (∇(q)) ⊙ (conv ∘ (u, ∇(u))) + (∇(q)) ⋅ (∇(p))))dΩ

  """
  stab_su(t, (u, p), (v, q)) = ∫(tau_su(t,u) ⋅ ((u ⋅ ∇(v))' ⊙ (conv ∘ (u, ∇(u))) + (u ⋅ ∇(v))' ⊙ ∂t(u) + (u ⋅ ∇(v))' ⊙ ∇(p)))dΩ
  stab_bk(t, u, v) = ∫(tau_bk(t,u) ⋅ (nu .* ∇(v) ⊙ ∇(u)))dΩ
  stab_ps(t, (u, p), (v, q)) = ∫(tau_ps(t,u) ⋅ ((∇(q)) ⊙ ∂t(u) + (∇(q)) ⊙ (conv ∘ (u, ∇(u))) + (∇(q)) ⋅ (∇(p))))dΩ
  """
  res(t, (u, p), (v, q)) = a(t, (u, p), (v, q)) + c(u, v) + m(∂t(u), v) + stab_su(t, (u, p), (v, q)) + stab_bk(t, u, v) + stab_ps(t, (u, p), (v, q))
elseif stabilization == 2
  dim= num_cell_dims(Ω)
  h = lazy_map(h->h^(1/dim),get_cell_measure(Ω))
  u_0 = Vs+Ua+Va
  r = 2
  tau_su_adv = (h ./ (2 .* u_0)).^(-r)
  tau_su_diff = (h .* h ./ (4 * nu)).^(-r)
  tau_su_unst =( h ./ (2 .* u_0)).^(-r)

   #Tezduyar

  tau_su= (tau_su_adv + tau_su_diff .+ tau_su_unst).^(-1 / r)
  tau_ps = (tau_su_adv + tau_su_diff).^(-1 / r)
  tau_bk = u_0^2 .* tau_ps

  stab_su(t, (u, p), (v, q)) = ∫(tau_su⋅ ((u ⋅ ∇(v))' ⊙ (conv ∘ (u, ∇(u))) + (u ⋅ ∇(v))' ⊙ ∂t(u) + (u ⋅ ∇(v))' ⊙ ∇(p)))dΩ
  stab_bk(t, u, v) = ∫(tau_bk ⋅ (∇(v) ⊙ ∇(u)))dΩ
  stab_ps(t, (u, p), (v, q)) = ∫(tau_ps ⋅ ((∇(q)) ⊙ ∂t(u) + (∇(q)) ⊙ (conv ∘ (u, ∇(u))) + (∇(q)) ⋅ (∇(p))))dΩ


  res(t, (u, p), (v, q)) = a(t, (u, p), (v, q)) + c(u, v) + m(∂t(u), v) + stab_su(t, (u, p), (v, q)) + stab_bk(t, u, v) + stab_ps(t, (u, p), (v, q))

elseif stabilization == 2.1
  dim= num_cell_dims(Ω)
#h = lazy_map(h->h^(1/dim),get_cell_measure(Ω))
  h = 2*D/N
  u1(t) = velocity(t - dt)
  r = 2    #Tezduyar

  tau_su_adv(t) = (h ./ (2 ⊙ u1(t))).^(-r)
  tau_su_diff(t) = (h .* h ./ (4 * nu)).^(-r)
  tau_su_unst(t) =( h ./ (2 .* u1(t))).^(-r)


  tau_su(t) = (tau_su_adv(t) + tau_su_diff(t) + tau_su_unst(t)).^(-1 / r)
  tau_ps(t) = (tau_su_adv(t) + tau_su_diff(t)).^(-1 / r)
  tau_bk(t) = u_0^2 .* tau_ps(t)

  stab_su(t, (u, p), (v, q)) = ∫(tau_su(t)⋅ ((u ⋅ ∇(v))' ⊙ (conv ∘ (u, ∇(u))) + (u ⋅ ∇(v))' ⊙ ∂t(u) + (u ⋅ ∇(v))' ⊙ ∇(p)))dΩ
  stab_bk(t, u, v) = ∫(tau_bk(t) ⋅ (∇(v) ⊙ ∇(u)))dΩ
  stab_ps(t, (u, p), (v, q)) = ∫(tau_ps(t) ⋅ ((∇(q)) ⊙ ∂t(u) + (∇(q)) ⊙ (conv ∘ (u, ∇(u))) + (∇(q)) ⋅ (∇(p))))dΩ


  res(t, (u, p), (v, q)) = a(t, (u, p), (v, q)) + c(u, v) + m(∂t(u), v) + stab_su(t, (u, p), (v, q)) + stab_bk(t, u, v) + stab_ps(t, (u, p), (v, q))




elseif stabilization == 4

  res(t, (u, p), (v, q)) = a(t, (u, p), (v, q)) + c(u, v) + m(∂t(u), v)
end


#writevtk(Ω,"Tau_r1",cellfields=[ "tau_su"=>tau_su(0), "tau_ps"=>tau_ps(0), "tau_bk"=>tau_bk(0)])
#writevtk(Ω,"Tau_r1",cellfields=[ "tau_su"=>tau_su, "tau_ps"=>tau_ps, "tau_bk"=>tau_bk])


op = TransientFEOperator(res, X, Y)



U0 = U(0.0)
P0 = P(0.0)
X0 = X(0.0)

uh0 = interpolate_everywhere(velocity(0), U0)
ph0 = interpolate_everywhere(pa(0), P0)

xh0 = interpolate_everywhere([uh0, ph0], X0)


t0 = 0.0
dt = 0.5 # Vs/(2*D)
dt < (Vs+Ua+Vs+Va)/(2*D/N)
tF = 20

θ = 0.5

using LineSearches: BackTracking
nls = NLSolver(
  show_trace=true, method=:newton, linesearch=BackTracking())

ode_solver = ThetaMethod(nls, dt, θ)

sol_t = solve(ode_solver, op, xh0, t0, tF)


_t_nn = t0
createpvd("TV_2d") do pvd
  for (xh_tn, tn) in sol_t
    global _t_nn
    _t_nn += dt
    uh_tn = xh_tn[1]
    ph_tn = xh_tn[2]
    ωh_tn = ∇ × uh_tn
    pvd[tn] = createvtk(Ω, "Results/TV_2d_$_t_nn" * ".vtu", cellfields=["uh" => uh_tn, "ph" => ph_tn, "ωh" => ωh_tn])
  end

end