using Gridap
using Gridap.Fields
using Gridap.CellData
using Gridap.Arrays
using LineSearches: BackTracking, Static, MoreThuente
using FillArrays
#using GridapGmsh
#using GridapDistributed
#using PartitionedArrays
#using GridapPETSc
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
ν = 0.001 #0.001 m2/s 
N = 64; #cells per dimensions
hf = VectorValue(0,0)
stabilization = 2
"""
1 complex time dipendant stabilization, not enought
2 very easy stabilization, it WORKS
2.1 improvement
3 no stabilizaztion
"""
initial_condition = true #print model of initial condition



#MESH DEFINITION
domain = (-D, D, -D, D)
partition = (N, N)
model = CartesianDiscreteModel(domain, partition; isperiodic=(true, true))
include("BCS.jl")

writevtk(model,"model")

#ANALITICAL SOLUTION, used also for initial condition
Tx(x, t) = pi / D * (x[1] - Ua * t)
Ty(x, t) = pi / D * (x[2] - Va * t)
Et(t) = exp(-(2 * ν * t * pi^2) / (D^2))
ua(x, t) = Ua - Vs * cos(Tx(x, t)) * sin(Ty(x, t)) * Et(t)
va(x, t) = Va + Vs * sin(Tx(x, t)) * cos(Ty(x, t)) * Et(t)
velocity(x, t) = VectorValue(ua(x, t), va(x, t))
pa(x, t) = -(Vs^2 / 4) * (cos(2 * Tx(x, t)) + cos(2 * Ty(x, t))) * Et(t)^2
ωa(x, t) = 2 * Vs * pi / D * cos(Tx(x, t)) * cos(Ty(x, t)) * Et(t)^2

ω₀=  2 * Vs * pi / D 

ua(t::Real) = x -> ua(x, t)
va(t::Real) = x -> va(x, t)
velocity(t::Real) = x -> velocity(x, t)
pa(t::Real) = x -> pa(x, t)
ωa(t::Real) = x -> ωa(x, t)


order = 1
reffeᵤ = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2*order)
V = TestFESpace(model, reffeᵤ, conformity=:H1)
reffeₚ = ReferenceFE(lagrangian,Float64,order)
#reffeₚ = ReferenceFE(lagrangian,Float64,order-1; space=:P)
#reffeₚ = ReferenceFE(lagrangian, Float64, order - 1)
#Q = TestFESpace(model,reffeₚ, conformity=:L2, constraint=:zeromean)
#Q = TestFESpace(model,reffeₚ, conformity=:L2, dirichlet_tags="interior")
Q = TestFESpace(model,reffeₚ, conformity=:H1, dirichlet_tags="centre")

#Since we impose Dirichlet boundary conditions on the entire boundary ∂Ω, the mean value of the pressure is constrained to zero in order have a well posed problem
#Q = TestFESpace(model, reffeₚ)


#Transient is just for the fact that the boundary conditions change with time
U = TrialFESpace(V)
#P = TrialFESpace(Q) #?transient
P = TransientTrialFESpace(Q, pa) #?transient



Y = MultiFieldFESpace([V, Q]) #?transient
X = TransientMultiFieldFESpace([U, P])

degree = 4
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

if initial_condition
  #Computing Initial condition for checking
  u0 = velocity(0)
  ω0 = ωa(0)
  ω1 = ∇ × u0 #checking that ω == ω1, so correct implementation of formula
  p0 = pa(0)
  writevtk(Ω, "Sol_t0", cellfields=["u" => u0, "p" => p0, "ω" => ω0, "ω1" => ω1])
end




# Momentum residual, without the viscous term
Rm(t,(u,p)) = ∂t(u) + u⋅∇(u) + ∇(p) - hf

# Continuity residual
Rc(u) = ∇⋅u

h = lazy_map(h->h^(1/2),get_cell_measure(Ω))

function τ(u,h)
    β=1.3
    τ₂ = h^2/(4*ν)
    val(x) = x
    val(x::Gridap.Fields.ForwardDiff.Dual) = x.value
    u = val(norm(u))
    
    if iszero(u)
        return τ₂
        
    end
    τ₃ =  0 #dt/2 #h/(2*u)

    τ₁ = h/(2*β) #h/(2*u)
    
    return 1/(1/τ₁ + 1/τ₂ + 1/τ₃)
  
end


τb(u,h) = (u⋅u)*τ(u,h)


var_equations(t,(u,p),(v,q)) = ∫(
    ν*∇(v)⊙∇(u) # Viscous term
    + v⊙Rm(t,(u,p)) # Other momentum terms
    + q*Rc(u)
 )dΩ # Continuity


stab_equations(t,(u,p),(v,q)) = ∫(  (τ∘(u,h)*(u⋅∇(v) + ∇(q)))⊙Rm(t,(u,p)) # First term: SUPG, second term: PSPG
    + τb∘(u,h)*(∇⋅v)⊙Rc(u) # Bulk viscosity. Try commenting out both stabilization terms to see what happens in periodic and non-periodic cases
)dΩ

res(t,(u,p),(v,q)) = var_equations(t,(u,p),(v,q)) + stab_equations(t,(u,p),(v,q))
op = TransientFEOperator(res,X,Y)
nls = NLSolver(show_trace=true, method=:newton, linesearch=MoreThuente(), iterations=30)

solver = FESolver(nls)




U0 = U(0.0)
P0 = P(0.0)
X0 = X(0.0)

uh0 = interpolate_everywhere(velocity(0), U0)
ph0 = interpolate_everywhere(pa(0), P0)

xh0 = interpolate_everywhere([uh0, ph0], X0)


t0 = 0.0
dt = 0.01 # Vs/(2*D)
dt < (Vs+Ua+Vs+Va)/(2*D/N)
tF = 0.5

θ = 0.5


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
    ωn = ωh_tn./ω₀
    p_analytic = pa(_t_nn)
    u_analytic = velocity(_t_nn)
    w_analytic = ωa(_t_nn)
    pvd[tn] = createvtk(Ω, "Results/TV_2d_$_t_nn" * ".vtu", cellfields=["uh" => uh_tn, "ph" => ph_tn, "wh" => ωh_tn,  "wn" => ωn, "p_analytic"=>p_analytic, "u_analytic"=>u_analytic,  "w_analytic"=>w_analytic])
  end

end

CFL = 1*dt./h