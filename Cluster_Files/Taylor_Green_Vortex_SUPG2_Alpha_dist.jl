using Pkg
Pkg.activate(".")
using Gridap
using Gridap.Fields
using Gridap.CellData
using Gridap.Arrays
using LineSearches: BackTracking, Static, MoreThuente
using FillArrays
using GridapDistributed
using PartitionedArrays
using JLD2
"""
Taylor Green 2D vortex SUPG
with stabilization
velocity 1st order
pressure 1st order
ODE solver: GeneralizedAlpha 
"""


function TV1(parts)

  #N_cells = [32, 64, 128, 256, 512, 1024]

  N_cells = [256, 512, 1024]

  for i = 1:1:length(N_cells)
    N = N_cells[i] #cells per dimensions

    #Parameters

    D = 0.5 #0.5 [m] vortex dimension

    Vs = 1 #1[m/s]swirling speed
    Ua = 0.3 #0.3 [m/s]convective velocity in x
    Va = 0.2 #0.2 [m/s]convective velocity in y
    ν = 0.001 #0.001 m2/s 

    order = 1 #Order of pressure and velocity

    #ODE settings
    cell_h = 2 * D / N

    CFL = 0.32 #2*dt./h

    t0 = 0.0
    dt = CFL * cell_h / 2 #0.0001 
    δt = 2 * D / Vs #Non dimensional time
    tF = 2.5 * δt

    Ntimestep = (tF - t0) / dt

    ρ∞ = 0.8 #GeneralizedAlpha parameter, ρ∞=1 no dissipation, ρ∞ = 0 max dissipation


    initial_condition = false #print model of initial condition


    hf = VectorValue(0.0, 0.0) #external force: NONE

    #MESH DEFINITION
    domain = (-D, D, -D, D)
    partition = (N, N)
    model = CartesianDiscreteModel(parts, domain, partition; isperiodic=(true, true))


    labels = get_face_labeling(model)
    h0 = 2 * D / N
    eps1 = h0 / 10

    vt(i) = VectorValue(y, i)
    v1(v) = v[1]
    v2(v) = v[2]

    function is_centre(x)
      norm(v1.(x)) <= eps1 && norm(v2.(x)) <= eps1
    end

    model_nodes = DiscreteModel(Polytope{0}, model)
    cell_nodes_coords = get_cell_coordinates(model_nodes)
    cell_node_centre = collect1d(lazy_map(is_centre, cell_nodes_coords))
    cell_node = findall(cell_node_centre)
    new_entity = num_entities(labels) + 1
    for centre_point in cell_node
      labels.d_to_dface_to_entity[1][centre_point] = new_entity
    end
    add_tag!(labels, "centre", [new_entity])

    #ANALITICAL SOLUTION, used also for initial condition
    Tx(x, t) = pi / D * (x[1] - Ua * t)
    Ty(x, t) = pi / D * (x[2] - Va * t)
    Et(t) = exp(-(2 * ν * t * pi^2) / (D^2))
    ua(x, t) = Ua - Vs * cos(Tx(x, t)) * sin(Ty(x, t)) * Et(t)
    va(x, t) = Va + Vs * sin(Tx(x, t)) * cos(Ty(x, t)) * Et(t)
    velocity(x, t) = VectorValue(ua(x, t), va(x, t))
    pa(x, t) = -(Vs^2 / 4) * (cos(2 * Tx(x, t)) + cos(2 * Ty(x, t))) * Et(t)^2
    ωa(x, t) = 2 * Vs * pi / D * cos(Tx(x, t)) * cos(Ty(x, t)) * Et(t)^2

    ω₀ = 2 * Vs * pi / D

    ua(t::Real) = x -> ua(x, t)
    va(t::Real) = x -> va(x, t)
    velocity(t::Real) = x -> velocity(x, t)
    pa(t::Real) = x -> pa(x, t)
    ωa(t::Real) = x -> ωa(x, t)



    reffeᵤ = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
    V = TestFESpace(model, reffeᵤ, conformity=:H1)
    reffeₚ = ReferenceFE(lagrangian, Float64, order)
    Q = TestFESpace(model, reffeₚ, conformity=:H1, dirichlet_tags="centre")

    U = TrialFESpace(V)
    P = TransientTrialFESpace(Q, pa)



    Y = MultiFieldFESpace([V, Q])
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

    #h = lazy_map(h->h^(1/2),get_cell_measure(Ω))



    # Momentum residual, without the viscous term
    Rm(t, (u, p)) = ∂t(u) + (∇(u))' ⋅ u + ∇(p) - hf

    # Continuity residual
    Rc(u) = ∇ ⋅ u


    function τ(u)

      h = 2 * D / N #Because Mesh elements have all the same dimension, if not τ(u,h)
      r = 1 # r=1, r = 2 Tezduyar
      τ₂ = h^2 / (4 * ν)
      val(x) = x
      val(x::Gridap.Fields.ForwardDiff.Dual) = x.value
      u = val(norm(u))

      if iszero(u)
        return τ₂

      end
      τ₃ = dt / 2
      τ₁ = h / (2 * u)
      return 1 / (1 / τ₁^r + 1 / τ₂^r + 1 / τ₃^r)


    end


    #τb(u,h) = (u⋅u)*τ(u,h)
    τb(u) = (u ⋅ u) * τ(u)

    var_equations(t, (u, p), (v, q)) = ∫(
      ν * ∇(v) ⊙ ∇(u) # Viscous term
      + v ⊙ Rm(t, (u, p)) # Other momentum terms
      + q * Rc(u)
    )dΩ # Continuity


    stab_equations(t, (u, p), (v, q)) = ∫((τ ∘ (u) * (u ⋅ ∇(v)' + ∇(q))) ⊙ Rm(t, (u, p)) # First term: SUPG, second term: PSPG u⋅∇(v) + ∇(q)
                                          +
                                          τb ∘ (u) * (∇ ⋅ v) ⊙ Rc(u) # Bulk viscosity. Try commenting out both stabilization terms to see what happens in periodic and non-periodic cases
    )dΩ


    res(t, (u, p), (v, q)) = var_equations(t, (u, p), (v, q)) + stab_equations(t, (u, p), (v, q))


    op = TransientFEOperator(res, X, Y)
    nls = NLSolver(show_trace=true, method=:newton, linesearch=MoreThuente(), iterations=30, ftol=1e-8)

    solver = FESolver(nls)




    U0 = U(0.0)
    P0 = P(0.0)
    X0 = X(0.0)

    #Initial Conditions
    uh0 = interpolate_everywhere(velocity(0), U0)
    ph0 = interpolate_everywhere(pa(0), P0)
    xh0 = interpolate_everywhere([uh0, ph0], X0)

    vuh0 = interpolate_everywhere(VectorValue(0, 0), U0)
    vph0 = interpolate_everywhere(0, P0)
    vxh0 = interpolate_everywhere([vuh0, vph0], X0)





    ode_solver = GeneralizedAlpha(nls, dt, ρ∞)
    sol_t = solve(ode_solver, op, (xh0, vxh0), t0, tF)




    global _t_nn, iteration, e_u, e_p, err
    global iteration, Ntimestep, N
    _t_nn = t0

    iteration = 0
    e_u = 10
    e_p = 10


    createpvd(parts, "TV_2d") do pvd
      for (xh_tn, tn) in sol_t
        global _t_nn
        _t_nn += dt
        global iteration
        global N_timestep
        global N
        iteration += 1
        println("it_num = $iteration of $Ntimestep Mesh = $N\n")
        uh_tn = xh_tn[1]
        ph_tn = xh_tn[2]
        """
        ωh_tn = ∇ × uh_tn
        ωn = ωh_tn ./ ω₀
        Δu = Δ(uh_tn)
        p_analytic = pa(_t_nn)
        u_analytic = velocity(_t_nn)
        w_analytic = ωa(_t_nn)
        """
        flush(stdout)
        if isapprox(_t_nn, tF; atol=0.1 * dt)
          global e_u
          global e_p
          eu = velocity(tF) - uh_tn
          ep = pa(tF) - ph_tn
          e_un = sqrt(sum(∫(eu ⋅ eu) * dΩ))
          e_ur = sqrt(sum(∫(velocity(tF) ⋅ velocity(tF)) * dΩ))
          e_pn = sqrt(sum(∫(ep * ep) * dΩ))
          e_pr = sqrt(sum(∫(pa(tF) * pa(tF)) * dΩ))

          e_u = e_un / e_ur
          e_u = e_pn / e_pr
        end
      end

    end
    err = [e_u, e_p] #error on velocity and on pressure

    @save "TG_CFL_0p32_AlphaM_rho_0p8_order_1_mesh_$N.jld2" err

  end
end


partition = (2, 2)
prun(TV1, mpi, partition)