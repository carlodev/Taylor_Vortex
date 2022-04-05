
"""
h_compute2D(model, velocity)

compute the dimension of the cell of a given model, now works in 2D

[tau_su, tau_ps, tau_bk] = stabilization_coefficients(h_element,nu,velocity;r=2)

"""

function stabilization_coefficients(t)
    n_node = length(model.grid.node_coords) # number of node_coordinate
    dim = length(model.grid.node_coords[1]) #dimension, 1D, 2D, 3D
    n_ele = length(model.grid.cell_node_ids) # number of elements
    node_per_element = length(model.grid.cell_node_ids[1])
    h_element = zeros(n_ele)
    u_mid = zeros(n_ele, 2)

    #2D case
    for i = 1:1:n_ele
        j = 1
        l = 0
        while (j <= node_per_element)
            actual_node_coordinates = model.grid.node_coords[model.grid.cell_node_ids[i][j]]
            k = j + 1
            if j < node_per_element
                k = j + 1
            else
                k = 1
            end
            next_node_coordinates = model.grid.node_coords[model.grid.cell_node_ids[i][k]]
            node_mean = 0.5 .* (actual_node_coordinates + next_node_coordinates)

            u_mid[i, 1] = velocity(node_mean, t)[1]
            u_mid[i, 2] = velocity(node_mean, t)[2]

            l = norm((actual_node_coordinates - next_node_coordinates) ⋅ u_mid[i, :]) / norm(u_mid[i, :])


            h_element[i] = l


            j = j + 1

        end

    end
    r = 2
    tau_su = zeros(n_ele)
    tau_ps = zeros(n_ele)
    tau_bk = zeros(n_ele)
    for i = 1:1:n_ele
        u = norm(u_mid[i, :])
        h = h_element[i]
        tau_su_adv = h / (2 * nu)
        tau_su_diff = h * h / (4 * nu)
        tau_su_unst = h / (2 * u)

        tau_ps_adv = h / (2 * nu)
        tau_ps_diff = h * h / (4 * nu)

        tau_su[i] = (tau_su_adv^(-r) + tau_su_diff^(-r) + tau_su_unst^(-r))^(-1 / r)
        tau_ps[i] = (tau_ps_adv^(-r) + tau_ps_diff^(-r))^(-1 / r)
        tau_bk[i] = (u^2) * tau_ps[i]
    end

    return (tau_su, tau_ps, tau_bk)
end
tau_su(t) = stabilization_coefficients(t)[1]
tau_ps(t) = stabilization_coefficients(t)[2]
tau_bk(t) = stabilization_coefficients(t)[3]

