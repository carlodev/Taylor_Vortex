"""
In the Taylor-Green Vortex the central node of the domain is estracted in order to set it as a boundary condition for pressure
"""

labels = get_face_labeling(model)
h0 = 2*D/N
eps1 = h0/10

vt(i) = VectorValue(y,i)
v1(v) = v[1]
v2(v) = v[2]

function is_centre(x)
    norm(v1.(x)) <= eps1  &&   norm(v2.(x)) <= eps1 
end

model_nodes = DiscreteModel(Polytope{0},model)
cell_nodes_coords = get_cell_coordinates(model_nodes)
cell_node_centre = collect1d(lazy_map(is_centre, cell_nodes_coords))
cell_node = findall(cell_node_centre)
new_entity = num_entities(labels) + 1
for centre_point in cell_node
 labels.d_to_dface_to_entity[1][centre_point] = new_entity
end
add_tag!(labels,"centre", [new_entity])
