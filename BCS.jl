labels = get_face_labeling(model)
h0 = 2*D/N
h1 = h0 + 0.1*h0
function is_in_area(coords)
    n = length(coords)
    x = (1/n)*sum(coords)
    print(coords)
    x[1] <= h1  && x[1] >= -h1 && x[2] <= h1  && x[2] >= -h1
   end

Dims = num_cell_dims(model)
model_faces = DiscreteModel(Polytope{D-1},model)
cell_face_coords = get_cell_coordinates(model_faces)
cell_face_to_is_in_area = collect1d(lazy_map(is_in_area, cell_face_coords))
cell_faces_in_area = findall(cell_face_to_is_in_area)
new_entity = num_entities(labels) + 1
for face in cell_faces_in_area
 labels.d_to_dface_to_entity[Dims][face] = new_entity
end
add_tag_from_tags!(labels,"centerq",[new_entity])
