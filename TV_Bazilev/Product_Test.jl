using Gridap

gv = TensorValue(1, 2, 2, 1)
rmv = VectorValue(4, 5)
tmm = VectorValue(8.0)

 trv = 3 * rmv

gv⋅outer(trv,trv)


v1 = [1, 2]
v2 = [3, 4]
v1 .* v2