using Gridap
using Gridap.Fields
using Gridap.CellData
using FillArrays

q_vertex = Point{2, Float64}[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
cell_q_vertex = Fill(q_vertex,num_cells(Ω))
x_Ω_vertex = CellPoint(cell_q_vertex,Ω,ReferenceDomain())

q_mean = Point{2, Float64}[(0.5, 0.0), (0.5, 0.0), (0.5, 1), (1, 0.5)]
cell_q_mean = Fill(q_mean,num_cells(Ω))
x_Ω_mean = CellPoint(cell_q,Ω,ReferenceDomain())

ffun(x) = Vector[x[2]-x[1], x[3] - x[2], x[4] - x[3], x[1] - x[4]]
ffun(x) = VectorValue(x[1],x[2])
f = GenericField(ffun)
cell_f = Fill(f,num_cells(Ω))
f_Ω = GenericCellField(cell_f,Ω,PhysicalDomain())
fx_Ω = f_Ω(x_Ω_vertex)


h1 = VectorValue(1, 2)
