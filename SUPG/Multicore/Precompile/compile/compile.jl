using PackageCompiler
create_sysimage(:Channel,
  sysimage_path=joinpath(@__DIR__,"..","Taylor_Green_Vortex_SUPG.so"),
  precompile_execution_file=joinpath(@__DIR__,"..","src", "Taylor_Green_Vortex_SUPG.jl"))