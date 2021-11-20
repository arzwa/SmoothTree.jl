module SmoothTree

using NewickTree, StatsBase, Parameters, Random, StatsFuns
export CCD

include("ccd.jl")
include("mulccd.jl")

end # module

