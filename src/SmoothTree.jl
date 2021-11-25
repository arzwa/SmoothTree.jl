module SmoothTree

using NewickTree, StatsBase, Parameters, Random, StatsFuns, Distributions
import Distributions: logpdf
export CCD

include("ccd.jl")
include("msc.jl")
include("rearrangement.jl")
include("epabc.jl")
#include("mulccd.jl")
#include("ep-abc-msc.jl")

# assumed throughout
_spname(x::String) = string(split(x, "_")[1])
_spname(n::Node) = _spname(name(n))

end # module

