module SmoothTree

using NewickTree, StatsBase, Parameters, Random, StatsFuns, Distributions
using Printf, ProgressBars
import Distributions: logpdf
export CCD, MSC, randtree, randsplits, setdistance!, setdistance_internal!

include("utils.jl")
include("ccd.jl")
include("bmp.jl")
include("msc.jl")
include("nni.jl")
include("epabc.jl")

# assumed throughout
_spname(x::String) = string(split(x, "_")[1])
_spname(n::Node) = _spname(name(n))

# ranking
ranking(xs) = sort(collect(proportionmap(xs)), by=last, rev=true)

# remove branch lengths
topologize!(tree) = for n in postwalk(tree); n.data.distance = NaN; end

end # module

