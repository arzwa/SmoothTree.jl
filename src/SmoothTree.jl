module SmoothTree

using NewickTree, StatsBase, Parameters, Random, StatsFuns, Distributions
using Printf, ProgressBars
import Distributions: logpdf
export CCD, MSC, randtree, randsplits, setdistance!, setdistance_internal!
export NatBMP, MomBMP, MSCModel, BranchModel, traceback, ep!, EPABC
export ranking, taxonmap

include("utils.jl")
include("ccd.jl")
include("msc.jl")
include("sparsesplits.jl")
include("bmp.jl")
include("branchmodel.jl")
include("mscmodel.jl")
include("epabc.jl")

end # module

