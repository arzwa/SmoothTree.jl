module SmoothTree

using NewickTree, StatsBase, Parameters, Random, StatsFuns, Distributions
using Printf, ProgressBars, SpecialFunctions
import Distributions: logpdf
export CCD, MSC, randtree, randsplits, setdistance!, setdistance_internal!
export NatMBM, MomMBM, MSCModel, BranchModel, traceback, ep!, pep!, EPABC
export ranking, taxonmap, BetaSplitTree, cladesize, splitsize

include("utils.jl")
include("betasplit.jl")
include("ccd.jl")
include("msc.jl")
include("sparsesplits.jl")
include("mbm.jl")
include("branchmodel.jl")
include("mscmodel.jl")
include("epabc.jl")

end # module

