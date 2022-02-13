module SmoothTree

using NewickTree, StatsBase, Parameters, Random, StatsFuns, Distributions
using Printf, ProgressBars, SpecialFunctions, ThreadTools
import Distributions: logpdf
export CCD, MSC, randtree, randsplits, setdistance!, setdistance_internal!
export NatMBM, MomMBM, MSCModel, BranchModel, traceback, ep!, pep!, EPABC
export ranking, clademap, BetaSplitTree, cladesize, splitsize, rootclade
export relabel, gaussian_mom2nat

include("trees.jl")
include("utils.jl")
include("betasplit.jl")
include("ccd.jl")
include("locus.jl")
include("msc.jl")
include("sparsesplits.jl")
include("mbm.jl")
include("branchmodel_.jl")
include("mscmodel_.jl")
include("epabc_.jl")
#include("epabc-is.jl")
include("pps.jl")

end # module

