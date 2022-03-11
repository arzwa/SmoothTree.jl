module SmoothTree

using NewickTree, StatsBase, Parameters, Random, StatsFuns, Distributions
using Printf, ProgressBars, SpecialFunctions, ThreadTools
import Distributions: logpdf

export clademap, rootall, rootclade, cladesize, ranking
export SplitCounts, SplitCountsUnrooted
export BetaSplitTree
export CCD, randsplits, randtree, randbranches
export BranchModel, MSCModel
export Locus
#export CCD, MSC, Locus, randtree, randsplits, randbranches, traceback
#export setdistance!, setdistance_internal!, maptree
#export NatMBM, MomMBM, MSCModel, BranchModel, ep!, pep!, EPABC, EPABCIS
#export ranking, clademap, BetaSplitTree, cladesize, splitsize, rootclade
#export relabel, gaussian_mom2nat, topologize

include("bimap.jl")
include("utils.jl")
include("splitmodel.jl")
include("ccd.jl")
include("algebra.jl")
include("msc.jl")
include("branchmodel.jl")
include("mscmodel.jl")
include("locus.jl")
include("epabc.jl")
#include("pps.jl")

end # module

