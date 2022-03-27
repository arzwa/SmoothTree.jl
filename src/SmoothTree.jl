module SmoothTree

using NewickTree, StatsBase, Parameters, Random, StatsFuns, Distributions
using Printf, ProgressBars, SpecialFunctions, LinearAlgebra
import Distributions: logpdf

export clademap, rootall!, rootclade, cladesize, ranking, getclade, getsplits
export SplitCounts, SplitCountsUnrooted
export BetaSplitTree
export CCD, randsplits, randtree, randbranches
export BranchModel, MSCModel
export Locus
export EPABCIS, EPABCSIS, ep!
export gaussian_mom2nat
export maptree, relabel

include("bimap.jl")
include("utils.jl")
include("splitmodel.jl")
include("ccd.jl")
include("algebra.jl")
include("msc.jl")
include("branchmodel.jl")
include("mscmodel.jl")
include("mvgaussian.jl")
include("locus.jl")
include("mullocus.jl")
include("epabc.jl")
#include("pps.jl")

end # module

