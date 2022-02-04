module SmoothTree

using NewickTree, StatsBase, Parameters, Random, StatsFuns, Distributions
using Printf, ProgressBars, SpecialFunctions
import Distributions: logpdf
export CCD, MSC, randtree, randsplits, setdistance!, setdistance_internal!
export NatMBM, MomMBM, MSCModel, BranchModel, traceback, ep!, pep!, EPABC
export ranking, taxonmap, BetaSplitTree, cladesize, splitsize, rootclade
export relabel, gaussian_mom2nat

include("utils.jl")
include("betasplit.jl")
include("ccd.jl")
include("msc.jl")
include("sparsesplits.jl")
include("mbm.jl")
include("branchmodel.jl")
include("mscmodel.jl")
include("epabc.jl")
include("pps.jl")

# this function sets up the main style of analysis
# note that h is the smoothing parameter for the input data
function epabc(data, tmap; β=-1.5, a=0., μ=1., V=2., kwargs...)
    T = keytype(tmap)
    ntax = length(tmap)
    root = T(sum(keys(tmap)))
    bsd  = BetaSplitTree(β, ntax)
    if a > 0.
        data = MomMBM.(data, Ref(bsd), h)
    end
    Sprior = NatMBM(root, bsd)
    θprior = BranchModel(T, gaussian_mom2nat([log(μ), V]))
    model = MSCModel(Sprior, θprior, tmap)
    alg = EPABC(data, model; kwargs...)
end

end # module

