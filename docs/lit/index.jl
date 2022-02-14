# SmoothTree
#
# Documentation/examples coming soon
#
# # Example: simulated data
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree
using Random; Random.seed!(123)

# The species tree topology we will use:
S = nw"(A,(B,((C,D),((E,F),(G,H)))));"

# We set all branch lengths to 1. in this example
SmoothTree.setdistance!(S, 2.)

# We need a map associating with each clade its name. 
spmap = SmoothTree.clademap(S)
ntaxa = length(spmap)

# simulate gene trees
M = SmoothTree.MSC(S, spmap)
N = 100
G = randtree(M, spmap, N)
ranking(G)

# Now put them in the relevant data structure for doing inference. From here
# on the code should be like for an analysis of actual empirical data.
data = SmoothTree.Locus.(G, Ref(spmap), 1/(2^ntaxa -1), -1.)

# use a Beta-splitting prior
root = UInt16(2^ntaxa-1)
Sprior = NatMBM(root, BetaSplitTree(-1., ntaxa))

# use information from observed gene trees for the prior
# Sprior = NatMBM(CCD(G, spmap), BetaSplitTree(-1., ntaxa), 10.)

# branch length prior
θprior = BranchModel(root, gaussian_mom2nat([0., 1]))

# the species tree model
model = MSCModel(Sprior, θprior)

alg = SmoothTree.EPABCIS(data, model, 5000, target=2000, miness=1., prunetol=1e-9)

trace = ep!(alg, 3, rnd=false)

# Numerical issues due to variance decreasing to zero (and becoming negative)
# possible causes:
# 1. randbranches
# 2. randsplits
# 3. something with weights
# 4. matchmoments

relabel(first(randtree(alg.model.S, 1000)), spmap)

SmoothTree.topologize(S)
