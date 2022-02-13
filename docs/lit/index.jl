# SmoothTree
#
# Documentation/examples coming soon
#
# # Example: simulated data
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree

# Simulate a species tree topology from a beta-splitting distribution. We set
# β=-1.
ntaxa = 7
root = rootclade(ntaxa)
tree_distribution = MomMBM(root, BetaSplitTree(-1., ntaxa))
S = randtree(tree_distribution)

# We set all branch lengths to 1. in this example
SmoothTree.setdistance!(S, 1.)

# We need a map associating with each clade its name. For this simulated data
# this is a bit silly.
spmap = SmoothTree.clademap(S)

# simulate gene trees
M = SmoothTree.MSC(S, spmap)
N = 100
G = randtree(M, spmap, N)
ranking(G)

# Now put them in the relevant data structure for doing inference. From here
# on the code should be like for an analysis of actual empirical data.
data = SmoothTree.Locus.(G, Ref(spmap), 1/(2^ntaxa -1), -1.)

# use a Beta-splitting prior
Sprior = NatMBM(root, BetaSplitTree(-1., ntaxa))

# use information from observed gene trees for the prior
# Sprior = NatMBM(CCD(G, spmap), BetaSplitTree(-1., ntaxa), 10.)

# branch length prior
θprior = BranchModel(root, gaussian_mom2nat([0., 2]))

# the species tree model
model = MSCModel(Sprior, θprior)

alg = SmoothTree.EPABCIS(data, model, 10000, target=1000, miness=1.1, prunetol=1e-5)

trace = ep!(alg, 3)

relabel.(randtree(alg.model.S, 1000), Ref(spmap)) |> ranking
SmoothTree.topologize(S)
