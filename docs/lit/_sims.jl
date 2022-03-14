
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, Distributions
using Plots, StatsPlots, Measures
default(gridstyle=:dot, legend=false, title_loc=:left, titlefont=8, framestyle=:box)

# The species tree topology we will use:
species = string.('A':'Z')[1:16]
spmap = clademap(species, UInt64)
root = sum(keys(spmap))
ntaxa = length(spmap)
S = randtree(CCD(SplitCounts(root), BetaSplitTree(-1., ntaxa)), spmap)

# We set some random branch lengths
l = SmoothTree.n_internal(S)
d = MixtureModel([LogNormal(log(0.5), 0.5), LogNormal(log(3.), .5)], [0.2,0.8])
#d = Gamma(3., 1/2)
θ = rand(d, l)
SmoothTree.setdistance_internal!(S, θ)

# plot the species tree
#plot(S, transform=true, scalebar=1)

# We need a map associating with each clade its name. 

# simulate gene trees
M = SmoothTree.MSC(S, spmap)
N = 100
G = randtree(M, spmap, N)
ranking(G)

# Now put them in the relevant data structure for doing inference. From here
# on the code should be like for an analysis of actual empirical data.
data = Locus.(G, Ref(spmap), prior=BetaSplitTree(-1., ntaxa), α=1e-2)

# use a Beta-splitting prior
Sprior = CCD(SplitCounts(unique(G), spmap), BetaSplitTree(-1., ntaxa), 20.)
prsample = randtree(Sprior, spmap, 10000) |> ranking

# branch length prior
μ = 1.5; V = 2.
tips = collect(keys(spmap))
θprior = BranchModel(root, gaussian_mom2nat([log(μ), V]), inftips=tips)

# the species tree model
model = MSCModel(Sprior, θprior)
alg = SmoothTree.EPABCIS(data, model, 50000, target=100, miness=10.,
                         prunetol=1e-6, α=1e-3, c=0.95)

trace = ep!(alg, 4)
