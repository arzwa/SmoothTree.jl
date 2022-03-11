using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree
using StatsBase, Distributions
using Serialization

mbdata = deserialize("docs/data/yeast-rokas/mb.jls")
spmap  = clademap(first(mbdata[1])[1])
root   = rootclade(spmap)
ntaxa  = cladesize(root)

Sprior = CCD(SplitCounts(root), BetaSplitTree(-1.5, ntaxa))
ϕprior = BranchModel(root, [0., -0.5])
model  = MSCModel(Sprior, ϕprior)

data = Locus.(mbdata, Ref(spmap), prior=BetaSplitTree(-1.5, ntaxa), α=0.1)

alg = SmoothTree.EPABCIS(data, model, 50000, target=100, miness=10.,
                         prunetol=1e-6, α=1e-3, c=0.95)

trace = SmoothTree.ep!(alg, 2)



