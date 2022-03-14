using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree
using StatsBase, Distributions
using Plots, StatsPlots
using Serialization
using SmoothTree: EPABCIS, ep!
theme(:wong2)
default(gridstyle=:dot, legend=false, framestyle=:box, title_loc=:left, titlefont=7)

mltrees = readnw.(readlines("docs/data/yeast-rokas/yeast.mltrees"))

mbdata = deserialize("docs/data/yeast-rokas/mb.jls")
spmap  = clademap(first(mbdata[1])[1])
root   = rootclade(spmap)
ntaxa  = cladesize(root)

μ, V = log(2.), 5.
Sprior = CCD(SplitCounts(root), BetaSplitTree(-1.5, ntaxa))
ϕprior = BranchModel(root, gaussian_mom2nat([μ, V]))
model  = MSCModel(Sprior, ϕprior)

data1 = Locus.(mltrees, Ref(spmap), prior=BetaSplitTree(-1.5, ntaxa), α=1e-6)
data2 = Locus.(mbdata,  Ref(spmap), prior=BetaSplitTree(-1.5, ntaxa), α=1e-6)

alg1   = EPABCIS(data1, model, 50000, target=200, miness=10., λ=0.1, α=1e-4, c=0.95)
trace1 = ep!(alg1, 5)

alg2   = EPABCIS(data2, model, 50000, target=200, miness=10., λ=0.1, α=1e-4, c=0.95)
trace2 = ep!(alg2, 5)

randtree(alg1.model.S, spmap, 10000) |> ranking
randtree(alg2.model.S, spmap, 10000) |> ranking

plot(getfield.(trace1, :ev), legend=:topleft)
plot(getfield.(trace2, :ev), legend=:topleft)

using SmoothTree: relabel, maptree
t1 = relabel(maptree(alg1.model), spmap)
t2 = relabel(maptree(alg2.model), spmap)
p1 = plot(t1, transform=true, scalebar=10.)
p2 = plot(t2, transform=true, scalebar=10.)
plot(p1, p2)
    
M  = alg.model
bs = SmoothTree.getbranchapprox(M.ϕ, randsplits(M.S))
ps = map(bs) do x
    plot(Normal(μ, √V), fill=true, color=:lightgray)
    plot!(x[end], color=:black)
end |> xs->plot(xs...)


# SIS
using SmoothTree: EPABCSIS
alg = EPABCSIS(data2, model, 10000, 5, target=200, miness=10., λ=0.1, α=1e-4, c=0.95)
trace = ep!(alg, 2)

randtree(alg.model.S, spmap, 10000) |> ranking


# Unrooted
# we cannot easily identify the root from a collection of unrooted trees it
# appears.
data3 = Locus.(mbdata,  Ref(spmap), prior=BetaSplitTree(-1.5, ntaxa), α=1e-6, 
               rooted=false)
Sprior = CCD(SplitCounts(root), BetaSplitTree(-1.99, ntaxa))
ϕprior = BranchModel(root, [0., -0.5])
model  = MSCModel(Sprior, ϕprior)
alg3   = EPABCIS(data3, model, 50000, target=200, miness=10., λ=0.1, α=1e-4, c=0.95)
trace3 = ep!(alg3, 2)
