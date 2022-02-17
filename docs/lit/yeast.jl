using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree
using StatsBase, Distributions, Plots, StatsPlots, LaTeXStrings
using Serialization
default(gridstyle=:dot, legend=false, framestyle=:box, title_loc=:left, titlefont=7)
data_ = deserialize("docs/data/yeast-rokas/ccd-ufboot.jls")
trees = readnw.(readlines("docs/data/yeast.mltrees"))
tmap  = clademap(trees[1])

# we can compare using ML gene trees against acknowledging uncertainty
data = Locus.(data_, Ref(tmap))
#data = Locus.(trees, Ref(tmap), 1e-32, -1.5)

μ, V = log(2.), 1.
root   = UInt16(sum(keys(tmap)))
Sprior = NatMBM(CCD(unique(trees), tmap), BetaSplitTree(-1.5, cladesize(root)), 1.)
θprior = BranchModel(root, gaussian_mom2nat([μ, V]), inftips=collect(keys(tmap)))
model  = MSCModel(Sprior, θprior)
alg    = EPABCIS(data, model, 50000, target=100, miness=10., λ=0.1, α=1e-3)

trace = ep!(alg, 5)

# posterior inspection
smple = relabel.(randtree(alg.model.S, 10000), Ref(tmap)) |> ranking

xs = traceback(first.(trace))
Zs = getindex.(trace, 2)

plot(plot(xs.θ), plot(xs.μ), plot(xs.V), plot(Zs))

bs = SmoothTree.getbranchapprox(alg.model, randsplits(alg.model.S))
bs = filter(x->!SmoothTree.isleafclade(x[2]), bs)
map(bs) do (γ, δ, d)
    plot(LogNormal(μ, √V), fill=true, color=:lightgray, fillalpha=0.3)
    plot!(LogNormal(d.μ, d.σ), fill=true, fillalpha=0.3, color=:black, xlim=(0,20))
end |> x->plot(x..., size=(650,350))

pps = SmoothTree.postpredsim(alg.model, data, 1000)
obs = proportionmap(trees)
comp = [(haskey(obs, k) ? obs[k] : 0, v) for (k,v) in pps]
comp = filter(x->x[1] > 0 || mean(x[2]) > 0.001, comp)
pl3 = plot(ylabel=L"P", xlabel=L"G", xtickfont=7)
for (i,x) in enumerate(sort(comp, rev=true))
    plot!(pl3, [i, i], quantile(x[2], [0.025, 0.975]), color=:lightgray, lw=4)
    scatter!(pl3, [(i, x[1])], color=:black)
end
plot(pl3, size=(300,200))

boxplot(pps, linecolor=:black, fillcolor=:lightgray, outliers=false)
scatter!(last.(obs), color=:black)

# the posterior predictive distribution for gene trees differs strongly.
# here we see that when we acknowledge gene tree uncertainty, we
# estimate much less true discordance... posterior predictive
# simulation suggest that almost 90% of the *true* gene trees matches
# the species tree.

