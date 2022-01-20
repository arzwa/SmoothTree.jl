using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree
using StatsBase, Distributions, Plots, StatsPlots
using Serialization
default(gridstyle=:dot, legend=false, framestyle=:box,
        title_loc=:left, titlefont=7)
data_ = deserialize("docs/data/yeast-rokas/ccd-ufboot.jls")
trees = readnw.(readlines("docs/data/yeast.mltrees"))
tmap = SmoothTree.taxonmap(trees, UInt16)

# we can compare using ML gene trees against acknowledging uncertainty
#data  = CCD.(data_, lmap=tmap)  
data  = CCD.(trees, lmap=tmap)
alg = SmoothTree.epabc(data, tmap, β=-1.5, h=0., V=2.)

trace = pep!(alg, 1)
trace = ep!(alg, 3)

# posterior inspection
pS = SmoothTree.MomMBM(alg.model.S)
smple = SmoothTree.ranking(randtree(pS, 10000))
tr = SmoothTree.relabel(first(smple)[1], tmap)

X, Y = traceback(trace)

pq = trace[end].q
p = plot()
for (k,v) in pq.cmap
    isnan(v[1]) && continue
    m, V = SmoothTree.gaussian_nat2mom(v...)
    plot!(Normal(m, √V))
end
plot(p)

ultimate_clades = map(SmoothTree.getclades(tr)) do x
    UInt16(sum([tmap[y] for y in name.(x)]))
end
ultimate_clades = filter(x->!SmoothTree.isleafclade(x), ultimate_clades)
traces = map(ultimate_clades[2:end]) do clade
    plot(plot(X[clade]), plot(Y[clade]))
end |> x->plot(x..., size=(900,200))

obs = ranking(trees)
pps = map(1:1000) do rep
    S = SmoothTree.randtree(trace[end])
    M = SmoothTree.MSC(S, Dict(id(n)=>[id(n)] for n in getleaves(S)))
    pps = proportionmap(randtree(M, tmap, length(trees)))
    xs = map(x->haskey(pps, x[1]) ? pps[x[1]] : 0., obs)
end |> x->permutedims(hcat(x...))

boxplot(pps, linecolor=:black, fillcolor=:lightgray, outliers=false)
scatter!(last.(obs), color=:black)

# the posterior predictive distribution for gene trees differs strongly.
# here we see that when we acknowledge gene tree uncertainty, we
# estimate much less true discordance... posterior predictive
# simulation suggest that almost 90% of the *true* gene trees matches
# the species tree.

