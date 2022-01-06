using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree
using StatsBase, Distributions, Plots, StatsPlots
using Serialization
default(gridstyle=:dot, legend=false, framestyle=:box, title_loc=:left, titlefont=7)
data_ = deserialize("docs/data/yeast-rokas/ccd-ufboot.jls")
trees = readnw.(readlines("docs/data/yeast.mltrees"))
tmap = SmoothTree.taxonmap(trees[1])

root = UInt16(sum(keys(tmap)))
Sprior = NatBMP(root, tmap["Calb"])
#Sprior = NatBMP(CCD(trees, lmap=tmap, α=5.))
smple  = ranking(randtree(MomBMP(Sprior), 10000))
θprior = [SmoothTree.gaussian_mom2nat(log(1.), 5.)...]

# we can compare using ML gene trees against acknowledging uncertainty
data  = CCD.(data_, lmap=tmap, α=.01)
#data  = CCD.(trees, lmap=tmap, α=.0)
model = MSCModel(Sprior, θprior, tmap)
alg   = EPABC(data, model, λ=0.05, α=1e-9)

trace = ep!(alg, 2, maxn=1e4, mina=100, target=100, fillup=true)

X, Y = traceback(trace)

xs = filter(x->size(x[2], 2) > 1, collect(X))
map(xs) do (k, x)
    p1 = plot(x, title="clade $(bitstring(k)[end-7:end])")
    p2 = plot(Y[k])
    plot(p1, p2)
end |> x-> plot(x..., size=(1200,500))

pq = trace[end].q
p = plot()
for (k,v) in pq.cmap
    v[1] == NaN && continue
    m, V = SmoothTree.gaussian_nat2mom(v...)
    plot!(Normal(m, √V))
end
plot(p)

pS = SmoothTree.MomBMP(trace[end].S)
smple = SmoothTree.ranking(randtree(pS, 10000))
SmoothTree.relabel(last(smple)[1], tmap)

maxtree = SmoothTree.ranking(randtree(CCD(trees), 1000))[1][1]
ultimate_clades = map(SmoothTree.getclades(maxtree)) do x
    UInt16(sum([tmap[y] for y in name.(x)]))
end
ultimate_clades = filter(x->!SmoothTree.isleafclade(x), ultimate_clades)
traces = map(ultimate_clades[2:end]) do clade
    plot(plot(X[clade]), plot(Y[clade]))
end |> x->plot(x..., size=(900,200))


obs = ranking(trees)
pps = map(1:1000) do rep
    S = SmoothTree.randsptree(trace[end])
    M = SmoothTree.MSC(S, Dict(id(n)=>[id(n)] for n in getleaves(S)))
    pps = proportionmap(randtree(M, tmap, length(trees)))
    xs = map(x->haskey(pps, x[1]) ? pps[x[1]] : 0., obs)
end |> x->permutedims(hcat(x...))

boxplot(pps, linecolor=:black, fillcolor=:lightgray, outliers=false)
scatter!(last.(obs), color=:black)



scatter(xs, color=:lightgray, size=(400,400)); plot!(x->x, color=:black, ls=:dot)

# the posterior predictive distribution for gene trees differs strongly.
# here we see that when we acknowledge gene tree uncertainty, we
# estimate much less true discordance... posterior predictive
# simulation suggest that almost 90% of the *true* gene trees matches
# the species tree.

