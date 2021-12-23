using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree
using StatsBase, Distributions, Plots
using Serialization
default(gridstyle=:dot, legend=false, framestyle=:box, title_loc=:left, titlefont=7)
data_ = deserialize("docs/data/yeast-rokas/ccd-ufboot.jls")
trees = readnw.(readlines("docs/data/yeast.mltrees"))
tmap = SmoothTree.taxonmap(trees[1])

Sprior = NatBMP(CCD(trees, lmap=tmap, α=5.))
smple  = ranking(randtree(MomBMP(Sprior), 10000))
θprior = [SmoothTree.gaussian_mom2nat(log(1.), 5.)...]

data  = CCD.(data_, lmap=tmap, α=.1)
#data  = CCD.(trees, lmap=tmap, α=.0)
model = MSCModel(Sprior, θprior)
alg   = EPABC(data, model, λ=0.1, α=1e-9)

trace = ep!(alg, 2, maxn=1e5, mina=10, target=100)

X, Y = traceback(trace)

xs = filter(x->size(x[2], 2) > 1, collect(X))
map(xs) do (k, x)
    p1 = plot(x, title="clade $(bitstring(k)[end-7:end])")
    p2 = plot(Y[k])
    plot(p1, p2)
end |> x-> plot(x..., size=(1200,500))

smple  = SmoothTree.ranking(randtree(SmoothTree.MomBMP(trace[end].S), 10000))

maxtree = SmoothTree.ranking(randtree(CCD(trees), 1000))[1][1]
ultimate_clades = map(SmoothTree.getclades(maxtree)) do x
    UInt16(sum([tmap[y] for y in name.(x)]))
end
ultimate_clades = filter(x->!SmoothTree.isleafclade(x), ultimate_clades)

traces = map(ultimate_clades[2:end]) do clade
    plot(X[clade], legend=false, framestyle=:box,
         gridstyle=:dot, title="clade $(bitstring(clade)[end-7:end])",
         title_loc=:left, titlefont=7)#, xscale=:log10)
end |> x->plot(x...)

S = SmoothTree.randsptree(trace[end])
M = SmoothTree.MSC(S, Dict(id(n)=>[id(n)] for n in getleaves(S)))
pps = proportionmap(randtree(M, tmap, 10000))
obs = proportionmap(trees)

xs = map(x->(x[2], haskey(pps, x[1]) ? pps[x[1]] : 0.), collect(obs))
scatter(xs, color=:lightgray, size=(400,400)); plot!(x->x, color=:black, ls=:dot)

