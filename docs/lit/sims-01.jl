using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree
using StatsBase, Distributions, Plots
using Serialization
using LaTeXStrings, StatsPlots
default(gridstyle=:dot, legend=false, framestyle=:box, title_loc=:left, titlefont=7)

function getcladeθ(S, m)
    d = Dict()
    function walk(n)
        isleaf(n) && return m[name(n)]
        a = walk(n[1])
        b = walk(n[2])
        d[a+b] = distance(n)
        return a+b
    end
    walk(S)
    return d
end

S = nw"(((((A,B),C),(D,E)),(F,(G,H))),O);"
S = nw"((((((((A,B),C),(D,E)),(F,(G,H))),I),(J,K)),L),O);"
#S = readnw(readline("docs/data/mammals-song/mltrees.nw"))
#S = readnw(nwstr(S[1][1][2][1][1]), UInt64)
ntaxa = length(getleaves(S))
m = SmoothTree.n_internal(S)
θ = exp.(rand(Normal(1,1), m))
SmoothTree.setdistance_internal!(S, θ)
m = taxonmap(S)
d = getcladeθ(S, m)
M = SmoothTree.MSC(S, m)
n = 100
G = randtree(M, m, n)
ranking(G)

#Sprior = NatBMP(CCD(G, lmap=m, α=10.))
Sprior = NatBMP(CCD(G, lmap=m, α=0.1))
#Sprior = NatBMP(CCD(G, lmap=m, α=0.01))
smple  = ranking(randtree(MomBMP(Sprior), 10000))
θprior = [SmoothTree.gaussian_mom2nat(log(1.), 5.)...]

data  = CCD.(G, lmap=m, α=1/2^(ntaxa-1))
model = MSCModel(Sprior, θprior, m)
alg   = EPABC(data, model, λ=0.2, α=1/2^(ntaxa-1))

_ = SmoothTree.ep_iteration!(alg, 1, maxn=1e5, mina=5, target=100, noisy=true)

# EP
trace = ep!(alg, 3, maxn=1e5, mina=5, target=100)

# Analysis
X, Y = traceback(trace)

xs = filter(x->size(x[2], 2) > 1, collect(X))
map(xs) do (k, x)
    p1 = plot(x, title="clade $(bitstring(k)[end-7:end])", xscale=:log10)
    p2 = plot(Y[k])
    plot(p1, p2)
end |> x-> plot(x..., size=(1200,500))

smple  = ranking(randtree(SmoothTree.MomBMP(trace[end].S), 10000))

SmoothTree.relabel(first(smple)[1], m)

mapS = first(smple)[1]
clades = filter(n->!SmoothTree.isleafclade(n), id.(postwalk(mapS)))[1:end-1]
map(clades) do g
    lm, V = Y[g][end,:]
    plot(Normal(lm, √V))
    vline!([log(d[g])])
end |> x->plot(x...)

obs = ranking(G)
pps = map(1:1000) do rep
    S = SmoothTree.randsptree(trace[end])
    M = SmoothTree.MSC(S, Dict(id(n)=>[id(n)] for n in getleaves(S)))
    pps = proportionmap(randtree(M, m, length(G)))
    xs = map(x->haskey(pps, x[1]) ? pps[x[1]] : 0., obs)
end |> x->permutedims(hcat(x...))
boxplot(pps, linecolor=:black, fillcolor=:lightgray, outliers=false)
scatter!(last.(obs), color=:black)

