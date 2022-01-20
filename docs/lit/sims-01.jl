using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree
using StatsBase, Distributions, Plots
using Serialization
using LaTeXStrings, StatsPlots, Measures
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

# Note, if one assumes θ ~ Gamma(α, 1/β), then E[exp(-θ)] = (1+1/β)^-k
# (from the mgf), which is approximately exp(-E[θ])

T = UInt16
#S = nw"(((A,B),C),(D,E));"
#S = nw"(((((A,B),C),(D,E)),(F,(G,H))),(I,J));"
S = nw"(((((((((A,B),C),(D,E)),(F,(G,H))),I),(J,K)),L),M),(O,P));"
#T = UInt64
#S = readnw("(((((((((((A,B),C),(D,E)),(F,(G,H))),I),(J,K)),L),M),(O,P)),Q),((R,S),T));", T)
#S = readnw(readline("docs/data/mammals-song/mltrees.nw"))
#T = UInt64
#S = readnw(nwstr(S[1][1][2][1]), T)
#S = readnw(nwstr(S[1][1][2][1]), T)
ntaxa = length(getleaves(S))
l = SmoothTree.n_internal(S)
#θ = rand(LogNormal(log(3.), 0.5), l)
θ = rand(Gamma(4., 1/2), l)
SmoothTree.setdistance_internal!(S, θ)
m = taxonmap(S, T)
d = getcladeθ(S, m)
M = SmoothTree.MSC(S, m)
N = 100
G = randtree(M, m, N)
ranking(G) .|> last

#Sprior = NatBMP(CCD(G, lmap=m, α=10.))
#Sprior = NatBMP(CCD(G, lmap=m, α=1.))
#Sprior = NatBMP(CCD(G, lmap=m, α=1e-4))
#Sprior = NatBMP(CCD(unique(G), α=0.1))
#smple  = ranking(randtree(MomBMP(Sprior), 10000))
root = T(2^ntaxa - 1)
bsd  = BetaSplitTree(-1.5, cladesize(root))
Sprior = NatMBM(root, bsd)
priormean = 1.
priorvar  = 2.
θprior = BranchModel(T, SmoothTree.gaussian_mom2nat([log(priormean), priorvar]))

data  = MomMBM.(CCD.(G, lmap=m), Ref(bsd), 1/2^(ntaxa-1))
model = MSCModel(Sprior, θprior, m)
alg   = EPABC(data, model, λ=0.1, α=1/2^(ntaxa-1), prunetol=1e-6,
              minacc=50, target=100)

# EP
trace = pep!(alg, 1)

trace = ep!(alg, 2)


smple = ranking(randtree(alg.model.S, 10000))
SmoothTree.topologize(S)
SmoothTree.relabel(first(smple)[1], m)

# Analysis
X1, X2 = traceback(trace)

#xs = filter(x->size(x[2], 2) > 1, collect(X1))
#map(xs) do (k, x)
#    p1 = plot(x, title="clade $(bitstring(k)[end-7:end])")#, xscale=:log10)
#    p2 = plot(X2[k])
#    plot(p1, p2)
#end |> x-> plot(x..., size=(1200,500))

# log scale
mapS = first(smple)[1]
clades = filter(n->!SmoothTree.isleafclade(n), id.(postwalk(mapS)))[1:end-1]
pls = map(clades) do g
    plot(Normal(log(priormean), √priorvar), color=:lightgray,
         fill=true, fillalpha=0.8, xlim=(-4.5,4.5), yticks=false, grid=false)
    for model in trace[1:20:end]
        lm, V = SmoothTree.gaussian_nat2mom(model.q[g])
        plot!(Normal(lm, √V), color=:black)
    end
    vline!([log(d[g])], color=:black, ls=:dot)
end 
#xlabel!(pls[13], L"\theta")
plot(pls...) #,layout=(3,5), size=(600,300))
#plot(p, bottom_margin=1.5mm)
#savefig("docs/img/17taxa-posterior.pdf")

# posterior prediction
pps = map(1:1000) do rep
    S = randtree(alg.model)
    M = SmoothTree.MSC(S, Dict(id(n)=>[id(n)] for n in getleaves(S)))
    pps = proportionmap(randtree(M, m, length(G)))
end

function merge_pmaps(xs)
    ks = union(keys.(xs)...)
    d = Dict(k=>zeros(length(xs)) for k in ks)
    for i in 1:length(xs)
        for (k,v) in xs[i]
            d[k][i] = v
        end
    end
    return d
end

pms = merge_pmaps(pps)

obs = proportionmap(G)
top = sort(collect(pms), by=x->mean(last(x)), rev=true)[1:50]
topts = Set(first.(top))
for (k, v) in obs
    if k ∉ topts
        if haskey(pms, k)
            push!(top, (k=>pms[k]))
        else
            push!(top, (k=>zeros(1)))
        end
    end
end

ppds = last.(top)
p = plot(xlabel="tree", ylabel="frequency") 
#boxplot(ppds, linecolor=:black, fillcolor=:lightgray, outliers=false)
map(enumerate(ppds)) do (i, ps)
    plot!(p, [i, i], quantile(ps, [0.025, 0.975]), color=:black)
end
for (i,k) in enumerate(first.(top))
    if haskey(obs, k)
        scatter!((i,obs[k]), color=:orange)
    end
end
plot(p)
#savefig("17taxa-pps.pdf")


plot(pls..., p)
