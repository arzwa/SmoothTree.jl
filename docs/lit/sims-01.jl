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
        c = a+b
        d[(c, a)] = distance(n[1])
        d[(c, b)] = distance(n[2])
        return a+b
    end
    walk(S)
    return d
end

# Note, if one assumes θ ~ Gamma(α, 1/β), then E[exp(-θ)] = (1+1/β)^-k
# (from the mgf), which is approximately exp(-E[θ])

# simulate a species tree
T = UInt32
ntaxa = 20
root = rootclade(ntaxa, T) 
S = randtree(MomMBM(root, BetaSplitTree(-1., ntaxa)))
l = SmoothTree.n_internal(S)
θ = rand(Gamma(4., 1/2), l)
SmoothTree.setdistance_internal!(S, θ)
m = taxonmap(S, T)
d = getcladeθ(S, m)

# simulate gene trees
M = SmoothTree.MSC(S, m)
N = 100
G = randtree(M, m, N)
ranking(G) .|> last

μ, V = 1., 1.
a = 1/2^(ntaxa-1)
bsd = BetaSplitTree(-1., ntaxa)
data = CCD.(G, Ref(m))
data = MomMBM.(data, Ref(bsd), a)
Sprior = NatMBM(CCD(unique(G), m), bsd, 10.)
#Sprior = NatMBM(T(sum(keys(m))), bsd)
θprior = BranchModel(root, gaussian_mom2nat([log(μ), V]))
model = MSCModel(Sprior, θprior, m)
alg = EPABC(data, model, prunetol=1e-5, λ=0.1, α=a, target=500, minacc=10, batch=500)

# MAP tree under the prior
maprior = ranking(randtree(Sprior, 10000)) .|> last

# EP
trace = pep!(alg, 1)
trace = ep!(alg, 2)

SmoothTree.tuneoff!(alg)
trace = [trace; ep!(alg, 2)]

# XXX somehow the length for the branch leading to ABC is not recorded

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
nodes = filter(n->!isleaf(n) && !isroot(n), postwalk(mapS))
clades = map(n->(id(parent(n)), id(n)), nodes) 
pls = map(clades) do g
    p = plot(Normal(log(μ), √V), color=:lightgray,
         fill=true, fillalpha=0.8, xlim=(-4.5,4.5), yticks=false, grid=false)
    for model in first.(trace[1:100:end])
        lm, VV = SmoothTree.gaussian_nat2mom(model.q[g])
        plot!(Normal(lm, √VV), color=:black)
    end
    try
        vline!([log(d[g])], color=:black, ls=:dot)
    catch
        @warn "incorrect phylogeny!"
    end
    p
end 
#xlabel!(pls[13], L"\theta")
plot(pls..., layout=(3,6)) #,layout=(3,5), size=(600,300))
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
