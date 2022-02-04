using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree
using StatsBase, Distributions, Plots
using Serialization
using LaTeXStrings, StatsPlots, Measures
default(gridstyle=:dot, legend=false, framestyle=:box, title_loc=:left, titlefont=7)

# simulate a species tree
T = UInt16
ntaxa = 6
root = rootclade(ntaxa, T) 
S = randtree(NatMBM(root, BetaSplitTree(-2., ntaxa)))  # simulate a pectinate tree
l = SmoothTree.n_internal(S)
SmoothTree.setdistance!(S, 1/(ntaxa-1))
m = taxonmap(S, T)

# simulate gene trees
M = SmoothTree.MSC(S, m)
N = 100
G = randtree(M, m, N)
ranking(G) 

#pth = mkpath("docs/data/anomalyzone")
#open(joinpath(pth, "2022-02-01-trees-.nw"), "w") do io
#    write(io, join(nwstr.(G), "\n"))
#end

# inference
μ, V = .5, 2.
a = 1/2^(ntaxa-1)
bsd = BetaSplitTree(-1.5, ntaxa)
data = CCD.(G, Ref(m))
data = MomMBM.(data, Ref(bsd), a)
Sprior = NatMBM(CCD(unique(G), m), bsd, 100.)
#Sprior = NatMBM(root, bsd)
θprior = BranchModel(root, gaussian_mom2nat([log(μ), V]))
model = MSCModel(Sprior, θprior, m)
alg = EPABC(data, model, prunetol=0., λ=0.1, α=1e-3, target=100, minacc=10, batch=100)

trace = ep!(alg, 2)
SmoothTree.tuneoff!(alg)
trace = [trace; ep!(alg, 1)]

#trace = [trace; ep!(alg, 3)]

smple = ranking(randtree(alg.model.S, 10000))
SmoothTree.topologize(S)
SmoothTree.relabel(first(smple)[1], m)

X1, X2 = traceback(trace)
mapS = first(smple)[1]
nodes = filter(n->!isleaf(n) && !isroot(n), postwalk(mapS))
clades = map(n->(id(parent(n)), id(n)), nodes) 
pls = map(clades) do g
    p = plot(Normal(log(μ), √V),color=:black, xlim=(-4.5,4.5), 
             grid=false, ylabel=L"p(\phi)")
    xs = trace[1:200:end]
    for (i,model) in enumerate(xs)
        lm, VV = SmoothTree.gaussian_nat2mom(model.q[g])
        i == length(xs) && plot!(Normal(lm, √VV), color=:lightgray, fill=true,
                                 fillalpha=0.5)
        plot!(Normal(lm, √VV), color=:black)
    end
    vline!([log(1/(ntaxa-1))], color=:black, ls=:dot, xlabel=L"\phi")
    p
end
title!(pls[1], "(A)")
p1 = plot(pls..., layout=(1,3), size=(800,150))

final_splits = SmoothTree.allsplits(alg.model.S)
all_splits = [final_splits; [(γ,γ-δ) for (γ,δ) in final_splits]]
qs = map(x->SmoothTree.MomBranchModel(x.q), trace)
splittrace = mapreduce(x->[MomMBM(x.S)[γ,δ] for (γ, δ) in final_splits], hcat, trace) |> permutedims
mutrace = mapreduce(x->[x[(γ,δ)][1] for (γ, δ) in all_splits], hcat, qs) |> permutedims
vtrace  = mapreduce(x->[x[(γ,δ)][2] for (γ, δ) in all_splits], hcat, qs) |> permutedims

kwargs = (xticks=0:1000:5000, xlabel="iteration", xguidefont=8)
ps4 = [
       plot(splittrace, title="(B)", ylabel=L"\theta"; kwargs...), 
       plot(mutrace, ylabel=L"\mu", xguidefont=8; kwargs...), 
       plot(vtrace, ylabel=L"\sigma^2"; kwargs...)
      ]
pl4 = plot(ps4..., layout=(1,3), size=(900,200))
plot(p1, pl4, layout=(2,1), size=(700,300), bottom_margin=3mm, left_margin=4mm)

savefig("docs/img/anomaly-5taxa-500trees.pdf")
