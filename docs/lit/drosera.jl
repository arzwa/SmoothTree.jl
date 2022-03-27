using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, Distributions, Plots, StatsPlots, StatsBase
theme(:wong2)
default(gridstyle=:dot, legend=false, title_loc=:left, titlefont=8, framestyle=:box)

# overload speciesname
SmoothTree.speciesname(x) = String(split(x, "_")[1])


species = ["dca_1", "dca_2", "dca_3", "dca_4", "dca_5", "dca_6", 
           "dre_1", "dre_2", "dre_3", "bvu"]

spmap = clademap(species)
ntaxa = length(spmap)

# get data
bsd = BetaSplitTree(-1.5, length(species))
data = map(readdir("docs/data/drosera/min5-mb-trprobs", join=true)) do f
    trees = SmoothTree.read_trprobs(f, outgroup="bvu")
    SmoothTree.MulLocus(trees, spmap, prior=bsd, α=1e-9, tag=f)
end
sort!(data, by=length, rev=true)

# Species tree prior
root = rootclade(spmap)
c1 = getclade(spmap, ["bvu"])
splitdict = Dict(root=>Dict(min(c1, root-c1)=>1))
X = SplitCounts(splitdict, root)
Sprior = CCD(X, BetaSplitTree(-1.5, ntaxa), 1e-9) 

# Branch prior and model 
μ, V = log(2.), 3.
ϕprior = BranchModel(root, gaussian_mom2nat([μ, V]))
model  = MSCModel(Sprior, ϕprior)

alg = EPABCSIS(loci[1:100], model, 50000, 5, 
               target=200, miness=5., 
               λ=0.1, α=1e-3, c=0.95, 
               prunetol=1e-7)

trace1 = ep!(alg, 1)

plot([plot(t, transform=true, title="$p") for (t, p) in
      first(ranking(randtree(alg.model.S, spmap, 10000)), 9)]...,
     size=(600,600))

