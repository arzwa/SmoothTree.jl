using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, Distributions, Plots, StatsPlots, StatsBase

# overload speciesname
SmoothTree.speciesname(x) = String(split(x, "_")[1])

# get data
data = map(readdir("docs/data/drosera/min5-mb-trprobs", join=true)) do f
    trees = SmoothTree.read_trprobs(f, outgroup="bvu")
end

species = ["dca_1", "dca_2", "dca_3", "dca_4", "dca_5", "dca_6", 
           "dre_1", "dre_2", "dre_3", "bvu"]

spmap = clademap(species)
ntaxa = length(spmap)

bsd = BetaSplitTree(-1.5, length(species))
loci = map(x->SmoothTree.MulLocus(x, spmap, prior=bsd, α=1e-4), data)

# Species tree prior
root = rootclade(spmap)
c1 = getclade(spmap, ["bvu"])
splitdict = Dict(root=>Dict(min(c1, root-c1)=>1))
X = SplitCounts(splitdict, root)
Sprior = CCD(X, BetaSplitTree(-1., ntaxa), 0.1) 


