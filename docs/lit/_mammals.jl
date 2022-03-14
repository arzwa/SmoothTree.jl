using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, StatsBase, ThreadTools, JSON

# load all the data (unsummarized)
# this takes a while
fnames = readdir("docs/data/mammals-song/424genes.ufboot", join=true)
function tfun(fpath) 
    ts = readnw.(readlines(fpath))
    ts = SmoothTree.rootall(ts, "Gal")
    @info fpath
    countmap(ts)
end
boots = tmap(tfun, rand(fnames, 50))

# load summarized data
# this also takes a while...
#boots = SmoothTree.readtrees("docs/data/mammals-song/424genes-ufboot.trees")

# load ML trees
trees = readnw.(readlines("docs/data/mammals-song/mltrees.nw"))

# get the taxon map
spmap = clademap(trees[1], UInt64)
ntaxa = length(spmap)

# get the CCDs
data = Locus.(boots, Ref(spmap), prior=BetaSplitTree(-1., ntaxa), α=1e-3)

# prior settings
Sprior = CCD(SplitCounts(unique(trees), spmap), BetaSplitTree(-1., ntaxa), 1.)
θprior = BranchModel(rootclade(spmap), gaussian_mom2nat([log(1.), 2.]))
model  = MSCModel(Sprior, θprior)

# set up the algorithm
alg   = EPABCIS(data, model, 100000, target=200, miness=5., λ=0.1, α=1e-2, c=0.95)
trace = ep!(alg, 5, traceit=1)

