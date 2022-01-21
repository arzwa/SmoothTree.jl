using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, StatsBase, ThreadTools, JSON

# load all the data (unsummarized)
# this takes a while
fnames = readdir("docs/data/mammals-song/424genes.ufboot", join=true)
function tfun(fpath) 
    ts = readnw.(readlines(fpath))
    ts = SmoothTree.rootall!(ts, "Gal")
    countmap(ts)
end
boots = tmap(tfun, fnames)

# load summarized data
# this also takes a while...
#boots = SmoothTree.readtrees("docs/data/mammals-song/424genes-ufboot.trees")

# load ML trees
trees = readnw.(readlines("docs/data/mammals-song/mltrees.nw"))

# get the taxon map
taxa = taxonmap(trees, UInt64)
ntaxa = length(taxa)

# get the CCDs
data = CCD.(boots, Ref(taxa))

# prior settings
Sprior = NatMBM(CCD(unique(trees), taxa), BetaSplitTree(-1., ntaxa), 100.)
θprior = BranchModel(UInt64, gaussian_mom2nat([log(1.), 2.]))
model  = MSCModel(Sprior, θprior, taxa)

# set up the algorithm
alg   = EPABC(data[1:10], model, λ=0.1, α=1/2^(ntaxa-1), minacc=20,
              target=100, prunetol=1e-6)

# run it
trace = pep!(alg, 1)
trace = ep!(alg, 3)
SmoothTree.tuneoff!(alg)
trace = [trace ; ep!(alg, 3)]

# save trace
open("epabc-trace.json", "w") do io
    JSON.print(traceback(trace))
end
