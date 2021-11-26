using Pkg; Pkg.activate(@__DIR__)
using NewickTree, SmoothTree, StatsBase, Plots, Serialization
using SmoothTree: CCD, MSC, randsplits

# This is a small experiment to see whether the CCD of *across* loci
# simulated under the MSC captures the tree topology distribution
# under the MSC reasonably well.

# We define a MSC model
S = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
SmoothTree.setdistance!(S, 1.)
model = MSC(S)

# We obtain a CCD based on 1000 random gene trees from the MSC
ccd = CCD(model, randsplits(model, 10000))

# We obtain an empirical distribution over gene trees under the MSC by
# simulating a very large number of trees
ref = randsplits(model, 10^7)
ref = sort(sort.(ref))
ref = sort(collect(proportionmap(ref)), by=last)

# We compute the CCD-based probability for each of the trees observed
# in the large sample and compare against the simple relative
# frequency estimate of the tree probability.
probs = map(ref) do (splits, p)
    (log(p), SmoothTree.logpdf(ccd, splits))
end

default(legend=false, framestyle=:box)
scatter(probs, color=:lightgray, size=(310,300)); 
plot!(x->x, color=:black, xlabel="SRF (n=10⁷)", ylabel="CCD (n=10⁴)") 

savefig("docs/img/2021-11-26-ccd-msc.pdf")

# It appears that the CCD captures the MSC induced distribution over
# tree topologies reasonably well.
