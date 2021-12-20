using Pkg; Pkg.activate(@__DIR__)
using NewickTree, SmoothTree, StatsBase, Plots, Serialization
using SmoothTree: CCD, MSC, randsplits
default(legend=false, framestyle=:box)

# This is a small experiment to see whether the CCD of *across* loci
# simulated under the MSC captures the tree topology distribution
# under the MSC reasonably well.

# We define a MSC model
S = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
SmoothTree.setdistance!(S, 1.)
model = MSC(S)

# We obtain a CCD based on 1000 random gene trees from the MSC
ccd1 = CCD(model, randsplits(model, 10000), α=1/63)
ccd2 = CCD(model, randsplits(model, 10000), α=0.)

# We obtain an empirical distribution over gene trees under the MSC by
# simulating a very large number of trees
ref = randsplits(model, 10^7)
ref = sort(sort.(ref))
ref = sort(collect(proportionmap(ref)), by=last)

# We compute the CCD-based probability for each of the trees observed
# in the large sample and compare against the simple relative
# frequency estimate of the tree probability.
map([ccd1, ccd2]) do ccd
    probs = map(ref) do (splits, p)
        (log10(p), SmoothTree.logpdf(ccd, splits)/log(10))
    end
    k = length(filter(x->isfinite(x[2]), probs))
    @info "$k/$(length(ref)) trees with non-zero support in CCD"
    scatter(probs, color=:lightgray, size=(310,300)); 
    plot!(x->x, color=:black, xlabel="SRF (n=10⁷)", ylabel="CCD (n=10⁴)", 
          ls=:dot, xlim=(-7.5,0)) 
end |> x->plot(x..., size=(620, 300))

savefig("docs/img/2021-12-04-ccd-msc.pdf")
savefig("/home/arzwa/vimwiki/build/img/abc-msc/2021-12-04-ccd-msc.png")

# It appears that the CCD captures the MSC induced distribution over
# tree topologies reasonably well.
#
# It also seems that adding regularization to the CCD does not lead to
# overestimation of tree probabilities for rare trees, and rather that
# overestimation is mostly due to the observed part of the CCD.
# Regularization should be a good thing, better to underestimate a
# probability than to call it 0.
