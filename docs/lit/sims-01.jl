using SmoothTree, Random, Distributions, Plots, StatsPlots

# The species tree
S = nw"((((((A,B),C),(D,E)),(F,(G,H))),I),O);"

# Now for reasonable intrenal branch lengths (coalescent units) recall
# that on this timescale, a branch of length one corresponds to the
# expected time for two genes to coalesce.
default(gridstyle=:dot, framestyle=:box, legend=false)
plot(Exponential(), xticks=1:8, color=:black, xlim=(0,8), ylim=(0,1))
vline!(quantile(Exponential(), [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]), color=:firebrick)

# Deep coalescence becomes important once branch lengths get below say
# 2.5. The 'zone of interest' for simulations would be somewhere
# between 0 and 3 I'd say.
quantile(Exponential(), [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

# Get some coalescent branch lengths in the interesting regime
n = SmoothTree.n_internal(S)
#θ = shuffle([2.5, 3.0, 1.5, 0.5, 3.0, 2.0, 0.25, 1.0])
θ = [2.5, 1.0, 1.5, 3.0, 0.5, 0.25, 3.0, 2.0]
setdistance_internal!(S, θ)
plot(S)

# simulate gene trees
N = 500
trees = randtree(MSC(S), N)
treedist = sort(collect(countmap(trees)), by=last, rev=true)

# hmm, but we actually want branch lengths in there, but we only have
# coal branch lengths in the MSC, and not a timetree with Ne
# parameters... Better turn to RevBayes.

trees = proportionmap(readnw.([x*";" for x in readlines("docs/rev/genetrees-01.nw")]))
trees = sort(collect(trees), by=last, rev=true)
last.(trees)


