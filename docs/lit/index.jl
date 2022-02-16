# SmoothTree
#
# # Example: simulated data for a five taxon species tree
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, Distributions

# The species tree topology we will use:
S = nw"(A,(B,((C,D),(E,F))));"

# We set some random branch lengths
l = SmoothTree.n_internal(S)
θ = rand(Gamma(2., 1/2), l)
SmoothTree.setdistance_internal!(S, θ)

# We need a map associating with each clade its name. 
spmap = SmoothTree.clademap(S)
ntaxa = length(spmap)

# simulate gene trees
M = SmoothTree.MSC(S, spmap)
N = 100
G = randtree(M, spmap, N)
ranking(G)

# Now put them in the relevant data structure for doing inference. From here
# on the code should be like for an analysis of actual empirical data.
data = SmoothTree.Locus.(G, Ref(spmap), 1/(2^ntaxa -1), -1.)

# use a Beta-splitting prior
root = UInt16(2^ntaxa-1)
Sprior = NatMBM(root, BetaSplitTree(-1., ntaxa))

# branch length prior
μ = 2.; V = 2.
θprior = BranchModel(root, gaussian_mom2nat([log(μ), V]))

# the species tree model
model = MSCModel(Sprior, θprior)

alg = SmoothTree.EPABCIS(data, model, 20000, target=1000, miness=5., prunetol=1e-5)

trace = ep!(alg, 3)

# Take a sample from the posterior approximation
relabel.(randtree(alg.model.S, 10000), Ref(spmap)) |> ranking

# now look at the branch approximation for the relevant tree
using Distributions, Plots, StatsPlots, LaTeXStrings
default(gridstyle=:dot, legend=false, title_loc=:left, titlefont=9, framestyle=:box, guidefont=8)

true_bl = SmoothTree.getbranchdict(S, spmap)

bs = SmoothTree.getbranchapprox(alg.model, randsplits(alg.model.S))
map(bs) do (γ, δ, d)
    plot(Normal(log(μ), √V), fill=true, color=:lightgray)
    plot!(d, color=:black)
    vline!([log(true_bl[(γ, δ)])], color=:black, ls=:dot)
end |> x->plot(x..., size=(650,350))

# plot the traceback of the algorithm
xs = SmoothTree.traceback(first.(trace))
plot(plot(xs.θ, title=L"\theta"), 
     plot(xs.μ, title=L"\mu"), 
     plot(xs.V, title=L"\sigma^2"), 
     plot(getindex.(trace,2), title=L"Z"), xlabel="iteration")

