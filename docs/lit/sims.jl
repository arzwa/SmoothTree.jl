# SmoothTree

# # Example: simulated data for a five taxon species tree
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, Distributions

# The species tree topology we will use:
species = string.('A':'T')
spmap = clademap(species, UInt64)
root = sum(keys(spmap))
ntaxa = length(spmap)
S = randtree(NatMBM(root, BetaSplitTree(-1., ntaxa)))
S = relabel(S, spmap)

# We set some random branch lengths
l = SmoothTree.n_internal(S)
θ = rand(Gamma(3., 1/2), l)
SmoothTree.setdistance_internal!(S, θ)

# We need a map associating with each clade its name. 

# simulate gene trees
M = SmoothTree.MSC(S, spmap)
N = 100
G = randtree(M, spmap, N)
ranking(G)

# Now put them in the relevant data structure for doing inference. From here
# on the code should be like for an analysis of actual empirical data.
data = Locus.(G, Ref(spmap), 1e-3, -1.)

# use a Beta-splitting prior
Sprior = NatMBM(CCD(G, spmap), BetaSplitTree(-1., ntaxa), 10.)
randtree(Sprior, 10000) |> ranking

# branch length prior
μ = 1.5; V = 2.
tips = collect(keys(spmap))
θprior = BranchModel(root, gaussian_mom2nat([log(μ), V]), inftips=tips)

# the species tree model
model = MSCModel(Sprior, θprior)

alg = SmoothTree.EPABCIS(data, model, 5000, target=100, miness=5., prunetol=1e-9)

trace = ep!(alg, 3)

# Take a sample from the posterior approximation
smple = relabel.(randtree(alg.model.S, 10000), Ref(spmap)) |> ranking

topologize(S)
smple[1][1]

# now look at the branch approximation for the relevant tree
using Distributions, Plots, StatsPlots
default(gridstyle=:dot, legend=false, title_loc=:left, titlefont=8, framestyle=:box)

true_bl = SmoothTree.getbranchdict(S, spmap)

bs = SmoothTree.getbranchapprox(alg.model, randsplits(alg.model.S))
bs = filter(x->!SmoothTree.isleafclade(x[2]), bs)
map(bs) do (γ, δ, d)
    plot(LogNormal(log(μ), √V), fill=true, color=:lightgray, xlim=(0,20))
    plot!(LogNormal(d.μ, d.σ), color=:black)
    #plot!(d, color=:black)
    #vline!([log(true_bl[(γ, δ)])], color=:black, ls=:dot)
    vline!([true_bl[(γ, δ)]], color=:black, ls=:dot)
end |> x->plot(x..., size=(1200,800))

xs = traceback(first.(trace))
plot(plot(xs.θ), plot(xs.μ), plot(xs.V), plot(getindex.(trace, 2)))
