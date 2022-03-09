# SmoothTree

# # Example: simulated data for a five taxon species tree
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, Distributions
using Plots, StatsPlots, Measures
default(gridstyle=:dot, legend=false, title_loc=:left, titlefont=8, framestyle=:box)

# The species tree topology we will use:
species = string.('A':'Z')[1:24]
spmap = clademap(species, UInt64)
root = sum(keys(spmap))
ntaxa = length(spmap)
S = randtree(NatMBM(root, BetaSplitTree(-1., ntaxa)))
S = relabel(S, spmap)

# We set some random branch lengths
l = SmoothTree.n_internal(S)
d = MixtureModel([LogNormal(log(0.5), 0.5), LogNormal(log(3.), .5)], [0.2,0.8])
#d = Gamma(3., 1/2)
θ = rand(d, l)
SmoothTree.setdistance_internal!(S, θ)

# plot the species tree
#plot(S, transform=true, scalebar=1)

# We need a map associating with each clade its name. 

# simulate gene trees
M = SmoothTree.MSC(S, spmap)
N = 100
G = randtree(M, spmap, N)
ranking(G)

# Now put them in the relevant data structure for doing inference. From here
# on the code should be like for an analysis of actual empirical data.
data = Locus.(G, Ref(spmap), 1e-1, -1.)

# use a Beta-splitting prior
Sprior = NatMBM(CCD(unique(G), spmap), BetaSplitTree(-1., ntaxa), 20.)
prsample = randtree(Sprior, 10000) |> ranking

# branch length prior
μ = 1.5; V = 2.
tips = collect(keys(spmap))
θprior = BranchModel(root, gaussian_mom2nat([log(μ), V]), inftips=tips)

# the species tree model
model = MSCModel(Sprior, θprior)
alg = SmoothTree.EPABCIS(data, model, 50000, target=100, miness=10.,
                         prunetol=1e-6, α=1e-3, c=0.95)

trace = ep!(alg, 2)

# Take a sample from the posterior approximation
smple = relabel.(randtree(alg.model.S, 10000), Ref(spmap)) |> ranking

topologize(S)
smple[1][1]
relabel(prsample[1][1], spmap)

# now look at the branch approximation for the relevant tree

true_bl = SmoothTree.getbranchdict(S, spmap)

bs = SmoothTree.getbranchapprox(alg.model.q, SmoothTree.getsplits(S, spmap))
bs = filter(x->!SmoothTree.isleafclade(x[2]), bs)
map(bs) do (γ, δ, d)
    #plot(Normal(log(μ), √V), fill=true, color=:lightgray)
    #plot!(d, color=:black)
    #vline!([log(true_bl[(γ, δ)])], color=:black, ls=:dot)
    plot(LogNormal(log(μ), √V), fill=true, color=:lightgray, xlim=(0,10))
    plot!(LogNormal(d.μ, d.σ), color=:black)
    vline!([true_bl[(γ, δ)]], color=:black, ls=:dot)
end |> x->plot(x..., size=(1200,800))



# fixed tree
tree = SmoothTree.getbranches(S, spmap)
alg = SmoothTree.EPABCIS(data, tree, θprior, 100000, target=1000, miness=10., prunetol=1e-6, α=1e-7)
trace = ep!(alg, 10)
xs = trace[2]

trace = [trace; ep!(alg, 10)]
xs = vcat(last.(trace)...)

plot(first.(xs),color=:black, margin=8mm)
p = twinx()
plot!(p, last.(xs), color=:lightgray)

