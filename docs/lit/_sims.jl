
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, Distributions
using Plots, StatsPlots, Measures
default(gridstyle=:dot, legend=false, title_loc=:left, titlefont=8, framestyle=:box)

# The species tree topology we will use:
species = string.('A':'Z')[1:9]
spmap = clademap(species, UInt64)
root = sum(keys(spmap))
ntaxa = length(spmap)
S = randtree(CCD(SplitCounts(root), BetaSplitTree(-1., ntaxa)), spmap)

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
data = Locus.(G, Ref(spmap), prior=BetaSplitTree(-1., ntaxa), α=1e-4)

# use a Beta-splitting prior
#Sprior = CCD(SplitCounts(unique(G), spmap), BetaSplitTree(-1., ntaxa), 50.)
Sprior = CCD(SplitCounts(root), BetaSplitTree(-1., ntaxa), 50.)
prsample = randtree(Sprior, spmap, 10000) |> ranking

# branch length prior
μ = 1.5; V = 2.
tips = collect(keys(spmap))
#θprior = BranchModel(root, gaussian_mom2nat([log(μ), V]), inftips=tips)
θprior = BranchModel(root, gaussian_mom2nat([log(μ), V]))

# the species tree model
model = MSCModel(Sprior, θprior)

#alg = SmoothTree.EPABCIS(data, model, 50000, target=100, miness=10., prunetol=1e-6, α=1e-3, c=0.95)
#trace = ep!(alg, 4)

# SIS
alg = SmoothTree.EPABCSIS(data, model, 50000, 5, target=1000, miness=5., α=1e-4, c=0.95, prunetol=1e-9)
trace = ep!(alg, 3)

trace = [trace ; ep!(alg, 3)]

randtree(alg.model.S, spmap, 10000) |> ranking |> first
SmoothTree.topologize(S)

true_bl = SmoothTree.getbranchdict(S, spmap)
M  = alg.model
bs = SmoothTree.getbranchapprox(M.ϕ, randbranches(M))
bs = filter(x->!SmoothTree.isleafclade(x[2]), bs)

pS = plot(S, transform=true, scalebar=5, pad=0, xlabel=L"5N_e", bottom_margin=4mm)
pZ = plot(getfield.(trace, :ev), color=:black, title=L"Z", margin=1mm, guidefont=7, grid=false)
ps = []
for (i, (γ, δ, d)) in enumerate(bs)
    p = plot(Normal(log(μ), √V), grid=false,
             fill=true, color=:lightgray, 
             title=bitstring(δ)[end-8:end], titlefont=6,
             xlim=(-3,4.5), ylim=(0,1.2))
             #xticks=i<8 ? false : :auto, 
             #yticks=(i-1)%7 == 0 ? :auto : false)
    plot!(d, color=:black)
    vline!([log(true_bl[(γ, δ)])], color=:black, ls=:dot)
    vline!([d.μ], color=:black)
    push!(ps, p)    
end 
#plot(pZ, ps..., size=(600,200), layout=(2,4))
plot(pS, pZ, ps..., size=(700,200), layout=@layout([a{0.15w} grid(2,4)]))
#savefig("docs/img/sims-9taxa-sis.pdf")


p2 = map(bs) do (γ, δ, d)
    plot(LogNormal(log(μ), √V), fill=true, color=:lightgray, fillalpha=0.5, xlim=(0,15), ylim=(0,Inf))
    plot!(LogNormal(d.μ, d.σ), color=:black)
    vline!([true_bl[(γ, δ)]], color=:black, ls=:dot)
    vline!([exp(d.μ)], color=:black)
end |> x->plot(x...)

