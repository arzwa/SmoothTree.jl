
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, Distributions, LinearAlgebra
using Plots, StatsPlots, Measures, LaTeXStrings
default(gridstyle=:dot, legend=false, title_loc=:left, titlefont=8, framestyle=:box)

# The species tree topology we will use:
species = string.('A':'Z')[1:6]
spmap = clademap(species)
root = rootclade(spmap)
ntaxa = length(spmap)
S = randtree(CCD(SplitCounts(root), BetaSplitTree(-1., ntaxa)), spmap)

# We set some random branch lengths
l = SmoothTree.n_internal(S)
d = MixtureModel([LogNormal(log(0.5), 0.5), LogNormal(log(3.), .5)], [0.2,0.8])
#d = MixtureModel([Normal(log(0.5), 0.5), Normal(log(3.), .5)], [0.2,0.8])
#d = Gamma(3., 1/2)
θ = rand(d, l)
SmoothTree.setdistance_internal!(S, θ)
plot(S, transform=true, scalebar=1)

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

# branch length prior
μ = 1.5; V = 5.

# fixed topology analysis
bs = SmoothTree.getbranches(S, spmap)
prior = SmoothTree.tonatural(MvNormal(fill(log(μ), length(bs)), √V))

alg = SmoothTree.EPABCSIS(data, bs, prior, 10000, 3, target=1000, miness=10., c=0.95)
trace = ep!(alg, 3)
post = SmoothTree.tomoment(alg.model)

scatter(bs.xs, post.μ, yerror=2sqrt.(diag(post.Σ)))
plot!(x->x)

diag(post.Σ)

# use a Beta-splitting prior
#Sprior = CCD(SplitCounts(unique(G), spmap), BetaSplitTree(-1., ntaxa), 50.)
Sprior = CCD(SplitCounts(root), BetaSplitTree(-1., ntaxa), 50.)
prsample = randtree(Sprior, spmap, 10000) |> ranking

tips = collect(keys(spmap))
#θprior = BranchModel(root, gaussian_mom2nat([log(μ), V]), inftips=tips)
θprior = BranchModel(root, gaussian_mom2nat([log(μ), V]))

# the species tree model
model = MSCModel(Sprior, θprior)

#alg = SmoothTree.EPABCIS(data, model, 50000, target=100, miness=10., prunetol=1e-6, α=1e-3, c=0.95)
#trace = ep!(alg, 4)

# SIS
alg = SmoothTree.EPABCSIS(data, model, 100000, 5, target=1000, miness=10.,
                          α=1e-3, c=0.95, prunetol=1e-9)
trace = ep!(alg, 1)
alg.siteupdate.N = 10000
alg.siteupdate.sims = alg.siteupdate.sims[1:10000]
trace = [trace ; ep!(alg, 2)]

randtree(alg.model.S, spmap, 10000) |> ranking |> first
SmoothTree.topologize(S)

true_bl = SmoothTree.getbranchdict(S, spmap)
M  = alg.model
bs = SmoothTree.getbranchapprox(M.ϕ, randbranches(M))

bs = filter(x->!SmoothTree.isleafclade(x[2]), bs)
pS = plot(S, transform=true, scalebar=2, pad=0, xlabel=L"2N_e", bottom_margin=4mm)
pZ = plot(getfield.(trace, :ev), color=:black, title=L"Z", margin=1mm, guidefont=7, grid=false)
ps = []
for (i, (γ, δ, d)) in enumerate(bs)
    p = plot(Normal(log(μ), √V), grid=false,
             fill=true, color=:lightgray, fillalpha=0.5,
             title=bitstring(δ)[end-9:end], titlefont=6,
             xlim=(-3,4.5), ylim=(0,1.2))
             #xticks=i<8 ? false : :auto, 
             #yticks=(i-1)%7 == 0 ? :auto : false)
    plot!(d, color=:black)
    vline!([log(true_bl[(γ, δ)])], color=:black, ls=:dot)
    vline!([d.μ], color=:black)
    push!(ps, p)    
end 
#plot(pZ, ps..., size=(600,200), layout=(2,4))
#plot(pS, pZ, ps..., size=(700,200), layout=@layout([a{0.15w} grid(2,4)]))
plot(pS, ps..., size=(700,200), layout=@layout([a{0.15w} grid(2,4)]), 
     ytickfont=6, xtickfont=6)
#savefig("docs/img/sims-10taxa-sis.pdf")


true_bl = SmoothTree.getbranchdict(S, spmap)
M  = alg.model
bs = SmoothTree.getbranchapprox(M.ϕ, randbranches(M))
p2 = map(bs) do (γ, δ, d)
    plot(Normal(log(μ), √V), fill=true, color=:lightgray, fillalpha=0.5)
    plot!(Normal(d.μ, d.σ), color=:black)
    vline!([log(true_bl[(γ, δ)])], color=:black, ls=:dot)
    vline!([d.μ], color=:black)
end |> x->plot(x...)

