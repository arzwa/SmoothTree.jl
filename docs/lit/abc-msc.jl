# A basic ABC approach for inference under the MSC
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, StatsBase, Distributions, LinearAlgebra
using Plots, StatsPlots, Measures
default(gridstyle=:dot, legend=false, framestyle=:box,
        xlim=(-Inf,Inf), ylim=(-Inf,Inf))

# 1. Fixed species tree, single parameter
# 1.1 Generate simulated data
n = 1000  # data set size (number of loci)
S = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
θ = 0.5
SmoothTree.setdistance!(S, θ)
model = SmoothTree.MSC(S)
data = CCD(model, SmoothTree.randsplits(model, n), α=1/n)

# 1.2 ABC based inference
prior = Exponential()
nsims = 10000
simsize = 100  # size of simulated data set
allsims = map(1:nsims) do iteration
    x = rand(prior)
    SmoothTree.setdistance!(S, x)
    simm = SmoothTree.MSC(S)
    sims = CCD(simm, randsplits(simm, simsize), α=1/simsize)
    #d = SmoothTree.symmkldiv(data, sims)
    #d = SmoothTree.kldiv(sims, data)
    d = SmoothTree.kldiv(data, sims)
    (d, x)
end
# it seems to work best with kldiv(data, sims), which makes sense, as
# we wish to minimize this quantity, i.e. this is the 'amount of
# information lost when approximating data by sims'. It also works
# reasonably well with the symmetric divergence.

# Plot the densities for different ABC kernel widths
ds = first.(allsims)
qs = quantile(ds, [0.05, 0.10, 0.20, 0.5])
p = vline([θ], color=:black, lw=2.)
plot!(prior, color=:black)
vline!([mean(prior)], lw=2, ls=:dot, color=:black)
for (i,h) in enumerate(reverse(qs))
    acc = ds .< h
    v = last.(allsims[acc])
    density!(v, fill=true, fillalpha=0.1); 
end
plot(p, size=(300,200), ylim=(-Inf,Inf), xlim=(0.,5))

# We seem to get best performance when setting α very low for the
# simulated data, what is the logic there? 

# 2. Fixed species tree, multivariate
# 2.1 Simulate data
n = 1000  # data set size (number of loci)
S = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
m = SmoothTree.n_internal(S)
θ = rand([0.5, 1., 1.5, 2.5], m)
SmoothTree.setdistance_internal!(S, θ)
model = SmoothTree.MSC(S)
data = CCD(model, SmoothTree.randsplits(model, n), α=1/n)

# 2.2 ABC inference
prior = MvNormal(zeros(m), 1.)
nsims = 100000
simsize = 100  # size of simulated data set
allsims = map(1:nsims) do iteration
    x = rand(prior)
    SmoothTree.setdistance_internal!(S, exp.(x))
    simm = SmoothTree.MSC(S)
    sims = CCD(simm, randsplits(simm, simsize), α=1/simsize)
    #d = SmoothTree.symmkldiv(data, sims)
    d = SmoothTree.kldiv(data, sims)
    (d, x)
end

# Plot the densities
ds = first.(allsims)
xs = permutedims(hcat(last.(allsims)...))
qs = quantile(ds, [0.05, 0.10, 0.20, 1.])
map(1:m) do k
    p = vline([log(θ[k])], color=:black, lw=2.)
    for (i, h) in enumerate(reverse(qs))
        acc = ds .< h
        v = xs[acc,k]
        i == 1 ? density!(v, color=:black) : 
            density!(v, fill=true, fillalpha=0.3); 
    end
    plot(p, ylim=(0,1))#, xlim=(0,8))
end |> x->plot(x..., layout=(1,5), size=(950,150), bottom_margin=4mm)

# It seems to work, but is clearly more challenging. More simulations
# obviously help, more data (10⁴ instead of 10³) does not seem to
# matter that much... 


# 3. Species tree inference
# 3.1 Simulated data
n = 1000  # data set size (number of loci)
S = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
m = SmoothTree.n_internal(S)
θ = rand([0.5, 1., 1.5, 2.5], m)
SmoothTree.setdistance_internal!(S, θ)
model = SmoothTree.MSC(S)
data = CCD(model, SmoothTree.randsplits(model, n), α=1/n)

# 3.2 Generate a prior for the species tree
# We could sample a bunch of trees from the gene tree data (summarized
# as CCD), and use a fairly large α to get a suitable prior.
treeprior = CCD(randtree(data, 10000), α=5000.)
θprior = MvNormal(zeros(m), 1.)

# examine the prior
priorsample = SmoothTree.ranking(randtree(treeprior, 10000))
sticks(last.(priorsample), xscale=:log10, color=:black, lw=5)

# 3.3 ABC based inference
nsims = 100000
simsize = 100  # size of simulated data set
allsims = map(1:nsims) do iteration
    T = randtree(treeprior)
    x = rand(θprior)
    SmoothTree.setdistance_internal!(T, exp.(x))
    simm = SmoothTree.MSC(T)
    sims = CCD(simm, randsplits(simm, simsize), α=1/simsize)
    #d = SmoothTree.symmkldiv(data, sims)
    d = SmoothTree.kldiv(data, sims)
    (d, T, x)
end

ds = first.(allsims)
qs = quantile(ds, [0.01, 0.05, 0.10, 0.20, 1.])
density(ds, color=:black); vline!(qs, ls=:dot, lw=2, color=:black)

map(qs) do q
    sims = filter(x->x[1] < q, allsims)
    trees = proportionmap(getindex.(sims, 2))
    sorted = sort(collect(trees), by=last, rev=true)
    t, p = first(sorted)
    p = round(p, digits=3)
    vs = round.(last.(sorted), digits=4)
    maptree = nwstr(t, dist=false)
    @info "$(isisomorphic(t, S)), P = $p $(length(sims)) $maptree"
end

# The space of trees seems to be too large to work with a prior which
# is too diffuse. We see some strange things, sometimes a slight
# difference in the prior may lead to a very different posterior
# distribution? Also the KL divergence histogram looks strange.
