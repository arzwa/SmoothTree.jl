using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree, Serialization, StatsBase, Distributions
using LinearAlgebra

# ABC test
# ========
# What is this approach? pseudolikelihood? 
# it does not work
S = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
θ = 1.
SmoothTree.setdistance!(S, θ)
model = SmoothTree.MSC(S)
data = CCD(model, SmoothTree.randsplits(model, 1000))

prior = Exponential()
d = map(1:100000) do iteration
    x = rand(prior)
    SmoothTree.setdistance!(S, x)
    simm = SmoothTree.MSC(S)
    t = randsplits(simm)
    l = logpdf(data, t)
    (log(rand()) < l, l, x)
end

acc = first.(d) 
v = last.(d[acc])
stephist(v, color=:black); vline!([θ])

# ABC
prior = Exponential()
d = map(1:10000) do iteration
    x = rand(prior)
    SmoothTree.setdistance!(S, x)
    simm = SmoothTree.MSC(S)
    sims = CCD(simm, randsplits(simm, 100))
    d = SmoothTree.kldiv(data, sims, 1e-10)
    (d, x)
end

acc = abs.(first.(d)) .< 20
v = last.(d[acc])
stephist(v, color=:black); vline!([θ])

# multivariate
S = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
SmoothTree.setdistance_internal!(S, [1., 1.5, 0.5, 1., 2.])
model = SmoothTree.MSC(S)
data = CCD(model, SmoothTree.randsplits(model, 1000))

prior = MvNormal(zeros(5), 1)
d = map(1:100000) do iteration
    θ = exp.(rand(prior))
    SmoothTree.setdistance_internal!(S, θ)
    simm = SmoothTree.MSC(S)
    t = randsplits(simm)
    l = logpdf(data, t)
    (log(rand()) < l, l, θ)
end

acc = first.(d) 
v = permutedims(hcat(last.(d[acc])...))
mean(v, dims=1)


# EP tests
# First generate a simulated data set 
# -----------------------------------
# 1. the univariate model
S = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
SmoothTree.setdistance!(S, exp(.75))
model = SmoothTree.MSC(S)

# 2. simulate data
# we do not include tree uncertainty here...
data = [CCD(model, [SmoothTree.randsplits(model)]) for i=1:500]

#mm = MSC1(S)
#trees = randtree(mm, 1000)
# add noise, note though that this does not accurately reflect noisy
# trees I guess... would be better to sample from a mixture of NNI
# trees with random weights?
#trees = map(trees) do tree
#    td = SmoothTree.RootedNNI(tree, Geometric(0.9))
#    proportionmap(SmoothTree.randtree(td, 100))
#end
#data = CCD.(trees)

# EP inference
# ------------
alg = SmoothTree._UvNormalEPABC(
    SmoothTree.MSC(deepcopy(S)), data, 
    0., 2., NaN, 10000, 1.)

res = map(1:length(data)) do i
    d = SmoothTree.ep_iteration!(alg, i)
    alg.μ, alg.V
end

plot(plot(first.(res)), plot(last.(res)), legend=false, size=(400,200))

# now with another species tree
T = nw"((smo,(((gge,iov),sgt),(xtz,dzq))),jvs);"
alg2 = SmoothTree._UvNormalEPABC(
    SmoothTree.MSC(T), data, 
    0., 2., NaN, 20000, 1.)

res3 = map(1:length(data)) do i
    d = SmoothTree.ep_iteration!(alg3, i)
    alg2.μ, alg2.V
end

# Multivariate
μ = ones(12) ./ 2
Σ = diagm(μ)
alg = GaussianEPABC(data, MSC(S), μ, Σ, 0.1, 10000, 0, μ, Σ, NaN)



# MSC sims 
S = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
relabel!(tree, fun) = for n in getleaves(tree); fun(n); end
relabel!.(trees, x->x.data.name = split(name(x), "_")[1])
setdistance!(tree, d) = for n in postwalk(tree); n.data.distance = d; end
setdistance!(tree, d::Vector) = for n in postwalk(tree); n.data.distance = d[id(n)]; end

setdistance!(S, rand(Exponential(2.), 13))
sims = map(x->SmoothTree.mscsim(S), 1:1000);
ccd = CCD(sims)
countmap(sims)


# basic strategy
# Note here we are only looking at a single gene tree..., but we can
# compare all gene trees against the same simulated CCD.
setdistance!(S, 1.)
sims = map(x->SmoothTree.mscsim(S), 1:1000);
ccd = CCD(sims)
SmoothTree.logpdf(ccd, trees) 

function dosample(S, prior, trees, N=10000)
    θ = rand(prior)
    for n in postwalk(S)
        n.data.distance = θ
    end
    ccd = CCD(map(x->SmoothTree.mscsim(S), 1:N))
    SmoothTree.logpdf(ccd, trees) 
end

sample = map(i->dosample(S, Exponential(), trees, 2000), 1:1000)

# this is some kind of estimate of some kind of marginal likelihood...
estimate = logsumexp(sample) - log(length(sample))

S2 = nw"((((gge,iov),(xtz,dzq)),(smo,sgt)),jvs);"
sample3 = map(i->dosample(S2, Exponential(), trees, 2000), 1:1000)
estimate3 = logsumexp(sample3) - log(length(sample3))


# we could also do it differently
# simulate a single tree under the MSC, and compute its average
# probability across the CCDs for all families?



# MUL tree
trees = deserialize("test/treesummary.jls")

ccds = map(trees) do treeset
    xs = collect(treeset)
    counts = last.(xs)
    weights = counts / sum(counts)
    CCD(first.(xs), weights=weights)
end

X = SmoothTree.TreeData.(ccds)

S = nw"(((((dca,dca),dre),((dca,dca),dre)),((dca,dre),dca)),bvu);"
model = (nv=19, θ=exp.(randn(19)), S=S) 

SmoothTree.logpdf(model, X)

extree = collect(trees[3])[3][1]
X = SmoothTree.TreeData(CCD([extree]))
SmoothTree.logpdf(model, X)
