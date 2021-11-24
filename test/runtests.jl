using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree, Serialization, StatsBase, Distributions
using LinearAlgebra

# Actual tests ======================================================
treesfile = joinpath(@__DIR__, "test/OG0006030.trees")
trees = readnw.(readlines(treesfile))
trees = SmoothTree.rootall!(trees)
ccd = CCD(trees)

@testset "Normalized" begin
    for (k,v) in ccd.cmap
        SmoothTree.isleafclade(k) && continue
        @test sum(values(ccd.smap[k])) ≈ v
    end
end


# Other stuff =======================================================
# EP tests
# First generate a simulated data set 
# -----------------------------------
# 1. the univariate model
S = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
SmoothTree.setdistance!(S, exp(1.5))
model = SmoothTree.MSC(S)

# 2. the data
# we do not include tree uncertainty here...
trees = SmoothTree.randtree(model, 100)
data = CCD.(trees)

# EP inference
# ------------
alg = SmoothTree._UvNormalEPABC(
    SmoothTree.MSC(S), data, 
    0., 2., NaN, 10000, 1.)

res = map(1:length(data)) do i
    d = SmoothTree.ep_iteration!(alg, i)
    alg.μ, alg.V
end

plot(plot(first.(res)), plot(last.(res)), legend=false, size=(400,200))

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
