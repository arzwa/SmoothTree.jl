
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, StatsBase, Distributions, LinearAlgebra
using Plots, StatsPlots, Measures
default(gridstyle=:dot, legend=false, framestyle=:box,
        xlim=(-Inf,Inf), ylim=(-Inf,Inf))

n = 1000  # data set size (number of loci)
S = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
θ = 1.
SmoothTree.setdistance!(S, θ)
model = SmoothTree.MSC(S)
data = CCD(model, SmoothTree.randsplits(model, n), α=1/n)

function initsis(data, treeprior, θprior, N, ϵ, n)
    # draw particles from the prior and simulate a data set until we
    # accept N of them at tolerance ϵ   
    particles = map(1:N) do i
        accept = false
        particle = nothing
        while !accept
            S = randtree(treeprior)
            θ = rand(θprior)
            SmoothTree.setdistance!(S, θ)
            model = SmoothTree.MSC(S)
            y = CCD(model, randsplits(model, n), α=1/n)
            d = SmoothTree.kldiv(data, y)
            accept = d < ϵ
            particle = (S, θ, y, d, -log(N))
        end
        particle
    end
    return particles
end

function lognormalize(ls)
   ps = exp.(ls .- maximum(ls))
   return ps ./ sum(sort(ps))
end

function sis_it(particles, data, treeprior, θprior, N, ϵ, n)
    # 1. construct sampling distribution
    # one way I can think of to do this is construct a CCD of the
    # trees from previous particles?
    # Here we keep using the prior for θ
    trees = first.(particles)
    weights = lognormalize(last.(particles))
    input = sample(trees, Weights(weights), 10000)
    g = CCD(input, α=1e-6)
    new_particles = map(1:N) do i
        accept = false
        particle = nothing
        while !accept
            S = randtree(g)
            θ = rand(θprior)
            SmoothTree.setdistance!(S, θ)
            model = SmoothTree.MSC(S)
            y = CCD(model, randsplits(model, n), α=1/n)
            d = SmoothTree.kldiv(data, y)
            w = logpdf(treeprior, S) - logpdf(g, S)
            accept = d < ϵ
            particle = (S, θ, y, d, w)
        end
        particle
    end
end

treeprior = CCD(randtree(data, 1000), α=10.)
θprior = Exponential()

# there may be an issue in logpdf still...

function showtop(particles, n=10)
    ts = ranking(first.(particles))
    for (t, p) in ts[1:n]
        @printf "%s %.3f\n" nwstr(t, dist=false) p
    end
end

ϵ = 10.
particles = initsis(data, treeprior, θprior, 1000, ϵ, 100)
particles = sis_it(particles, data, treeprior, θprior, 1000, ϵ, 100)

for i=1:50
    ϵ *= 0.99
    @show ϵ
    particles = sis_it(particles, data, treeprior, θprior, 1000, ϵ, 100)
end



