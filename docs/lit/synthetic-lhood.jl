using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree, Serialization, StatsBase, Distributions
using LinearAlgebra, Parameters, Printf, AdaptiveMCMC
using Plots, StatsPlots
default(gridstyle=:dot, legend=false, framestyle=:box,
        xlim=(-Inf,Inf), ylim=(-Inf,Inf))

# Synthetic likelihood approach
S = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
θ = 1.5
SmoothTree.setdistance!(S, θ)
model = SmoothTree.MSC(S)
trees = SmoothTree.randtree(model, 200)
data = countmap(trees)

mutable struct UvMH{T,D,A,B,V}
    data::D
    prior::A
    proposal::B
    θ::T
    p::T
    S::V
end

function mhabcstep!(kernel::UvMH, nsims=10000, α=1e-4)
    @unpack data, θ, p, proposal, prior, S = kernel
    θ_, _ = proposal(θ)
    θ_ < 0. && return kernel.θ, kernel.p
    SmoothTree.setdistance!(S, θ_)
    simm = SmoothTree.MSC(S)
    ccd = CCD(simm, randsplits(simm, nsims), α=α)
    l_ = logpdf(ccd, data)
    p_ = l_ + logpdf(prior, θ_)
    @printf "θ=%5.3f θ'=%5.3f π=%10.3f π'=%10.3f\n" θ θ_ p p_
    if log(rand()) < p_ - p 
        kernel.θ = θ_
        kernel.p = p_
        proposal.accepted += 1
    end
    return kernel.θ, kernel.p
end

# need *a lot* of simulations to get some mixing...
# it gets stuck easily... But it gets to reasonable values...
kernel = UvMH(data, Exponential(), AdaptiveRwProposal(), exp(randn()), -Inf, S)
chain1 = map(x->mhabcstep!(kernel, 5e4, 10.), 1:1000)
kernel = UvMH(data, Exponential(), AdaptiveRwProposal(), exp(randn()), -Inf, S)
chain2 = map(x->mhabcstep!(kernel, 5e4, 10.), 1:1000)

plot(first.(chain1)[100:end])
plot!(first.(chain2)[100:end])
