
# 1. Simple ABC approach
# ======================
data = CCD(readnw.(readlines("docs/data/yeast.trees")), α=0.01, αroot=0.)

m = 5
treeprior = CCD(randtree(data, 1000), α=100., αroot=0.)
θprior = MvNormal(zeros(m), 1.)

nsims = 100000
simsize = 106  # size of simulated data set
allsims = map(1:nsims) do iteration
    T = randtree(treeprior)
    x = rand(θprior)
    SmoothTree.setdistance_internal_rooted!(T, exp.(x))
    simm = SmoothTree.MSC(T)
    sims = CCD(simm, randsplits(simm, simsize), α=0.01, αroot=0.)
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
    @info "KL < $q" 
    for (t, p) in sorted[1:5]
        ts = nwstr(t, dist=false)
        @printf "P = %.4f %s\n" p ts
    end
end


# 2. Synthetic likelihood
# =======================
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
    θ_ = θ + rand(proposal)
    θ_ < 0. && return kernel.θ, kernel.p
    SmoothTree.setdistance!(S, θ_)
    simm = SmoothTree.MSC(S)
    ccd = CCD(simm, randsplits(simm, nsims), α=α)
    l_ = logpdf(ccd, data)
    p_ = l_ + logpdf(prior, θ_)
    @info "θ=$θ,p=$p to θ=$θ_,p=$p_"
    if log(rand()) < p_ - p 
        kernel.θ = θ_
        kernel.p = p_
    end
    return kernel.θ, kernel.p
end

data = readnw.(readlines("docs/data/yeast.trees"))
S = randtree(CCD(data))

kernel = UvMH(data, Exponential(), Normal(0,0.1), exp(randn()), -Inf, S)
chain1 = map(x->mhabcstep!(kernel, 50000, 1.), 1:1000)
chain2 = map(x->mhabcstep!(kernel, 10000, 1.), 1:1000)

plot(first.(chain1)[100:end])
plot!(first.(chain2)[100:end])

# its hard to figure out for which `nsims` and `α` it works... It
# seems it can work though... 10000/10 seems to work
data = map(readdir("docs/data/yeast-rokas/ufboot", join=true)) do f
    @info f
    trees = readnw.(readlines(f))
    trees = SmoothTree.rootall!(trees, "Calb")
    countmap(trees)
end


using Serialization
serialize("docs/data/yeast-rokas/ccd-ufboot.jls", data)

