# if properly implemented, SparseSplits becomes a special case, and we
# can get rid of it...

# For the sparse representation under Beta-splitting models we have
# the principle that unrepresented splits of the same size have the
# same probability. In other words, in order not to have to store all
# 2^(n-1)-1 splits explicitly, we need to store those splits which are
# represented and in addition one probability value for each possible
# size of clades (n÷2 values). Note that we 'fold' the split
# distribution because it is symmetric. This requires us to be careful
# when dealing with splits of size n of clades of size 2n.

refsplit(γ::T) where T = T(γ - 2^(ndigits(γ, base=2) - 1))

# used for both natural and moment parameterizations
"""
    SparseSplits
"""
struct SparseSplits{T,V}
    splits::Dict{T,V}  # explicitly represented splits
    γ  ::T             # parent clade
    n  ::Vector{Int}   # total number of splits of each size
    k  ::Vector{Int}   # number of unrepresented splits of each size
    η0 ::Vector{V}     # parameter for unrepresented splits of size [1..|γ|-1]
    ref::T             # reference split
end

# accessors
Base.haskey(m::SparseSplits, δ) = haskey(m.splits, δ)

function Base.getindex(m::SparseSplits, δ) 
    haskey(m.splits, δ) ? m.splits[δ] : m.η0[splitsize(m.γ, δ)]
end

SparseSplits(γ::T, β) where T = SparseSplits(γ, Dict{T,Int}(), β, 1.)

"""
Get a natural parameter SparseSplits object from split counts `d`,
assuming the Beta-splitting Dirichlet prior with pseudo-count α and
shape parameter `β`. 
"""
function SparseSplits(γ, d::Dict{T}, β, α) where T
    ρ = refsplit(γ)
    s = cladesize(γ) 
    ns = nsplits.(s, 1:s÷2) 
    as = α .* β.q[s-2]
    aρ = as[splitsize(γ, ρ)]
    pρ = haskey(d, ρ) ? log(aρ + d[ρ]) : log(aρ)  # unnormalized pr of split ρ
    η0 = log.(as) .- pρ 
    dd = Dict{T,Float64}()
    kc = zeros(Int, s÷2)  # counter
    for (δ, count) in d
        k = splitsize(γ, δ)
        dd[δ] = log(as[k] + count) - pρ
        kc[k] += 1
    end
    k = ns .- kc
    SparseSplits(dd, γ, ns, k, η0, ρ)
end
# TODO should handle α=0

# Natural to moment parameterization conversion
function nat2mom(x::SparseSplits)
    η0 = exp.(x.η0)
    d = Dict(k=>exp(v) for (k,v) in x.splits)
    S = sum(values(d))
    Z = S + sum(x.k .* η0)
    for (k, v) in d
        d[k] /= Z
    end
    return SparseSplits(d, x.γ, x.n, x.k, η0/Z, x.ref)
end

# Moment to natural parameterization conversion
function mom2nat(x::SparseSplits)
    ρ = x.ref
    pρ = log(x[ρ])
    η0 = log.(x.η0) .- pρ
    d = Dict(k=>log(p) - pρ for (k,p) in x.splits)
    return SparseSplits(d, x.γ, x.n, x.k, η0, ρ)
end

# Sampling
# a random split for a SparseSplits split distribution
# Assumes moment parameter!
function randsplit(x::SparseSplits)
    length(x.splits) == 0 && return _priorsplit(x)
    splitps = collect(x.splits)
    pr = [w - x.η0[splitsize(x.γ, δ)] for (δ, w) in splitps]
    if rand() < sum(pr)
        i = sample(Weights(pr))
        return first(splitps[i])
    else 
        return _priorsplit(x)
    end
end

function _priorsplit(x::SparseSplits)
    k = sample(Weights(x.η0 .* x.n))
    return randsplitofsize(x.γ, k)
end

# Define linear operations
# only for natural params...
function Base.:+(x::SparseSplits{T,V}, y::SparseSplits) where {T,V}
    @assert x.γ == y.γ
    splits = union(keys(x.splits), keys(y.splits))
    k = copy(x.n)
    d = Dict{T,V}()
    for δ in splits
        d[δ] = x[δ] + y[δ]
        k[splitsize(x.γ, δ)] -= 1
    end
    SparseSplits(d, x.γ, x.n, k, x.η0 + y.η0, x.ref)
end

function Base.:-(x::SparseSplits{T,V}, y::SparseSplits) where {T,V}
    @assert x.γ == y.γ
    splits = union(keys(x.splits), keys(y.splits))
    k = copy(x.n)
    d = Dict{T,V}()
    for δ in splits
        d[δ] = x[δ] - y[δ]
        k[splitsize(x.γ, δ)] -= 1
    end
    SparseSplits(d, x.γ, x.n, k, x.η0 - y.η0, x.ref)
end

Base.:*(a, x::SparseSplits) = x * a
function Base.:*(x::SparseSplits{T,V}, a::V) where {T,V}
    d = Dict(γ => η * a for (γ, η) in x.splits)    
    SparseSplits(d, x.γ, x.n, x.k, a .* x.η0, x.ref)
end

"""
    prune(x::SparseSplits, atol=1e-9)

Prune out splits which are barely supported (have η ≈ η0, with
absolute tolerance `atol`)
"""
function prune(x::SparseSplits{T,V}, atol) where {T,V}
    d = Dict{T,V}()
    ks = x.k
    for (k,v) in x.splits
        s = splitsize(x.γ, k)
        if !isapprox(v, x.η0[s], atol=atol)
            d[k] = v
        else
            ks[s] += 1
        end
    end 
    # should we adjust η0?
    SparseSplits(d, x.γ, x.n, ks, x.η0, x.ref) 
end

# we have no in-place method
