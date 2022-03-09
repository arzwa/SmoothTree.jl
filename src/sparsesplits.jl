# sparsesplits.jl
# ===============
# Represent an arbitrary split distribution sparsely with respect to a
# Beta-splitting distribution.

# For the sparse representation under Beta-splitting models we have the
# principle that unrepresented splits of the same size have the same
# probability. In other words, in order not to have to store all `2^(n-1)-1`
# splits explicitly, we need to store those splits which are represented and in
# addition one probability value for each possible size of clades (`n÷2`
# values).  Note that we 'fold' the split distribution because it is symmetric.
# This requires us to be careful when dealing with splits of size `n` of clades
# of size `2n`.

# The exponential family representation of a k-categorical distribution has k-1
# natural parameters. We need to be able to localize the `k`th split, which we
# define to be the following. Note that we do not need the rest of the order on
# splits, only a last element, since we use dictionaries to represent the split
# distribution. In principle we could use an actual `SparseVector`, if we
# specify the complete order on splits. 
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
# XXX this takes up a lot of time!
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

# randsplit for natural parameter, without using nat2mom
function randsplitnat(x::SparseSplits)
    η0 = exp.(x.η0)
    vs = exp.(values(x.splits))
    δs = collect(keys(x.splits))
    pr = x.k .* η0
    Z  = sum(vs) + sum(pr)
    vs ./= Z
    η0 ./= Z
    r = rand() 
    i = 1
    # sort δs, vs by vs?
    while i <= length(δs)
        pδ = vs[i]
        r -= (pδ - η0[splitsize(x.γ, δs[i])])
        r < 0. && return δs[i]
        i += 1
    end
    k = sample(Weights(pr))
    return randsplitofsize(x.γ, k)
end

function _priorsplit(x::SparseSplits)
    k = sample(Weights(x.η0 .* x.n))
    return randsplitofsize(x.γ, k)
end


# Non-mutating linear operations
# ------------------------------ 
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
    SparseSplits(d, x.γ, x.n, copy(x.k), a .* x.η0, x.ref)
end
# XXX there was a bug here, we forgot to copy!


# Mutating linear operations
# --------------------------
# These are mutating in the first argument
function mul!(x::SparseSplits{T,V}, a::V) where {T,V}
    for (γ, η) in x.splits
        x.splits[γ] *= a
    end
    x.η0 .*= a
    return x
end

function add!(x::SparseSplits, y::SparseSplits)
    splits = union(keys(x.splits), keys(y.splits))
    for δ in splits
        x.splits[δ] = x[δ] + y[δ] 
        !haskey(x, δ) && (x.k[splitsize(x.γ, δ)] -= 1)
    end
    x.η0 .+= y.η0
    return x
end

function sub!(x::SparseSplits, y::SparseSplits)
    splits = union(keys(x.splits), keys(y.splits))
    for δ in splits
        x.splits[δ] = x[δ] - y[δ] 
        !haskey(x, δ) && (x.k[splitsize(x.γ, δ)] -= 1)
    end
    x.η0 .-= y.η0
    return x
end


# Pruning
# -------
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
function prune!(x::SparseSplits{T,V}, atol) where {T,V}
    for (δ, v) in x.splits
        s = splitsize(x.γ, δ)
        if isapprox(v, x.η0[s], atol=atol)
            delete!(x.splits, δ)
            x.k[s] += 1
        end
    end 
    return x
end

