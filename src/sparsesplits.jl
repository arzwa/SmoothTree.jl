# we pick a fixed *reference split* which serves as the supremum in
# the order of splits. 
# We first defined this to be the leaf clade with largest clade index
# but that was a big mistake, since we elsewhere have the convention
# that a split is represented by its smallest clade, so that this is
# not even an actual split! It's complement always is though
refsplit(γ::T) where T = T(γ - 2^(ndigits(γ, base=2) - 1))

# used for both natural and moment parameterizations
"""
    SparseSplits

Sparse representation of a split distribution, where all
non-explicitly stored splits are assumed to have equal probability.
"""
struct SparseSplits{T,V}
    splits::Dict{T,V}  # explicitly represented splits
    n  ::Int           # total number of splits
    k  ::Int           # number of explicitly represented splits
    η0 ::V             # parameter for unrepresented splits
    ref::T             # reference split
end

# accessors
Base.haskey(m::SparseSplits, δ) = haskey(m.splits, δ)
Base.getindex(m::SparseSplits, δ) = haskey(m.splits, δ) ? m.splits[δ] : m.η0

# Define linear operations
function Base.:+(x::SparseSplits, y::SparseSplits)
    splits = union(keys(x.splits), keys(y.splits))
    d = Dict(δ=>x[δ] .+ y[δ] for δ in splits)
    SparseSplits(d, x.n, length(d), x.η0 + y.η0, x.ref)
end

function Base.:-(x::SparseSplits, y::SparseSplits)
    splits = union(keys(x.splits), keys(y.splits))
    d = Dict(δ=>x[δ] .- y[δ] for δ in splits)
    SparseSplits(d, x.n, length(d), x.η0 - y.η0, x.ref)
end

Base.:*(a, x::SparseSplits) = x * a
function Base.:*(x::SparseSplits{T,V}, a::V) where {T,V}
    d = Dict(γ => η * a for (γ, η) in x.splits)    
    SparseSplits(d, x.n, x.k, a*x.η0, x.ref)
end

"""
Get a natural parameter SparseSplits object from a split distribution
`d`, assuming the full support is represented in `d`.
"""
function SparseSplits(γ, d::Dict{T,V}) where {T,V}
    ρ = refsplit(γ)
    n = _ns(cladesize(γ))
    k = length(d)
    p = sum(values(d))
    p0 = log((1 - p)/(n - k))  # probability of unrepresented split
    pρ = haskey(d, ρ) ? log(d[ρ]) : p0
    η0 = p0 - pρ
    dd = Dict(δ => log(p) - pρ for (δ, p) in d)
    SparseSplits(dd, n, k, η0, ρ)
end

"""
Get a natural parameter SparseSplits object from split counts `d`,
assuming the Dirichlet-BMP model with Dirichlet parameter `α`
"""
function SparseSplits(γ, d::Dict{T,V}, α) where {T,V}
    ρ = refsplit(γ)
    n = _ns(cladesize(γ))  # total number of splits
    k = length(d)          # number of represented splits
    pρ = haskey(d, ρ) ? log(α + d[ρ]) : log(α)  # unnormalized pr of split ρ
    η0 = log(α) - pρ 
    dd = Dict(δ => log(α + c) - pρ for (δ, c) in d)
    SparseSplits(dd, n, k, η0, ρ)
end

# Natural to moment parameterization conversion
function nat2mom(x::SparseSplits)
    ρ = x.ref
    d = Dict(k=>exp(v) for (k,v) in x.splits)
    S = sum(values(d))
    Z = S + (x.n - x.k) * exp(x.η0) 
    for (k, v) in d
        d[k] /= Z
    end
    return SparseSplits(d, x.n, x.k, exp(x.η0)/Z, ρ)
end

# Moment to natural parameterization conversion
function mom2nat(x::SparseSplits)
    ρ = x.ref
    pρ = log(x[ρ])
    η0 = log(x.η0) - pρ
    d = Dict(k=>log(p) - pρ for (k,p) in x.splits)
    return SparseSplits(d, x.n, x.k, η0, ρ)
end

"""
    prune(x::SparseSplits, atol=1e-9)

Prune out splits which are barely supported (have η ≈ η0, with
absolute tolerance `atol`)
"""
function prune(x::SparseSplits{T,V}, atol) where {T,V}
    d = Dict{T,V}()
    for (k,v) in x.splits
        !(isapprox(v, x.η0, atol=atol)) && (d[k] = v)
    end 
    # should we adjust η0?
    SparseSplits(d, x.n, length(d), x.η0, x.ref) 
end

function prune!(x::SparseSplits, atol)
    for (k,v) in x.splits
        isapprox(v, x.η0, atol=atol) && delete!(x.splits, k)
    end 
end
