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

# unnormalized beta-splitting density (n,i)
function betasplitf(β, n, i) 
    p = (2i == n ? 0.5 : 1.0)
    β == Inf && return p * binomial(n,i)/_ns(n) 
    p * gamma(β+1+i)*gamma(β+1+n-i)/(gamma(i+1)*gamma(n-i+1))
end

function betasplitpmf(β, n)
    # use recursive formula from Jones?
    ps = [betasplitf(β, n, i) for i=1:n÷2]
    ps ./ sum(ps)
end

nsplits(n, i) = n == 2i ? binomial(n,i)÷2 : binomial(n,i)

# we define the split size to be the size of the smaller clade, note
# that this does not commute with the order on splits! i.e. a split
# is identified by the subclade with smallest id but this id need not
# be the id of the smallest subclade.
splitsize(γ, δ) = min(cladesize(δ), cladesize(γ-δ))

"""
Get a natural parameter SparseSplits object from split counts `d`,
assuming the Beta-splitting Dirichlet prior with pseudo-count α and
shape parameter `β`. 
"""
function SparseSplits(γ, d::Dict{T,V}, β, α) where {T,V}
    ρ = refsplit(γ)
    s = cladesize(γ)       # clade size
    ns = [nsplits(s, i) for i=1:s÷2] 
    # pseudocounts at split level 
    # XXX have to normalize the betasplit probability
    ps = betasplitpmf(β, s)
    as = [α * ps[i] / ns[i] for i=1:s÷2]
    aρ = as[splitsize(γ, ρ)]
    pρ = haskey(d, ρ) ? log(aρ + d[ρ]) : log(aρ)  # unnormalized pr of split ρ
    η0 = log.(as) .- pρ 
    xs = collect(d)
    ks = splitsize.(Ref(γ), first.(xs))
    dd = Dict(δ => log(as[k] + c) - pρ for (k, (δ, c)) in zip(ks, xs))
    kc = counts(ks, 1:s÷2)
    k  = ns .- kc
    SparseSplits(dd, γ, ns, k, η0, ρ)
end
# TODO should handle α=0

# Natural to moment parameterization conversion
function nat2mom(x::SparseSplits)
    ρ = x.ref
    d = Dict(k=>exp(v) for (k,v) in x.splits)
    S = sum(values(d))
    Z = S + sum(x.k .* exp.(x.η0))
    for (k, v) in d
        d[k] /= Z
    end
    return SparseSplits(d, x.γ, x.n, x.k, exp.(x.η0)/Z, ρ)
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
function randsplit(x::SparseSplits)
    splitps = collect(x.splits)
    weights = last.(splitps)
    splits  = first.(splitps)
    ssizes  = splitsize.(Ref(x.γ), splits)
    pr = [w - x.η0[k] for (w, k) in zip(weights, ssizes)]
    if rand() < sum(pr)
        i = sample(1:length(pr), Weights(pr))
        return splits[i]
    else 
        k = sample(1:length(x.n), Weights(x.η0 .* x.n))
        return randsplitofsize(x.γ, k)
    end
end

# a random split for the Beta splitting model
function randsplit(γ, β)
    s = cladesize(γ)
    p = [betasplitf(β, s, i) for i=1:s÷2]
    k = sample(1:length(p), Weights(p))
    randsplitofsize(γ, k)
end

# a random split clade γ of size k 
function randsplitofsize(γ::T, k) where T
    g = digits(γ, base=2)  # binary expansion
    n = sum(g)             # number of leaves
    o = sort(shuffle(1:n)[1:k], rev=true)  # which leaves end up in left/right subclade
    # o = [a,b] means we obtain a split by taking/removing the ath and
    # bth leaf from γ. Note that this does not mean the ath and bth
    # index in the binary expansion, but rather the ath and bth one in
    # the binary expansion. The below ugliish while loop constructs
    # the requirred binary expansion of the split from `o`.
    j = pop!(o)
    i = 1  # index over g
    k = 1  # counts how many ones we've passed
    while true
        if g[i] == 1 && k == j
            g[i] = 0
            length(o) == 0 && break
            j = pop!(o)
            k += 1
        elseif g[i] == 1
            k += 1
        end
        i += 1
    end
    splt = T(evalpoly(2, g))
    left = min(splt, γ - splt)
    return left
end

# Define linear operations
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

