
# used for both natural and moment parameterizations
"""
    BetaSparseSplits
"""
struct BetaSparseSplits{T,V}
    splits::Dict{T,V}  # explicitly represented splits
    n  ::Int           # total number of splits
    k  ::Vector{Int}   # number of unrepresented splits of each size
    η0 ::Vector{V}     # parameter for unrepresented splits of size [1..|γ|-1]
    ref::T             # reference split
end

# unnormalized beta-splitting density (n,i)
function betasplitf(β, n, i) 
    gamma(β+1+i)*gamma(β+1+n-i)/(gamma(i+1)*gamma(n-i+1))
end

nsplits(n, i) = n == 2i ? binomial(n,i)÷2 : binomial(n,i)

# we define the split size to be the size of the smaller clade, note
# that this does not commute with the order on splits! i.e. a split
# is identified by the subclade with smallest id but this id need not
# be the id of the smallest subclade.
splitsize(γ, δ) = min(cladesize(δ), cladesize(γ-δ))

"""
Get a natural parameter BetaSparseSplits object from split counts `d`,
assuming the Beta-splitting Dirichlet prior with pseudo-count α and shape 
parameter β.
"""
function BetaSparseSplits(γ, d::Dict{T,V}, β, α) where {T,V}
    ρ = refsplit(γ)
    s = cladesize(γ)       # clade size
    n = _ns(s)             # total number of splits
    ns = [nsplits(s, i) for i=1:s÷2] 
    # pseudocounts at split level 
    as = map(i->α * betasplitf(β, s, i) / ns[i], 1:s÷2)
    aρ = as[splitsize(γ, ρ)]
    pρ = haskey(d, ρ) ? log(aρ + d[ρ]) : log(aρ)  # unnormalized pr of split ρ
    η0 = log.(as) .- pρ 
    xs = collect(d)
    ks = splitsize.(Ref(γ), first.(xs))
    dd = Dict(δ => log(as[k] + c) - pρ for (k, (δ, c)) in zip(ks, xs))
    kc = counts(ks, 1:s÷2)
    k  = ns .- kc
    BetaSparseSplits(dd, n, k, η0, ρ)
end

# Natural to moment parameterization conversion
function nat2mom(x::BetaSparseSplits)
    ρ = x.ref
    d = Dict(k=>exp(v) for (k,v) in x.splits)
    S = sum(values(d))
    Z = S + sum(x.k .* exp.(x.η0))
    for (k, v) in d
        d[k] /= Z
    end
    return BetaSparseSplits(d, x.n, x.k, exp.(x.η0)/Z, ρ)
end
