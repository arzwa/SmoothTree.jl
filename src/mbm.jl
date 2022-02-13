# We have use for an abstract type here
abstract type AbstractMBM{T,V} end

# XXX: We might not need the MomMBM after all, perhaps only for
# interpretability

# XXX: to be correct, we should have the betasplit distribution also
# transformed between moment/natural spaces, but we will not need it
# in natural parameter space, since we store it for the sake of
# unrepresented clades...

"""
    NatMBM

A MBM model in natural parameter space.
"""
struct NatMBM{T,V} <: AbstractMBM{T,V}
    beta::BetaSplitTree{V}
    smap::Dict{T,SparseSplits{T,V}}
    root::T
end

"""
    MomMBM

A MBM model in moment parameter space.
"""
struct MomMBM{T,V} <: AbstractMBM{T,V}
    beta::BetaSplitTree{V}
    smap::Dict{T,SparseSplits{T,V}}
    root::T
end

# accessors
Base.show(io::IO, m::M) where M<:AbstractMBM = write(io, "$M(o=$(m.root))")
Base.haskey(m::AbstractMBM, γ) = haskey(m.smap, γ)
Base.haskey(m::AbstractMBM, γ, δ) = haskey(m, γ) && haskey(m.smap[γ], δ)
Base.getindex(m::AbstractMBM, γ) = m.smap[γ]
Base.getindex(m::AbstractMBM, γ, δ) = m.smap[γ][δ]

# 'empty' MBM (uniform on splits, this does *not* correspond to the
# zero MBM...)
const SplitDict{T} = Dict{T,SparseSplits{T,Float64}}
NatMBM(root::T, β) where T<:Integer = NatMBM(β, SplitDict{T}(), root)
MomMBM(root::T, β) where T<:Integer = MomMBM(β, SplitDict{T}(), root)

# natural parameter -> moment parameter
# XXX note: this ignores the beta split prior distribution
MomMBM(x::NatMBM) = MomMBM(x.beta, Dict(k=>nat2mom(x) for (k,x) in x.smap), x.root)
NatMBM(x::MomMBM) = NatMBM(x.beta, Dict(k=>mom2nat(x) for (k,x) in x.smap), x.root)

# get the mean MBM implied by a Dirichlet-MBM
"""
    MomMBM(x::CCD, β::BetaSplitTree, α::Real)
    NatMBM(...)

Get the posterior mean Markov branching model assuming a Dirichlet
prior distribution with parameter proportional to a Beta-splitting
model and weight `α` (i.e. the Dirichlet parameter vector for each
Categorical split distribution sums to `α`, so that `α` serves as a
total pseudocount) and observed splits recorded in `x`.
"""
MomMBM(x::CCD, args...) = MomMBM(NatMBM(x, args...))
function NatMBM(x::CCD, β::BetaSplitTree, α::Real)
    α == 0. && @warn "α = 0 is not well defined"
    # note that the ccd contains cherries, which is not informative
    smap = Dict(γ=>SparseSplits(γ, d, β, α) for (γ, d) in x.smap if !ischerry(γ))
    NatMBM(β, smap, x.root)
end

# linear operations
# XXX note: these ignore the beta split prior distribution
function Base.:+(x::NatMBM{T,V}, y::NatMBM{T,V}) where {T,V}
    d = Dict(γ=>v for (γ, v) in x.smap)
    for (γ, v) in y.smap
        d[γ] = haskey(d, γ) ? d[γ] + v : v
    end
    NatMBM(x.beta, d, x.root) 
end

function Base.:-(x::NatMBM{T,V}, y::NatMBM{T,V}) where {T,V}
    d = Dict(γ=>v for (γ, v) in x.smap)
    for (γ, v) in y.smap
        d[γ] = haskey(d, γ) ? d[γ] - v : -1.0*v
    end
    NatMBM(x.beta, d, x.root) 
end

Base.:*(a::V, x::NatMBM{T,V}) where {T,V} = x*a
function Base.:*(x::NatMBM{T,V}, a::V) where {T,V}
    NatMBM(x.beta, Dict(γ=>v*a for (γ,v) in x.smap), x.root)
end

# randtree for moment parameters
randtree(model::MomMBM) = _randwalk(Node(model.root), model)

# generic extension of randtree (also works for MSC)
randtree(model, n) = map(_->randtree(model), 1:n)

# randtree for natural parameter SparseSplits  # FIXME? 
# I guess no need to fix this, but we should take this into account in
# the EP algorithm by only converting the cavity once to moment space
# and simulating species trees from that MomMBM...
randtree(model::NatMBM) = randtree(MomMBM(model))
randtree(model::NatMBM, n) = randtree(MomMBM(model), n)

# recursion for randtree
function _randwalk(node, model::MomMBM)
    clade = id(node)
    isleafclade(clade) && return node
    splt = randsplit(model, clade)
    n1 = Node(splt, node)
    n2 = Node(clade - splt, node)
    _randwalk(n1, model)
    _randwalk(n2, model)
    return node
end

function randsplits(model::MomMBM{T}) where T
    _randsplits(Tuple{T,T}[], model.root, model)
end

function _randsplits(splits, γ, model)
    isleafclade(γ) && return splits
    left = randsplit(model, γ)
    rght = γ - left
    push!(splits, (γ, min(left, rght)))
    splits = _randsplits(splits, left, model)
    splits = _randsplits(splits, rght, model)
    return splits
end

function randsplit(m::MomMBM, γ)
    # a cherry clade has NaN entries in SparseSplits
    haskey(m, γ) && !ischerry(γ) ? randsplit(m[γ]) : randsplit(m.beta, γ)
end

function randsplit(m::NatMBM, γ)
    # a cherry clade has NaN entries in SparseSplits
    haskey(m, γ) && !ischerry(γ) ? randsplit(nat2mom(m[γ])) : randsplit(m.beta, γ)
end

# we implement logpdf for both Mom/Nat
function logpdf(m::MomMBM, splits::Vector{T}) where T<:Tuple
    ℓ = 0.
    for (γ, δ) in splits
        ℓ += splitpdf(m, γ, δ)
    end
    return ℓ
end

splitpdf(m::MomMBM, γ, δ) = haskey(m, γ) ? log(m[γ][δ]) : logpdf(m.beta, γ, δ)

function logpdf(m::NatMBM, splits::Splits)
    ℓ = 0.
    for (γ, δ) in splits
        ℓ += splitpdf(m, γ, δ)
    end
    return ℓ
end

function splitpdf(m::NatMBM, γ, δ) 
    !haskey(m, γ) && return logpdf(m.beta, γ, δ)
    x = m[γ]
    y = exp.(collect(values(x.splits)))
    Z = sum(y) + sum(x.k .* exp.(x.η0))
    return m[γ][δ] - log(Z)
end

"""
    prune

Prune a sparsely represented MBM object by setting all represented
splits with split probabilities indistinguishable from the probability
of an unrepresented split to the latter (thereby removing the split
from the set of explicitly represented splits).
"""
function prune(x::M, atol) where {T,V,M<:AbstractMBM{T,V}}
    newd = Dict{T,SparseSplits{T,V}}()
    clades = Set(x.root)
    # first we prune all splits with negligible probability
    for (γ, x) in x.smap
        newd[γ] = prune(x, atol)
        union!(clades, keys(newd[γ].splits))
    end
    # then we prune clades which feature explicitly in none of the
    # split distributions
    toprune = setdiff(keys(newd), clades)
    for γ in toprune
        delete!(newd, γ)
    end
    return M(newd, x.root)
end

function prune!(x::M, atol) where {T,V,M<:AbstractMBM{T,V}}
    clades = Set(x.root)
    for (γ, x) in x.smap
        prune!(x, atol)
        # all splits with non-negligible probabilities are to be kept
        # note that we also need to keep the complements, which are
        # not in the split distribution of γ but may have their own
        # split distribution in the smap!
        union!(clades, keys(x.splits))
        union!(clades, γ .- keys(x.splits))  
    end
    # those clades nowhere seen will be deleted
    toprune = setdiff(keys(x.smap), clades)  
    for γ in toprune
        delete!(x.smap, γ)
    end
end

function allsplits(x::AbstractMBM)
    map(collect(x.smap)) do (γ, splits)
        map(x->(γ,x), collect(keys(splits.splits)))
    end |> x->reduce(vcat, x)
end

"""
    logpartition(x::AbstractMBM)

Compute the log-partition function for `x`.

The log-partition function of a categorical distribution on k categories with
moment parameter `θ = (θ1, θ2, …, θ{k-1})` is `-log(1-∑i θi) = -log θk`. The
MBM defines a categorical distribution on tree topologies. We have defined an
order on trees (in particular we have a well-defined last tree, see `reftree`).
So it appears we can easily compute -log θk. 
"""
logpartition(x::NatMBM) = -logpdf(MomMBM(x), reftree(x.root)) 
logpartition(x::MomMBM) = -logpdf(x, reftree(x.root)) 

# get the last tree in the induced tree order
function reftree(x::T) where T
    splits = Tuple{T,T}[]
    function walk(x)
        isleafclade(x) && return 
        a = refsplit(x)
        b = x - a
        push!(splits, (x, a))
        walk(a)
        walk(b)
    end
    walk(x)
    return splits
end

