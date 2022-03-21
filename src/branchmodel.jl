"""
    BranchModel{T,V}

The `BranchModel` stores for each clade, potentially, a natural parameter
vector, and has a fallback `η0` representing the natural parameter vector for
unrepresented clades. For the global model, the latter will be the prior. For
an individual site `η0` will be zero.
"""
struct BranchModel{T,V}
    root::T
    smap::Dict{Tuple{T,T},Vector{V}}  # (clade, split) => natural parameter 
    η0::Vector{V}          # natural parameter for unrepresented clade
    inftips::Vector{T}
end

# Initialize an empty branchmodel object.
function BranchModel(root::T, η0::V; inftips=T[]) where {T,V} 
    BranchModel(root, Dict{Tuple{T,T},V}(), η0, inftips)
end

# We define some accessor functions.
Base.haskey(m::BranchModel, γ) = haskey(m.smap, γ)
Base.getindex(m::BranchModel, γ) = haskey(m, γ) ? m.smap[γ] : m.η0
Base.getindex(m::BranchModel, γ, δ) = m[(γ,δ)] 


# Linear operations
# -----------------
# A branch model should support a vector algebra. We can have these in two
# flavors: non-mutating and mutating. The former being more elegant (and bug
# proof), while the latter being more efficient.

function Base.:+(x::BranchModel{T,V}, y::BranchModel{T,V}) where {T,V}
    clades = union(keys(x.smap), keys(y.smap))
    d = Dict(γ=>x[γ] .+ y[γ] for γ in clades)
    return BranchModel(x.root, d, x.η0 .+ y.η0, x.inftips)
end

function Base.:-(x::BranchModel{T,V}, y::BranchModel{T,V}) where {T,V}
    clades = union(keys(x.smap), keys(y.smap))
    d = Dict(γ=>x[γ] .- y[γ] for γ in clades)
    return BranchModel(x.root, d, x.η0 .- y.η0, x.inftips)
end

Base.:*(a, x::BranchModel) = x * a
function Base.:*(x::BranchModel{T,V}, a::V) where {T,V}
    d = Dict(γ=>a*v for (γ, v) in x.smap)
    BranchModel(x.root, d, a*x.η0, x.inftips)
end


# Mutating linear operations 
# --------------------------
# Mind the prior contribution! For a represented branch `b` in `x` and an
# unrepresented branch in `y`, one has to add `y.η0` to `x[b]` 

function add!(x::BranchModel, y::BranchModel)
    bs = union(keys(x.smap), keys(y.smap))
    for b in bs
        x.smap[b] = x[b] .+ y[b]
    end
    x.η0 .+= y.η0
    return x
end

function sub!(x::BranchModel, y::BranchModel)
    bs = union(keys(x.smap), keys(y.smap))
    for b in bs
        x.smap[b] = x[b] .- y[b]
    end
    x.η0 .-= y.η0
    return x
end

function mul!(x::BranchModel{T,V}, a::V) where {T,V}
    for b in collect(keys(x.smap))
        x.smap[b] .*= a
    end
    x.η0 .*= a
    return x
end


# moment <-> natural transformations
# get a univariate gaussian from natural parameters
gaussian_nat2mom(η1, η2) = (-η1/(2η2), -1.0/(2η2))
gaussian_mom2nat(μ , V ) = (μ/V, -1.0/(2V))
gaussian_nat2mom(η::Vector) = [-η[1]/(2η[2]), -1.0/(2η[2])]
gaussian_mom2nat(θ::Vector) = [θ[1]/θ[2], -1.0/(2θ[2])]

# a random branch length from the model
function randbranch(ϕ::BranchModel, γ, δ) 
    δ ∈ ϕ.inftips && return Inf
    return exp(randgaussian_nat(ϕ[(γ, δ)]))
end

# draw a random Gaussian number from natural parameterization
randgaussian_nat(η) = randgaussian_nat(η[1], η[2])

function randgaussian_nat(η1, η2) 
    μ, V = gaussian_nat2mom(η1, η2)
    return μ + √(V) * randn()
end

# Moment matching, i.e. get a BranchModel in moment space Note that each tree
# is associated with an (implicit) parameter vector for *all* clades, so that
# when a clade is not in a tree, we have to add a virtual draw from the cavity.
# for fixed tree analysis, correct dispatching
matchmoments(bs, ws, cavity::BranchModel, α) = matchmoments(bs, ws, cavity)

function matchmoments(branches, weights, cavity::BranchModel{T,V}) where {T,V}
    d = Dict{Tuple{T,T},Vector{V}}()
    # obtain moment estimates
    for (b, w) in zip(branches, weights)
        addtree!(d, b, w)
    end
    # add unrepresented prior samples
    cavity_contribution!(d, cavity)
    # convert to natural parameters
    ϕ = Dict(γ => suff2nat(v[2], v[3]) for (γ, v) in d)
    BranchModel(cavity.root, ϕ, copy(cavity.η0), cavity.inftips)
end

# recursively process a tree to get the sufficient statistics for
# branch parameters
function addtree!(d, b::Branches, w)
    for (γ, δ, x) in b
        !isfinite(x) && continue
        lx = log(x)  # input is on ℝ⁺
        branch = (γ, δ)
        !haskey(d, branch) && (d[branch] = zeros(3))
        d[branch][1] += w
        d[branch][2] += w*lx
        d[branch][3] += w*lx^2
    end
end

# add the cavity (pseudo-prior) contribution to the moment estimates
# (for those clades which are not either observed or unobserved in all trees)
function cavity_contribution!(d, cavity)
    for (γ, xs) in d
        w = 1. - xs[1]  # number of cavity draws to 'add'
        μ, V = gaussian_nat2mom(cavity[γ])
        d[γ][2] += w*μ
        d[γ][3] += w*(V + μ^2)
    end
end

# compute moments from sufficient statistics and convert to natural
# parameters
function suff2nat(μ, Esϕ) 
    V = Esϕ - μ^2
    [gaussian_mom2nat(μ, V)...]
end

function prune!(m::BranchModel, atol)
    for (γ, v) in m.smap
        all(isapprox(v, m.η0, atol=atol)) && delete!(m.smap, γ)
    end
end

# should the 2π factor appear somewhere?
gaussian_logpartition(η1, η2) = -η1^2/4η2 - 0.5log(-2η2)

function logpartition(m::BranchModel)
    n = cladesize(m.root)
    N = 3^n - 2^(n+1) + 1  # total number of parameters 
    Z = 0.
    for (k, v) in m.smap
        Z += gaussian_logpartition(v[1], v[2])
    end
    Z += (N-length(m.smap)) * gaussian_logpartition(m.η0[1], m.η0[2])
    return Z
end

"""
    logpdf(m::BranchModel, clade, subclade, length)
"""
function logpdf(m::BranchModel, γ, δ, d)
    μ, V = gaussian_nat2mom(m[(γ, δ)])
    if isnan(V)  
        μ, V = gaussian_nat2mom(m.η0)
    end
    return logpdf(Normal(μ, √V), log(d))
end

"""
    logpdf(model, branches)
"""
function logpdf(model::BranchModel, branches::Branches)
    l = 0.
    for i=1:length(branches)
        γ, δ, d = branches[i]
        isfinite(d) && (l += logpdf(model, γ, δ, d))
    end
    return l
end

function getbranches(n::DefaultNode, m::AbstractDict{T}) where {T}
    nn = length(postwalk(n))-1
    branches = Branches(undef, T, nn)
    _getbranches(branches, n, m, 1)
    return branches #reverse(branches)  # return in preorder
end

function _getbranches(branches, n, m, i)
    isleaf(n) && return m[name(n)], distance(n), i
    a, da, j = _getbranches(branches, n[1], m, i+2)
    b, db, j = _getbranches(branches, n[2], m, j)
    branches[i] = (a + b, a, da)
    branches[i+1] = (a + b, b, db)
    return a + b, distance(n), j
end

function getbranchdict(n, m)
    Dict((x[1],x[2])=>x[3] for x in getbranches(n, m))
end

