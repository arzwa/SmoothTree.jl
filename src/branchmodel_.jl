# There seems to be no good reason why we define the branch model at the clade
# level and not the split level...

"""
    BranchModel{T,V}

The `BranchModel` stores for each clade, potentially, a natural
parameter vector, and has a fallback `η0` representing the natural
parameter vector for unrepresented clades. For the global model, 
the latter will be the prior. For an individual site `η0` will be
zero.
"""
struct BranchModel{T,V}
    root ::T
    cmap ::Dict{Tuple{T,T},Vector{V}}  # (clade, split) => natural parameter 
    η0   ::Vector{V}          # natural parameter for unrepresented clade
end

# initialize an empty branchmodel object
BranchModel(root::T, η0::V) where {T,V} = BranchModel(root, Dict{Tuple{T,T},V}(), η0)

# some accessors
Base.haskey(m::BranchModel, γ) = haskey(m.cmap, γ)
Base.getindex(m::BranchModel, γ) = haskey(m, γ) ? m.cmap[γ] : m.η0

# A branch model should support a vector algebra
function Base.:+(x::BranchModel{T,V}, y::BranchModel{T,V}) where {T,V}
    clades = union(keys(x.cmap), keys(y.cmap))
    d = Dict(γ=>x[γ] .+ y[γ] for γ in clades)
    return BranchModel(x.root, d, x.η0 .+ y.η0)
end

function Base.:-(x::BranchModel{T,V}, y::BranchModel{T,V}) where {T,V}
    clades = union(keys(x.cmap), keys(y.cmap))
    d = Dict(γ=>x[γ] .- y[γ] for γ in clades)
    return BranchModel(x.root, d, x.η0 .- y.η0)
end

Base.:*(a, x::BranchModel) = x * a
function Base.:*(x::BranchModel{T,V}, a::V) where {T,V}
    d = Dict(γ=>a*v for (γ, v) in x.cmap)
    BranchModel(x.root, d, a*x.η0)
end

# moment <-> natural transformations
# get a univariate gaussian from natural parameters
gaussian_nat2mom(η1, η2) = (-η1/(2η2), -1.0/(2η2))
gaussian_mom2nat(μ , V ) = (μ/V, -1.0/(2V))
gaussian_nat2mom(η::Vector) = [-η[1]/(2η[2]), -1.0/(2η[2])]
gaussian_mom2nat(θ::Vector) = [θ[1]/θ[2], -1.0/(2θ[2])]

# a random branch length from the model
randbranch(q::BranchModel, γ, δ) = exp(randgaussian_nat(q[(γ, δ)]))

# draw a random Gaussian number from natural parameterization
randgaussian_nat(η) = randgaussian_nat(η[1], η[2])
function randgaussian_nat(η1, η2) 
    μ, V = gaussian_nat2mom(η1, η2)
    return μ + √(V) * randn()
end

# Moment matching, i.e. get a BranchModel in moment space Note that each tree
# is associated with an (implicit) parameter vector for *all* clades, so that
# when a clade is not in a tree, we have to add a virtual draw from the cavity.
function matchmoments(branches, weights, cavity::BranchModel{T,V}) where {T,V}
    d = Dict{Tuple{T,T},Vector{V}}()
    # obtain moment estimates
    for (b, w) in zip(branches, weights)
        addtree!(d, b, w)
    end
    # add unrepresented prior samples
    cavity_contribution!(d, cavity)
    # convert to natural parameters
    q = Dict(γ => suff2nat(v[2], v[3]) for (γ, v) in d)
    BranchModel(cavity.root, q, cavity.η0)
end

# recursively process a tree to get the sufficient statistics for
# branch parameters
function addtree!(d, b::Branches, w)
    for (γ, δ, x) in b
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
function suff2nat(μ, Esq) 
    V = Esq - μ^2
    [gaussian_mom2nat(μ, V)...]
end

function prune!(m::BranchModel, atol)
    for (γ, v) in m.cmap
        all(isapprox(v, m.η0, atol=atol)) && delete!(m.cmap, γ)
    end
end

# should the 2π factor appear somewhere?
gaussian_logpartition(η1, η2) = -η1^2/4η2 - 0.5log(-2η2)

function logpartition(m::BranchModel)
    n = cladesize(m.root)
    N = 3^n - 2^(n+1) + 1  # total number of parameters 
    Z = 0.
    for (k, v) in m.cmap
        Z += gaussian_logpartition(v[1], v[2])
    end
    Z += (N-length(m.cmap)) * gaussian_logpartition(m.η0[1], m.η0[2])
    return Z
end

"""
    logpdf(m::BranchModel, clade, subclade, length)
"""
function logpdf(m::BranchModel, γ, δ, d)
    μ, V = gaussian_nat2mom(m[(γ, δ)])
    if isnan(V)  # happens...
        μ, V = gaussian_nat2mom(m.η0)
    end
    return logpdf(Normal(μ, √V), log(d))
end

