"""
    BranchModel{T,V}

The `BranchModel` stores for each clade, potentially, a natural
parameter vector, and has a fallback `η0` representing the natural
parameter vector for unrepresented clades. For the global model, 
the latter will be the prior. For an individual site `η0` will be
zero.
"""
struct BranchModel{T,V}
    cmap ::Dict{T,Vector{V}}  # clade => natural parameter 
    η0   ::Vector{V}          # natural parameter for unrepresented clade
end

# initialize an empty branchmodel object
BranchModel(T, η0::V) where V = BranchModel(Dict{T,V}(), η0)

# some accessors
Base.haskey(m::BranchModel, γ) = haskey(m.cmap, γ)
Base.getindex(m::BranchModel, γ) = haskey(m, γ) ? m.cmap[γ] : m.η0

# A branch model should support a vector algebra
function Base.:+(x::BranchModel{T,V}, y::BranchModel{T,V}) where {T,V}
    clades = union(keys(x.cmap), keys(y.cmap))
    d = Dict(γ=>x[γ] .+ y[γ] for γ in clades)
    return BranchModel(d, x.η0 .+ y.η0)
end

function Base.:-(x::BranchModel{T,V}, y::BranchModel{T,V}) where {T,V}
    clades = union(keys(x.cmap), keys(y.cmap))
    d = Dict(γ=>x[γ] .- y[γ] for γ in clades)
    return BranchModel(d, x.η0 .- y.η0)
end

Base.:*(a, x::BranchModel) = x * a
function Base.:*(x::BranchModel{T,V}, a::V) where {T,V}
    d = Dict(γ=>a*v for (γ, v) in x.cmap)
    BranchModel(d, a*x.η0)
end

# moment <-> natural transformations
# get a univariate gaussian from natural parameters
gaussian_nat2mom(η1, η2) = (-η1/(2η2), -1.0/(2η2))
gaussian_mom2nat(μ , V ) = (μ/V, -1.0/(2V))
gaussian_nat2mom(η::Vector) = [-η[1]/(2η[2]), -1.0/(2η[2])]
gaussian_mom2nat(θ::Vector) = [θ[1]/θ[2], -1.0/(2θ[2])]

# draw a random Gaussian number from natural parameterization
function randgaussian_nat(η1, η2) 
    μ, V = gaussian_nat2mom(η1, η2)
    return μ + √(V) * randn()
end

# Moment matching, i.e. get a BranchModel in moment space
# Note that each tree is associated with an (implicit) parameter
# vector for *all* clades, so that when a clade is not in a tree, we
# have to add a virtual draw from the cavity.
function matchmoments(trees, cavity::BranchModel{T,V}) where {T,V}
    d = Dict{T,Vector{V}}()
    # obtain moment estimates
    for tree in trees 
        _record_branchparams!(d, tree)
    end
    # add unrepresented prior samples (do or don't?)
    _cavity_contribution!(d, cavity, length(trees))
    # convert to natural parameters
    q = Dict(γ => _mom2nat(v[2], v[3], v[1]) for (γ, v) in d)
    BranchModel(q, cavity.η0)
end

# recursively process a tree to get the sufficient statistics for
# branch parameters
function _record_branchparams!(d, node)
    isleaf(node) && return id(node)
    left = _record_branchparams!(d, node[1]) 
    rght = _record_branchparams!(d, node[2]) 
    clade = left + rght
    x = log(node.data.distance)
    !haskey(d, clade) && (d[clade] = zeros(3))
    d[clade] .+= [1., x, x^2]
    return clade
end

# add the cavity (pseudo-prior) contribution to the moment estimates
# (for those clades which are not observed in all trees)
function _cavity_contribution!(d, cavity, N)
    for (γ, xs) in d
        n = N - xs[1]  # number of cavity draws to 'add'
        μ, V = gaussian_nat2mom(cavity[γ]...)
        d[γ][2] += n*μ
        d[γ][3] += n*(V + μ^2)
        d[γ][1] = N
    end
end

# compute moments from sufficient statistics and convert to natural
# parameters
function _mom2nat(xs, xsqs, N) 
    μ = xs/N
    V = xsqs/N - μ^2
    [gaussian_mom2nat(μ, V)...]
end

# draw random branch lengths for a given tree according to a BranchModel
function _randbranches!(node, q::BranchModel)
    if isleaf(node)
        node.data.distance == Inf
        return id(node)
    else
        left = _randbranches!(node[1], q)
        rght = _randbranches!(node[2], q)
        clade = left + rght
        η = q[clade]
        node.data.distance = exp(randgaussian_nat(η[1], η[2]))
        return clade
    end
end

"""
    MomBranchModel

BranchModel in moment parameter space, not really used, more for
convenience (interpretability).
"""
struct MomBranchModel{T,V}
    cmap ::Dict{T,Vector{V}}  # clade => natural parameter 
    η0   ::Vector{V}          # natural parameter for unrepresented clade
end

function MomBranchModel(q::BranchModel)
    m = Dict(γ=>gaussian_nat2mom(η) for (γ, η) in q.cmap)
    MomBranchModel(m, gaussian_nat2mom(q.η0))
end

function BranchModel(q::MomBranchModel)
    m = Dict(γ=>gaussian_mom2nat(η) for (γ, η) in q.cmap)
    BranchModel(m, gaussian_mom2nat(q.η0))
end

# pruning: note that we cannot just remove those clades not in the
# associated BMP. When we prune a BMP, we will remove for instance
# probability 1 clades with two leaves from the smap (since they need
# not be represented explicitly), however, their clade in the branch
# model should not be removed!
function prune(m::BranchModel, atol)
    d = Dict(γ=>v for (γ,v) in m.cmap if all(isapprox(v, m.η0, atol=atol)))
    BranchModel(d, m.η0)
end

function prune!(m::BranchModel, atol)
    for (γ, v) in m.cmap
        all(isapprox(v, m.η0, atol=atol)) && delete!(m.cmap, γ)
    end
end

