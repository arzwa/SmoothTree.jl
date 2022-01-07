# Hold for each clade, potentially, the natural parameters of a
# Gaussian, but only store explicitly when distinct from the prior.
struct BranchModel{T,V}
    cmap ::Dict{T,Vector{V}}  # clade => natural parameter 
    prior::Vector{V} # natural parameter for prior
end

# initialize an empty branchmodel object
BranchModel(x::NatBMP{T}, prior::V) where {T,V} = BranchModel(Dict{T,V}(), prior)

# get the cavity distribution
function cavity(qfull::BranchModel{T,V}, qsite) where {T,V}
    qcav = Dict{T,Vector{V}}()
    for (clade, η) in qfull.cmap
        qcav[clade] = haskey(qsite, clade) ? η - qsite[clade] : η
    end 
    return BranchModel(qcav, qfull.prior)
end

# some accessors
Base.haskey(m::BranchModel, γ) = haskey(m.cmap, γ)
Base.getindex(m::BranchModel, γ) = haskey(m, γ) ? m.cmap[γ] : m.prior

function _randbranches!(node, q)
    if isleaf(node)
        node.data.distance == Inf
        return id(node)
    else
        left = _randbranches!(node[1], q)
        rght = _randbranches!(node[2], q)
        clade = left + rght
        η = haskey(q, clade) ? q[clade] : q.prior
        node.data.distance = exp(randgaussian_nat(η[1], η[2]))
        return clade
    end
end

function randgaussian_nat(η1, η2) 
    μ, V = gaussian_nat2mom(η1, η2)
    return μ + √(V) * randn()
end

# get a univariate gaussian from natural parameters
gaussian_nat2mom(η1, η2) = (-η1/(2η2), -1.0/(2η2))
gaussian_mom2nat(μ , V ) = (μ/V, -1.0/(2V))

# moment matching
function approximate_tilted(trees, qcavity::BranchModel{T,V}) where {T,V}
    d = Dict{T,Vector{V}}()
    # obtain moment estimates
    for tree in trees 
        _record_branchparams!(d, tree)
    end
    # add unrepresented prior samples (do or don't?)
    _cavity_contribution!(d, qcavity, length(trees))
    BranchModel(d, qcavity.prior)
end

# λx + (1-λ)y
function convexcombination(x, y, λ)
    q = Dict(γ => _mom2nat(v[2], v[3], v[1]) for (γ, v) in x.cmap)
    for (γ, η1) in q # convex combination in η space (damped update)
        η2 = haskey(y, γ) ? y[γ] : y.prior
        q[γ] = (λ .* η1) .+ ((1-λ) .* η2) 
    end
    return BranchModel(q, x.prior)
end

function _record_branchparams!(d, node)
    isleaf(node) && return id(node) #lmap[name(node)]
    left = _record_branchparams!(d, node[1]) 
    rght = _record_branchparams!(d, node[2]) 
    clade = left + rght
    x = log(node.data.distance)
    if !haskey(d, clade)
        d[clade] = zeros(3)
    end
    d[clade] .+= [1., x, x^2]
    return clade
end

# add the cavity (pseudo-prior) contribution to the moment estimates
function _cavity_contribution!(d, q, N)
    for (γ, xs) in d
        n = N - xs[1]  # number of cavity draws to 'add'
        η = haskey(q, γ) ? q[γ] : q.prior
        μ, V = gaussian_nat2mom(η...)
        d[γ][2] += n*μ
        d[γ][3] += n*(V + μ^2)
        d[γ][1] = N
    end
end

function _mom2nat(xs, xsqs, N) 
    μ = xs/N
    V = xsqs/N - μ^2
    [gaussian_mom2nat(μ, V)...]
end

# Vector algebra
# if a clade is not present, it means that its branch length
# distribution is given by the prior
function Base.:+(x::BranchModel{T,V}, y::BranchModel{T,V}) where {T,V}
    clades = union(keys(x.cmap), keys(y.cmap))
    d = Dict(γ=>zeros(2) for γ in clades)
    for γ in clades
        d[γ] .+= haskey(x, γ) ? x[γ] : x.prior
        d[γ] .+= haskey(y, γ) ? y[γ] : y.prior
    end
    return BranchModel(d, x.prior .+ y.prior)
end

function Base.:*(x::BranchModel{T,V}, a::V) where {T,V}
    d = Dict(γ=>a*v for (γ, v) in x.cmap)
    BranchModel(d, a*x.prior)
end

