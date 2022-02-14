"""
    MSCModel

An object for conducting variational species tree inference under the
multispecies coalescent model. Note that:

- We approximate the tree topology posterior by a MBM
- We approximate the branch parameter posterior by independent
  Gaussians, one for each *clade*, i.e. representing the parameter for
  the branch leading to that clade as a crown group.
"""
struct MSCModel{T,V}
    S::NatMBM{T,V}       # species tree distribution approximation
    q::BranchModel{T,V}  # branch parameter distribution approximation
end

Base.show(io::IO, m::MSCModel) = write(io, "$(typeof(m))")

# linear operations
Base.:+(x::MSCModel, y::MSCModel) = MSCModel(x.S + y.S, x.q + y.q)
Base.:-(x::MSCModel, y::MSCModel) = MSCModel(x.S - y.S, x.q - y.q)
Base.:*(x::MSCModel, a) = MSCModel(x.S * a, x.q * a)
Base.:*(a, x::MSCModel) = MSCModel(x.S * a, x.q * a)

# efficiency of randtree/randbranches is similar
# randtree somewhat less efficient, but for larger models with more represented
# splits the difference is negligible, for instance
# julia> @btime randtree(alg.model);
#  114.373 μs (203 allocations: 122.53 KiB)
# julia> @btime randbranches(alg.model);
#  113.998 μs (129 allocations: 119.28 KiB)
# since working with trees is more elegant than the branches/splits structs,
# perhaps we should? The coalescent sims are also marginally faster in the tree
# case...
function randtree(model::MSCModel)
    tree = Node(model.S.root, Inf)
    _randtree(tree, model.S.root, model)
end

function _randtree(node, γ, model)
    isleafclade(γ) && return node
    left = randsplit(model.S, γ)
    rght = γ - left
    dl = randbranch(model.q, γ, left)
    dr = randbranch(model.q, γ, rght)
    nl = Node(left, dl, node)
    nr = Node(rght, dr, node)
    _randtree(nl, left, model)
    _randtree(nr, rght, model)
    return node
end
# NewickTree.distance(n::Node{<:Integer,Float64}) = n.data
# NewickTree.name(n::Node{<:Integer,Float64}) = n.id

# simulate a species tree from the MSCModel
# speed-up by pre-allocating and indexing, not pushing to en empty array?
function randbranches(model::MSCModel{T,V}) where {T,V}
    branches = Branches{T}()
    _randbranches(branches, model.S.root, model)
end

function _randbranches(branches, γ, model)
    isleafclade(γ) && return branches
    left = randsplit(model.S, γ)
    rght = γ - left
    dl = randbranch(model.q, γ, left)
    dr = randbranch(model.q, γ, rght)
    push!(branches, (γ, left, dl))
    push!(branches, (γ, rght, dr))
    branches = _randbranches(branches, left, model)
    branches = _randbranches(branches, rght, model)
    return branches
end

# simulate a gene tree from species tree branches
function randsplits(branches::Branches{T}, init::V) where {T,V}
    root = branches[1][1]
    splits = Splits{T}()
    inner = V()  # NOTE we cannot modify init -- messes with threads!
    for i=length(branches):-2:1
        γ, δ1, θ1 = branches[i]
        γ, δ2, θ2 = branches[i-1]
        # note, the copy is necessary here in case there are multiple lineages
        # in a tip branch... because _censoredcoalsplits modifies l1, l2...
        l1 = isleafclade(δ1) ? copy(init[δ1]) : inner[δ1]
        l2 = isleafclade(δ2) ? copy(init[δ2]) : inner[δ2]
        left, splits = _censoredcoalsplits!(splits, θ1, l1)
        rght, splits = _censoredcoalsplits!(splits, θ2, l2)
        inner[γ] = vcat(left, rght)
    end
    _, splits = _censoredcoalsplits!(splits, Inf, inner[root])
    return splits
end

function matchmoments(branches, weights, cavity, α)
    S = NatMBM(CCD(branches, weights), cavity.S.beta, α)
    q = matchmoments(branches, weights, cavity.q)
    return MSCModel(S, q)
end

function prune!(model::MSCModel, atol=1e-9)
    prune!(model.S, atol)
    prune!(model.q, atol)
end

"""
    logpartition(model::MSCModel)

Compute the logpartition function of the MSCModel.
"""
logpartition(model::MSCModel) = logpartition(model.S) + logpartition(model.q)

"""
    logpdf(model, branches)
"""
function logpdf(model::MSCModel, branches::Branches)
    l = 0.
    for i=1:2:length(branches)
        γ, δ, d = branches[i]
        l += splitpdf(model.S, γ, min(γ-δ, δ))
        l += logpdf(model.q, γ, δ, d)
        γ, δ, d = branches[i+1]
        l += logpdf(model.q, γ, δ, d)
    end
    return l
end

# Get the Gaussian approximations for the branch lengths under the model given
# a tree.
function getbranchapprox(model, splits::Splits)
    branches = mapreduce(x->[x, (x[1], x[1]-x[2])], vcat, splits)
    map(branches) do (γ, δ)
        μ, V = gaussian_nat2mom(model.q[(γ, δ)])
        (γ, δ, Normal(μ, √V))
    end
end

