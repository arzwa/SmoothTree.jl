# Need a way to keep the general implementation, but where in the case of tip
# branches with a single sample we do not store the approximation...

"""
    MSCModel

An object for conducting variational species tree inference under the
multispecies coalescent model. Note that:

- We approximate the tree topology posterior by a MBM
- We approximate the branch parameter posterior by independent
  Gaussians, one for each *clade*, i.e. representing the parameter for
  the branch leading to that clade as a crown group.
"""
struct MSCModel{T,V,M,P}
    S::CCD{T,M,P}        # species tree distribution approximation
    ϕ::BranchModel{T,V}  # branch parameter distribution approximation
end

Base.show(io::IO, m::MSCModel) = write(io, "$(typeof(m))")

# linear operations
Base.:+(x::MSCModel, y::MSCModel) = MSCModel(x.S + y.S, x.ϕ + y.ϕ)
Base.:-(x::MSCModel, y::MSCModel) = MSCModel(x.S - y.S, x.ϕ - y.ϕ)
Base.:*(x::MSCModel, a) = MSCModel(x.S * a, x.ϕ * a)
Base.:*(a, x::MSCModel) = MSCModel(x.S * a, x.ϕ * a)
Base.copy(x::MSCModel) = MSCModel(copy(x.S), copy(x.ϕ))

function add!(x::MSCModel, y::MSCModel)
    add!(x.S, y.S)
    add!(x.ϕ, y.ϕ)
    return x
end

function sub!(x::MSCModel, y::MSCModel)
    sub!(x.S, y.S)
    sub!(x.ϕ, y.ϕ)
    return x
end

function mul!(x::MSCModel, a)
    mul!(x.S, a)
    mul!(x.ϕ, a)
    return x
end

# simulate a species tree from the MSCModel
# simulate species tree branches, modifying a pre-existing `Branches` vector
function randbranches(m::MSCModel{T}) where T
    n = cladesize(m.S.root)
    b = Branches(undef, T, 2n-2)
    randbranches!(b, m)
    return b
end
randbranches!(b, m::MSCModel) = randbranches!(b, m.S, m.ϕ)
randbranches!(branches, S, ϕ) = _randbranches!(branches, 1, S.root, S, ϕ)

function _randbranches!(branches, i, γ, S, ϕ)
    isleafclade(γ) && return i
    left = randsplit(S, γ)
    rght = γ - left
    dl = randbranch(ϕ, γ, left)
    dr = randbranch(ϕ, γ, rght)
    branches[i]   = (γ, left, dl)
    branches[i+1] = (γ, rght, dr)
    j = _randbranches!(branches, i+2, left, S, ϕ)
    j = _randbranches!(branches, j, rght, S, ϕ)
    return j
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

function matchmoments(branches, weights, cavity::MSCModel, α)
    S = CCD(SplitCounts(branches, weights), cavity.S.prior, α)
    ϕ = matchmoments(branches, weights, cavity.ϕ)
    return MSCModel(S, ϕ)
end

function prune!(model::MSCModel, atol=1e-9)
    prune!(model.S, atol)
    prune!(model.ϕ, atol)
end

"""
    logpartition(model::MSCModel)

Compute the logpartition function of the MSCModel.
"""
logpartition(model::MSCModel) = logpartition(model.S) + logpartition(model.ϕ)

"""
    logpdf(model, branches)
"""
function logpdf(model::MSCModel, branches::Branches)
    l = 0.
    for (γ, δ, d) in branches
        (δ < γ - δ) && (l += logpdf(model.S, γ, δ))
        isfinite(d) && (l += logpdf(model.ϕ, γ, δ, d))
    end
    return l
end

# Get the Gaussian approximations for the branch lengths under the model given
# a tree.
function getbranchapprox(ϕ, splits::Branches)
    map(splits) do (γ, δ, _)
        μ, V = gaussian_nat2mom(ϕ[(γ, δ)])
        (γ, δ, Normal(μ, √V))
    end
end

"""
    traceback(trace::Vector{<:MSCModel})

Trace back the history of the approximation.
"""
function traceback(trace::Vector{<:MSCModel})
    ss = allsplits(trace[end].S)
    bs = allbranches(trace[end].S)
    Ms = MomMBM.(getfield.(trace, :S))
    ϕs = getfield.(trace, :ϕ)
    strace = mapreduce(x->[x[γ,δ] for (γ,δ) in ss], hcat, Ms) |> permutedims
    btrace = mapreduce(x->[x[γ,δ] for (γ,δ) in bs], hcat, ϕs) |> permutedims
    X = gaussian_nat2mom.(btrace)
    return (θ=strace, μ=first.(X), V=last.(X))
end

# assumes the MAP tree is represented (otherwise not well defined anyhow...) 
function maptree(model::MSCModel{T}, spmap) where T
    _maptree(Node(model.S.root), model.S.root, model.S, model.ϕ, spmap)
end

function _maptree(node, γ, S, ϕ, spmap)
    isleafclade(γ) && return node
    if ischerry(γ)
        left = randsplit(S, γ)
    else
        #ss = tomoment(S[γ])
        ps = collect(S[γ].splits)
        left = argmax(last, ps)[1]
    end
    rght = γ - left
    nl = isleafclade(left) ? spmap[left] : ""
    nr = isleafclade(rght) ? spmap[rght] : ""
    dl = exp(gaussian_nat2mom(ϕ[(γ,left)])[1])
    dr = exp(gaussian_nat2mom(ϕ[(γ,rght)])[1])
    push!(node, Node(left, d=dl, n=nl))
    push!(node, Node(rght, d=dr, n=nr))
    _maptree(node[1], left, S, ϕ, spmap)
    _maptree(node[2], rght, S, ϕ, spmap)
    return node
end

function addmapbranches!(tree, spmap, ϕ::BranchModel)
    function walk(n)
        isleaf(n) && return n, spmap[name(n)]
        c1, δ1 = walk(n[1])
        c2, δ2 = walk(n[2])
        γ = δ1 + δ2
        c1.data.distance = exp(gaussian_nat2mom(ϕ[(γ,δ1)])[1])
        c2.data.distance = exp(gaussian_nat2mom(ϕ[(γ,δ2)])[1])
        return n, γ
    end     
    walk(tree)
    return tree
end


