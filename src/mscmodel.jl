"""
    MSCModel

An object for conducting variational species tree inference under the
multispecies coalescent model. Note that:

- We approximate the tree topology posterior by a MBM
- We approximate the branch parameter posterior by independent
  Gaussians, one for each *clade*, i.e. representing the parameter for
  the branch leading to that clade as a crown group.
"""
struct MSCModel{T,V,U,W}
    S::NatMBM{T,V}       # species tree distribution approximation
    q::BranchModel{U,W}  # branch parameter distribution approximation
    m::BiMap{T,String}   # species label to clade map
end

Base.show(io::IO, m::MSCModel) = write(io, "$(typeof(m))")

# linear operations
Base.:+(x::MSCModel, y::MSCModel) = MSCModel(x.S + y.S, x.q + y.q, x.m)
Base.:-(x::MSCModel, y::MSCModel) = MSCModel(x.S - y.S, x.q - y.q, x.m)
Base.:*(x::MSCModel, a) = MSCModel(x.S * a, x.q * a, x.m)
Base.:*(a, x::MSCModel) = MSCModel(x.S * a, x.q * a, x.m)

# In the EP algorithm we sample a lot of species trees, using the
# moment-space MBM
struct MSCSampler{T,V,U,W}
    S::MomMBM{T,V}
    q::BranchModel{U,W}
end

MSCSampler(m::MSCModel) = MSCSampler(MomMBM(m.S), m.q)

Base.eltype(m::MSCSampler{T,V,U,W}) where {T,V,U,W} = Node{T,NewickData{W,String}}

"""
    randtree(model::MSCSampler)

Simulate a species tree from an MSCModel (in the EP-ABC algorithm this
is used to simulate from the cavity)
"""
function randtree(model::MSCSampler)
    S = randtree(model.S)  # a random tree topology
    _randbranches!(S, model.q)
    return S
end
# NOTE: we are not yet dealing with the rooted case, where the
# branches stemming from the root should have infinite length...
randtree(model::MSCModel) = randtree(MSCSampler(model))
randtree(model::MSCModel, n) = randtree(MSCSampler(model), n)

"""
    updated_model(trees, cavity, α)

Method to update the full approximation by moment matching.  This
matches the moments of the MBM distribution to the Dirichlet-MBM
posterior with prior `α` for the accepted trees, and updates the
Gaussian distributions for the branch parameters.
"""
function matchmoments(trees, cavity::MSCModel{T}, α) where T
    m = shitmap(trees[1], T)  # XXX sucks?
    S = NatMBM(CCD(trees, m), cavity.S.beta, α)#, αroot=0.)) respect rooting?
    q = matchmoments(trees, cavity.q)
    # XXX we should get the CCD and branch lengths in one pass over `trees`
    return MSCModel(S, q, cavity.m)
end

# a weighted sample (importance sampling)
function matchmoments(trees, weights, cavity::MSCModel{T}, α) where T
    m = shitmap(trees[1], T)  # XXX sucks?
    S = NatMBM(CCD(zip(trees, weights), m, Float64), cavity.S.beta, α)
    q = matchmoments(trees, weights, cavity.q)
    # XXX we should get the CCD and branch lengths in one pass over `trees`
    return MSCModel(S, q, cavity.m)
end

shitmap(tree, T) = BiMap(Dict(T(id(n))=>name(n) for n in getleaves(tree)))
# issue is that we are using the id field here (which is good here
# internally, not in general), but `CCD` uses a taxonmap and the name
# field (good in general)...
   
function prune(model::MSCModel, atol=1e-9)
    S = prune(model.S, atol)
    q = prune(model.q, atol)
    return MSCModel(S, q, model.m)
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

