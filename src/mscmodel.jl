"""
    MSCModel

An object for conducting variational species tree inference under the
multispecies coalescent model. Note that:

- We approximate the tree topology posterior by a BMP
- We approximate the branch parameter posterior by independent
  Gaussians, one for each *clade*, i.e. representing the parameter for
  the branch leading to that clade as a crown group.
"""
struct MSCModel{T,V,W}
    S::NatBMP{T,V}       # species tree distribution approximation
    q::BranchModel{T,W}  # branch parameter distribution approximation
    m::BiMap{T,String}   # species label to clade map
end

Base.show(io::IO, m::MSCModel) = write(io, "$(typeof(m))")

# initialize a MSCModel
MSCModel(x::NatBMP, θprior, m) = MSCModel(x, BranchModel(x, θprior), m)

# linear operations
Base.:+(x::MSCModel, y::MSCModel) = MSCModel(x.S + y.S, x.q + y.q, x.m)
Base.:-(x::MSCModel, y::MSCModel) = MSCModel(x.S - y.S, x.q - y.q, x.m)
Base.:*(x::MSCModel, a) = MSCModel(x.S * a, x.q * a, x.m)
Base.:*(a, x::MSCModel) = MSCModel(x.S * a, x.q * a, x.m)

# In the EP algorithm we sample a lot of species trees, using the
# moment-space BMP
struct MSCSampler{T,V,W}
    S::MomBMP{T,V}
    q::BranchModel{T,W}
end

MSCSampler(m::MSCModel) = MSCSampler(MomBMP(m.S), m.q)

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
matches the moments of the BMP distribution to the Dirichlet-BMP
posterior with prior `α` for the accepted trees, and updates the
Gaussian distributions for the branch parameters.
"""
function matchmoments(trees, cavity::MSCModel{T}, α) where T
    m = shitmap(trees[1], T)  # XXX sucks?
    S = NatBMP(CCD(trees, lmap=m, α=α))#, αroot=0.)) respect rooting?
    q = matchmoments(trees, cavity.q)
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
