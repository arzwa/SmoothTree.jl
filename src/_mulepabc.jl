
mutable struct MULEPABC{X,M} <: AbstractEPABC
    data ::X
    model::M
    sites::Vector{M}
    λ    ::Float64  # for damped update...
    α    ::Float64  # Dirichlet-BMP parameter for 'moment matching'
    speciesmap::Dict{String,Vector{String}}  # maps species name -> subgenomes
end

function MULEPABC(data, model::T, speciesmap; λ=1., α=0.1) where T
    sites = Vector{T}(undef, length(data))
    MULEPABC(data, model, sites, λ, α, speciesmap)
end

function ep_iteration!(alg::MULEPABC, i; mina=10, target=100,
                       maxn=1e5, noisy=false, adhoc=0.)
    @unpack data, model, sites = alg
    x = data[i]
    cavity = isassigned(sites, i) ? getcavity(model, sites[i]) : model
    S = randsptree(cavity)
    # XXX the init is where the gene to species mapping happens!
    accepted = typeof(S)[]
    nacc = n = 0
    while true   # this could be parallelized to some extent using blocks
        n += 1
        init = randinit(x, model.m, alg.speciesmap)
        G = randsplits(MSC(S, init))
        l = logpdf(x, G) + adhoc
        noisy && n % 1000 == 0 && (@info "$n $l")
        if log(rand()) < l
            noisy && (@info "accepted! ($nacc)")
            push!(accepted, S)
            nacc += 1
        end
        (n ≥ maxn || nacc ≥ target) && break
        S = randsptree(cavity)
    end
    nacc < mina && return false, nacc, n, alg.model, cavity
    model_ = updated_model(accepted, model, cavity, alg)
    site_  = new_site(model_, cavity)
    return true, nacc, n, model_, site_
end

function randinit(x::CCD{T}, tmap, speciesmap) where T
    # sample subgenomes without replacement!
    urn = Dict(x=>shuffle(y) for (x,y) in speciesmap)
    init = Dict{T,Vector{T}}()
    for (γ, gene) in x.lmap
        species = _spname(gene)
        subgenome = pop!(urn[species])
        spγ = tmap[subgenome]
        init[spγ] = [γ]
    end
    return init
end

