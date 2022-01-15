"""
    kldiv(p::CCD, q::CCD)

Compute the (a) KL divergence `d(p||q) = ∑ₓp(x)log(p(x)/q(x))`.

Not sure how exactly we should do this. For each clade compute the
kldivergence for its split distribution, and weight these
kldivergences by the probability that a tree contains the clade?

This is hacky, we get something, but I don't think we can call it the
KL divergence...
"""
function kldiv(p::CCD, q::CCD)  
    D = 0. 
    # there are four different cases:
    # 1. clade observed in both
    # 2. clade observed in p, not in q
    # 3. clade observed in q, not in p
    # 4. clade observed in neither p nor q
    # in case 4, the p/q ratio becomes 1, so they can be ignored
    #for (γ, dict) in p.smap
    for γ in union(keys(p.cmap), keys(q.cmap))
        isleafclade(γ) && continue
        if inccd(p, γ)  # observed in (p and q) or p
            # this is for all contributing splits of an observed γ
            d = 0.
            splits = collect(keys(p.smap[γ]))
            if inccd(q, γ) 
                union!(splits, collect(keys(q.smap[γ])))
            end
            for δ in splits
                pγδ = _splitp(p, γ, δ)
                qγδ = _splitp(q, γ, δ)
                # note this is how KL is defined when pγδ = -Inf (pr 0
                # under p). Motivation is that lim(x->0) xlog(x) = 0
                # (not -Inf)
                !isfinite(pγδ) && continue
                d += exp(pγδ)*(pγδ - qγδ)
            end
            D += d*p.cmap[γ]/p.cmap[p.root]
        elseif inccd(q, γ) && p.α > 0.  # clade not observed in p but in q
            d = 0.
            k = cladesize(γ)
            pγδ = -log(_ns(k))  # conditional probability in p
            for (δ, _) in q.smap[γ]
                qγδ = _splitp(q, γ, δ)
                d += exp(pγδ)*(pγδ - qγδ)
            end
            D += d*_pclade(nl(p), k)
            # this does not properly take into account the actual BMP
            # but assumes the uniform split BMP...
        end
    end
    return D
end

# symmetrized KL divergence
symmkldiv(p, q) = kldiv(p, q) + kldiv(q, p)

# probability that clade of size n has subclade of size i
function _psplitsize(n, i) 
    p = binomial(n, i) / _ns(n)  
    n == 2i ? 0.5 * p : p
end

# probability of clade of k in tree of n leaves under uniform splits
function _pcladesize(n, k)
    n == k && return 1.
    n <  k && return 0.
    (k == 2 || k == 1) && return 1.
    p = 0. 
    for i=1:(n÷2)
        a = _pcladesize(i, k)
        b = _pcladesize(n-i,k) 
        p += (a + b - a*b)*_psplitsize(n, i)
    end
    return p
end

# probability of specific clade of size k under the uniform split BMP
_pclade(n, k) = _pcladesize(n, k) / binomial(n, k)


