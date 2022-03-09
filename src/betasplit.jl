# we need both the probability at the split size level for random
# sampling (in `randsplit` for unrepresented clades) and the
# probability at the per split level for computing posterior mean MBM
# (in SparseSplits). For most analyses, this need only be computed
# once...

"""
    BetaSplitTree(β, n)

Stores the relevant pmf's for a Beta-splitting Markov branching model on
cladograms.

- `β` ∈ (-2, Inf]: the shape parameter (β → -2.0 gives comb tree with
  probability one, β=-1.5 gives PDA, β=-1 gives AB, β=0 gives Yule,
  β=Inf gives random partition tree)
- `p`: Beta-splitting pmfs for clades of size 3 to `n`
- `q`: Beta-splitting probabilities *per split*, i.e. probability of a
  split of size i of a clade of size k is `p[k-2][i]/binomial(k,i)` when
  2i ≠ k.  
- `n`: root clade *size*

## References
- Aldous, David. "Probability distributions on cladograms." Random
  discrete structures. Springer, New York, NY, 1996. 1-18.
- Jones, Graham R. "Tree models for macroevolution and phylogenetic
  analysis." Systematic biology 60.6 (2011): 735-746.
"""
struct BetaSplitTree{T}
    β::T
    p::Vector{Vector{T}}
    q::Vector{Vector{T}}
    n::Int
end

# for convenience
(m::BetaSplitTree{T})(β::T) where T = BetaSplitTree(β, m.n)
(m::BetaSplitTree)(n::Int) = BetaSplitTree(m.β, n)

# XXX we get overflows starting from n = 67
# Graham Jones has a recursive algorithm which is prrobably
# numerically (more) stable.
function BetaSplitTree(β::Real, n::Int)
    p = betasplitpmf.(β, 3:n)
    q = map(k->p[k-2] ./ nsplits.(k, 1:k÷2), 3:n)
    BetaSplitTree(β, p, q, n)
end

# use recursive formula from Jones?
function betasplitpmf(β, n)
    ps = [betasplitf(β, n, i) for i=1:n÷2]
    ps ./ sum(ps)
end

# unnormalized beta-splitting density (n,i)
function betasplitf(β, n, i) 
    p = (2i == n ? 0.5 : 1.0)
    β == Inf && return p * binomial(n,i)/(2^(n-1)-1)
    p * gamma(β+1+i)*gamma(β+1+n-i)/(gamma(i+1)*gamma(n-i+1))
end

nsplits(n, i) = n == 2i ? binomial(n,i)÷2 : binomial(n,i)

# we define the split size to be the size of the smaller clade, note
# that this does not commute with the order on splits! i.e. a split
# is identified by the subclade with smallest id but this id need not
# be the id of the smallest subclade.
splitsize(γ, δ) = min(cladesize(δ), cladesize(γ-δ))

"""
    logpdf(m::BetaSplitTree, γ, δ)

Get the log-probability of observing clade `γ` with split `δ` under
the Beta-splitting model `m`.
"""
function logpdf(m::BetaSplitTree, γ, δ)
    s = cladesize(γ)
    s <= 2 && return 0.
    i = splitsize(γ, δ)
    return log(m.q[s-2][i])
end

"""
    randsplit(m::BetaSplitTree, γ)

Generate a random split of clade `γ` for the Beta splitting model `m`
"""
function randsplit(m::BetaSplitTree, γ)
    s = cladesize(γ)
    s <= 3 && return randsplitofsize(γ, 1)
    p = m.p[s-2]
    k = sample(1:length(p), Weights(p))
    randsplitofsize(γ, k)
end

# a random split clade γ of size k 
"""
    randsplitofsize(γ, k)

Pick a split uniformly from the set of splits of size `k` in clade `γ`.
"""
function randsplitofsize(γ::T, k) where T
    g = digits(γ, base=2)  # binary expansion
    n = sum(g)             # number of leaves
    o = reverse!(sample(1:n, k, replace=false, ordered=true))
    # `o` records which leaves end up in left/right subclade
    # o = [a,b] means we obtain a split by taking/removing the ath and
    # bth leaf from γ. Note that this does not mean the ath and bth
    # index in the binary expansion, but rather the ath and bth one in
    # the binary expansion. The below ugliish while loop constructs
    # the requirred binary expansion of the split from `o`.
    j = pop!(o)
    i = 1  # index over g
    k = 1  # counts how many ones we've passed
    while true
        if g[i] == 1 && k == j
            g[i] = 0
            length(o) == 0 && break
            j = pop!(o)
            k += 1
        elseif g[i] == 1
            k += 1
        end
        i += 1
    end
    splt = T(evalpoly(2, g))
    left = min(splt, γ - splt)
    return left
end
