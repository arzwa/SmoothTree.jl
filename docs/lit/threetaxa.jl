# Three-taxon case
# Species tree S = ((A,B),C)
# Gene tree probabilities
# AB|C: p1 = 1 - (2/3) * exp(-θ)
# AC|B: p2 = (1/3) * exp(-θ)
# BC|A: p3 = (1/3) * exp(-θ)
# Gene trees = X | S, θ ~ Multinomial(p1, p2, p3)
# Consider a prior on θ, p(θ), then:
# p(S|X) ∝ p(X|S)p(S) = p(S)∫p(X|θ,S)p(θ)dθ
# Compute this for all three possible S
# Compare against EP-ABC approach

function triple_p(tree, m)
    iout = isleaf(tree[1]) ? 1 : 2
    out = tree[iout]
    ing = tree[iout % 2 + 1]
    a = m[name(ing[1])]
    b = m[name(ing[2])]
    c = m[name(out)]
    o = sortperm([a+b, a+c, b+c])
    θ = distance(ing)
    p1 = (1/3)*exp(-θ)
    p = [1 - 2p1, p1, p1]
    return p[o]
end

function triples(X, m)
    xs = map(collect(X)) do (tree,v)
        i = isleaf(tree[1]) ? 2 : 1
        a = name(tree[i][1])
        b = name(tree[i][2])
        γ = m[a] + m[b]
        (γ, v)
    end
    last.(xs)[sortperm(first.(xs))]
end

function loglhood(tree, m, G)
    p = triple_p(tree, m)
    logpdf(Multinomial(sum(G), p), G)
end

function setθ!(tree, θ)
    i = isleaf(tree[1]) ? 2 : 1
    tree[i].data.distance = θ
    return tree
end

function marginal_lhood(G, S, m, prior)
    f(θ) = exp(loglhood(setθ!(deepcopy(S), exp(θ)), m, G) + logpdf(prior, θ))
    q1, q2 = quantile(prior, [0.001, 0.999])
    integral, err = quadgk(θ -> f(θ), q1, q2, rtol=1e-8)
    return integral
end

# Simulate data
#S = SmoothTree.MSC(nw"((A:Inf,B:Inf):0.2,C:Inf);")
θ = 0.6
S = SmoothTree.MSC(readnw("((B:Inf,C:Inf):$θ,A:Inf);"))
m = SmoothTree.taxonmap(S.tree)
Y = randtree(S, 100)
X = countmap(Y)
G = triples(X, m)

# get posterior probs by integration
using QuadGK
trees = collect(keys(X))
prior = Normal()
ls = map(tree->marginal_lhood(G, tree, m, prior), trees)
pp = ls ./ sum(ls)
pp = Dict(t=>p for (t,p) in zip(trees, pp))

Sprior = NatBMP(CCD(trees, lmap=m, α=1.))
smple  = ranking(randtree(MomBMP(Sprior), 10000))
θprior = [SmoothTree.gaussian_mom2nat(log(1.), 5.)...]
data  = CCD.(Y, lmap=m, α=0.)
model = MSCModel(Sprior, θprior, m)
alg   = EPABC(data, model, λ=0.05, α=1e-9)

trace = ep!(alg, 10, maxn=1e5, mina=200, target=1000);
smple = SmoothTree.ranking(randtree(SmoothTree.MomBMP(trace[end].S), 10000))

A, B = traceback(trace)
p1 = plot(A[0x0007])
p2 = plot(B[0x0003])
hline!(p2, [log(θ)], ls=:dot, color=:black)

S = SmoothTree.randsptree(trace[end])
M = SmoothTree.MSC(S)
pps = proportionmap(randtree(M, 100000))
obs = proportionmap(Y)

xs = map(x->(x[2], haskey(pps, x[1]) ? pps[x[1]] : 0.), collect(obs))
scatter(xs, color=:lightgray, size=(400,400)); plot!(x->x, color=:black, ls=:dot)


