using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree
using StatsBase, Distributions, Plots
using Serialization
using LaTeXStrings, StatsPlots, Measures
default(gridstyle=:dot, legend=false, framestyle=:box,
        title_loc=:left, titlefont=7)

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

# p(θ|X) = ∑p(X|θ,S)p(θ)p(S)/ ∑∫p(X|θ,S)p(θ)dθp(S)
function postpdf(G, trees, m, prior; steps=1000) 
    ls = map(tree->marginal_lhood(G, tree, m, prior), trees)
    q1, q2 = quantile(prior, [0.001, 0.999])
    step = (q2-q1)/steps
    f(θ,S) = loglhood(setθ!(deepcopy(S), exp(θ)), m, G) + logpdf(prior, θ)
    map(q1:step:q2) do θ
        cond = map(S->f(θ,S), trees)
        (θ, sum(exp.(cond))/sum(ls))
    end
end

# Simulate data
#S = SmoothTree.MSC(nw"((A:Inf,B:Inf):0.2,C:Inf);")
θ = 0.5
S = readnw("((B:Inf,C:Inf):$θ,A:Inf);")
m = taxonmap(S)
M = SmoothTree.MSC(S, m)
N = 100
Y = randtree(M, m, N)
X = countmap(Y)
G = triples(X, m)
prior = Normal(log(1.), √5.)

# get posterior probs by integration
using QuadGK
trees = collect(keys(X))
ls = map(tree->marginal_lhood(G, tree, m, prior), trees)
pp = ls ./ sum(ls)
pp = Dict(t=>p for (t,p) in zip(trees, pp))
ppdf = postpdf(G, trees, m, prior)

# EP
T = UInt16
bsd    = BetaSplitTree(-1.5, 3)
Sprior = NatMBM(T(7), bsd)
θprior = SmoothTree.gaussian_mom2nat([log(1.), 5.])
q      = BranchModel(Tuple{T,T}, θprior )
data   = CCD.(Y, Ref(m))
model  = MSCModel(Sprior, q, m)
alg    = EPABC(data, model, λ=0.1, α=1e-9, maxsim=1e5, target=500, minacc=100)

trace  = ep!(alg, 10);
smple  = ranking(randtree(MomMBM(alg.model.S), 10000))
relabel(first(smple)[1], m)

combine(xs::Vector{<:Dict}) = Dict(k=>[x[k] for x in xs] for k in keys(xs[1]))

A, B = traceback(trace)
S = SmoothTree.randtree(trace[end])
M = SmoothTree.MSC(S, Dict(id(n)=>[id(n)] for n in getleaves(S)))
pps = combine(map(_->proportionmap(randtree(M, m, N)), 1:1000))
obs = proportionmap(Y)

c = (0x0007, 0x0003)
p1 = plot(B[c], label=[L"\log \mu" L"\sigma^2"],
          xlabel=L"n", color=:black, ls=[:solid :dash], title="(A)")
hline!(p1, [log(θ)], ls=:dot, color=:black, label="")
μ, V = B[c][end,:]
p2 = plot(prior, color=:lightgray, fill=true, size=(300,200),
          xlabel=L"\theta", label=L"p(\theta)", title="(B)",
          xlim=(-2.5,2.5), alpha=0.5)
plot!(ppdf, color=:gray, fillalpha=0.5, fill=true, linealpha=0.,
      label=L"p(\theta|X)")
plot!(Normal(μ, √V), color=:black, label=L"Q(\theta)")
vline!([log(θ)], ls=:dot, color=:black, label="")
p3 = plot(xticks=1:3, xlabel=L"G", ylabel=L"P", title="(C)")
for (i,(k,v)) in enumerate(pps)
    violin!([i], [v], flyer=false, color=:lightgray, label="")
    plot!([i-0.5, i+0.5], [obs[k], obs[k]], color=:black, label="")
end
plot(p1, p2, p3, size=(700,200), layout=(1,3), bottom_margin=4mm,
     legend=true, fg_legend=:transparent, dpi=300)

savefig("docs/img/threetaxon.pdf")
savefig("docs/img/threetaxon.png")

# Convergence, posterior approximation and posterior predictive
# distribution plots
