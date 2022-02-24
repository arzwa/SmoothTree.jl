# Here I compare the EP approximation with numerical integration for a
# three-taxon simulated data set. 
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree, QuadGK
using StatsBase, Distributions, Plots
using LaTeXStrings, StatsPlots, Measures
default(gridstyle=:dot, legend=false, framestyle=:box,
        title_loc=:left, titlefont=9)

# a bit involved...
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
    sum(log.(p .^ G))
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
θ = 0.5
S = readnw("((B:Inf,C:Inf):$θ,A:Inf);")
m = SmoothTree.clademap(S)
M = SmoothTree.MSC(S, m)
N = 100
Y = randtree(M, m, N)
X = countmap(Y)
G = triples(X, m)

# numerical analysis
prior = Normal(log(1.), √5.)
trees = collect(keys(X))
ls = map(tree->marginal_lhood(G, tree, m, prior), trees)
pp = ls ./ sum(ls)
pp = Dict(t=>p for (t,p) in zip(trees, pp))
ppdf = postpdf(G, trees, m, prior)
ev = log(sum((1/3) .* ls))

# EP-ABC
root   = UInt16(7)
bsd    = BetaSplitTree(-1.5, 3)  # uniform
Sprior = NatMBM(root, bsd)
θprior = SmoothTree.gaussian_mom2nat([log(1.), 5.])
q      = BranchModel(root, θprior )
data   = SmoothTree.Locus.(Y, Ref(m))
model  = MSCModel(Sprior, q)
alg    = SmoothTree.EPABCIS(data, model, 10000, target=10000, miness=10.)
trace  = ep!(alg, 10);

relabel.(randtree(alg.model.S, 10000), Ref(m)) |> ranking

# some plots
bs = SmoothTree.getbranchapprox(alg.model, randsplits(alg.model.S))
d  = bs[1][3]
p2 = plot(prior, color=:lightgray, fill=true, size=(300,200),
          xlabel=L"\theta", label=L"p(\theta)", title="(B)",
          xlim=(-2.5,2.5), alpha=0.5, legend=true)
plot!(ppdf, color=:gray, fillalpha=0.5, fill=true, linealpha=0.,
      label=L"p(\theta|X)")
plot!(d, color=:black, label=L"Q(\theta)")
vline!([log(θ)], color=:black, ls=:dot, label=false)

# plot the traceback of the algorithm
xs = SmoothTree.traceback(first.(trace))
plot(plot(xs.θ, title=L"\theta"), 
     plot(xs.μ, title=L"\mu"), 
     plot(xs.V, title=L"\sigma^2"), 
     plot(getindex.(trace,2), title=L"Z"), xlabel="iteration")



# an opportunity to investigate the effect of some algorithm settings
αs = [1e-16, 1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.2]
Zs = map(αs) do α
    alg    = SmoothTree.EPABCIS(data, model, 10000, target=9000, miness=10., α=α)
    trace  = ep!(alg, 5);
    getindex.(trace, 2)
end

plot(hcat(Zs...), label=reshape(αs, 1, 7), legend=:bottomleft, ylabel=L"Z", xlabel="iteration")
hline!([ev], ls=:dot, color=:black)
#savefig("docs/img/three-taxa-alpha-Z.pdf")

λs = [0.01, 0.05, 0.1, 0.2, 0.5]
Zs = map(λs) do λ
    alg    = SmoothTree.EPABCIS(data, model, 10000, target=100, miness=10., λ=λ, α=1e-5)
    trace  = ep!(alg, 5);
    getindex.(trace, 2)
end

plot(hcat(Zs...), label=reshape(λs, 1, 5), legend=:bottomleft)
