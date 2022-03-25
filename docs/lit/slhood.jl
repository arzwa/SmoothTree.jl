# Synthetic likelihood idea
using Pkg; Pkg.activate(@__DIR__)
using NewickTree, SmoothTree, Plots, StatsPlots, Distributions, Measures, LaTeXStrings
using StatsBase, ThreadTools
theme(:wong2)
default(gridstyle=:dot, legend=false, framestyle=:box, title_loc=:left, titlefont=7)

species = string.('A':'Z')[1:8]
spmap = clademap(species)
root = rootclade(spmap)
ntaxa = length(spmap)
S = randtree(CCD(SplitCounts(root), BetaSplitTree(-1., ntaxa)), spmap)

function hybcoal(S)
    fname, io = mktemp()
    close(io)
    cmd = `/home/arzwa/bin/hybrid-coal -sp $(nwstr(S)) -o $fname`
    run(pipeline(cmd, stderr=devnull))
    xs = map(split, readlines(fname * ".prob"))
    run(`rm $fname $fname.prob`)
    Dict(readnw(string(x[1]))=>parse(Float64, x[2]) for x in xs)
end

# We set some random branch lengths
l = length(postwalk(S))
d = MixtureModel([LogNormal(log(0.5), 0.5), LogNormal(log(3.), .5)], [0.2,0.8])
θ = rand(d, l)
SmoothTree.setdistance!(S, θ)
spmap = clademap(S)
plot(S, transform=true, scalebar=1)


# Compute gene tree distribution using hybcoal
# --------------------------------------------
genetreedist = hybcoal(S)

# Compute gene tree distribution using simulation 
# -----------------------------------------------
init = Dict(k=>[k] for k in keys(spmap))
for n in postwalk(S)
    n.id = SmoothTree.getclade(spmap, name.(getleaves(n)))
end
m = SmoothTree.MSC(S, init)
randtree(m, spmap)

# get the reference probability distribution
reference = randtree(m, spmap, 10^6)
tdist = proportionmap(reference)

# get a sample to compare CCD approx to reference
smple = randtree(m, spmap, 10000)
X = SplitCounts(smple, spmap)

filtered = [(SmoothTree.getsplits(k, spmap),v) for (k,v) in genetreedist if log10(v) > -9]

ps = []
for β=[-1.99, 0., 10.], α=[0.01, 1., 100.]
    ccd = CCD(X, BetaSplitTree(β, length(spmap)), α)
    xs = [(log10(v), logpdf(ccd, k)*log10(ℯ)) for
          (k,v) in filtered]
    p = scatter(xs, ms=2, title="\$\\alpha=$α, \\beta=$β\$")
    plot!(p, x->x, color=:lightgray)
    push!(ps, p)
end
plot(ps..., size=(600,600), xlim=(-9,0.1), ylim=(-12,0.1), xtickfontsize=6,
     ytickfontsize=6, titlefontsize=9)

ccd = CCD(X, BetaSplitTree(-1., length(spmap)), 1.)
xs = [(log10(v), logpdf(ccd, SmoothTree.getsplits(k, spmap))*log10(ℯ)) for (k,v) in genetreedist]
p = scatter(xs, ms=2)

plot(S, transform=true, scalebar=5)

# QQ like plot
o = sortperm(last.(filtered))
x = cumsum(last.(filtered)[o])
ps = []
for β=[-1.99, 0., 10.], α=[0.01, 1., 100.]
    ccd = CCD(X, BetaSplitTree(β, length(spmap)), α)
    p = [exp(logpdf(ccd, k)) for k in first.(filtered)[o]]
    y = cumsum(p)
    p = scatter(x, y, xscale=:log10, yscale=:log10, ms=2, size=(300,300))
    title!(p, "\$\\alpha=$α, \\beta=$β\$", titlefontsize=9)
    plot!(x->x)
    push!(ps, p)
end
plot(ps..., size=(600,600), xticks=[10.0^i for i=-10:2:0], yticks=[10.0^i for i=-10:2:0])


# minimize KL divergence
using Optim

function kldiv(ccd::CCD, genetrees)
    d = 0.
    for (k,v) in genetrees
        l = logpdf(ccd, k) 
        d += v*(log(v) - l)  # this is the relevant one...
        #p = exp(l)
        #d += p*(l - log(v))
    end
    return d
end

f(x) = kldiv(CCD(X, BetaSplitTree(x[1], length(spmap)), x[2]), filtered)

l = [-1.999, 0.]
u = [10., Inf]
x0 = [0., 1.]
results = optimize(f, l, u, x0, Fminbox())
results.minimizer


# fit it
using Turing

reference = randtree(m, spmap, 10^6)
tdist = countmap(reference)

smple = randtree(m, spmap, 10000)

@model bsd(smple, spmap, tdist) = begin
    β ~ Uniform(-1.999, 10.)
    α ~ Exponential()
    ccd = CCD(SplitCounts(smple, spmap), BetaSplitTree(β, length(spmap)), α)
    for (k, v) in tdist
        Turing.@addlogprob! v*logpdf(ccd, SmoothTree.getsplits(k, spmap))
    end
end

chain = sample(bsd(smple, spmap, tdist), NUTS(), 200)



