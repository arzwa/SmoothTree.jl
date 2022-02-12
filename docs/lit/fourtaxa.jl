using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree
using StatsBase, Distributions, Plots
using Serialization
using LaTeXStrings, StatsPlots, Measures
default(gridstyle=:dot, legend=false, framestyle=:box,
        title_loc=:left, titlefont=9)

# Simulation and numerical posterior for the four-taxon case
# this calls 'hybrid-coal' by Zha & Degnan
function hybcoal(S)
    fname, io = mktemp()
    close(io)
    cmd = `/home/arzwa/bin/hybrid-coal -sp $(nwstr(S)) -o $fname`
    run(pipeline(cmd, stderr=devnull))
    xs = map(split, readlines(fname * ".prob"))
    run(`rm $fname $fname.prob`)
    Dict(readnw(string(x[1]))=>parse(Float64, x[2]) for x in xs)
end

# Note that the likelihood p(X|θ,S) is a 15-cell multinomial
# likelihood. When we are dealing with a single allele per species, we
# have two parameters which determine the distribution of gene trees.
θ = log.([0.5, 1.])

# There are two possile tree shapes: an asymmetric tree
S = readnw("(((A,B),C),D);")
# and a symmetric one
#S = readnw("(((A:Inf,B:Inf):$(θ[1]),(C:Inf,D:Inf):$(θ[2]));")

SmoothTree.setdistance!(S, Inf)
SmoothTree.setdistance_internal!(S, exp.(θ))

# Simulate a data set
m = taxonmap(S)
M = SmoothTree.MSC(S, m)
N = 200
Y = randtree(M, m, N)
X = countmap(Y)

# check correctness BTW
truedist = hybcoal(S)
sim = proportionmap(randtree(M, m, 10000))

# all possible species trees
allS = collect(keys(truedist))

function marginal_lhood(X, S, prior, rtol=1e-2)
    function f(θ)
        SmoothTree.setdistance!(S, Inf)
        SmoothTree.setdistance_internal!(S, exp.(θ))
        d = hybcoal(S)
        l = 0.
        for (k,v) in X
            l += v * log(d[k])
        end
        return exp(l + loglikelihood(prior, θ))
    end
    function ff(θ1, q1, q2)
        i, e = quadgk(θ2->f([θ1, θ2]), q1, q2, rtol=rtol)
        return i
    end
    q1, q2 = quantile(prior, [0.001, 0.999])
    i, e = quadgk(θ1->ff(θ1, q1, q2), q1, q2, rtol=rtol)
    return log(i)
end

# this computes the marginal likelihood for each species tree
#allml = map(T->marginal_lhood(X, T, Normal(), 1e-2), allS)

# unnormalized posterior for the relevant θ associated with S,
# conditional on S
function posteriorθ(X, S, prior; steps=100, q=0.1)
    q1, q2 = quantile(prior, [q, 1-q])
    step = (q2-q1)/steps
    function f(θ, S)
        SmoothTree.setdistance!(S, Inf)
        SmoothTree.setdistance_internal!(S, exp.(θ))
        d = hybcoal(S)
        l = 0.
        for (k,v) in X
            l += v * log(d[k])
        end
        return l + loglikelihood(prior, θ)
    end
    [(θ1, θ2, f([θ1, θ2], deepcopy(S))) for θ1=q1:step:q2, θ2=q1:step:q2]
end

function lognormalize(ls)
    ps = exp.(ls .- maximum(ls))
    return ps ./ sum(ps)
end

prior = Normal(log(1.), √5.)
#prior = Normal()
ppd = posteriorθ(X, S, prior, steps=200, q=0.1)

x = getindex.(ppd, 1)[:,1]
y = getindex.(ppd, 2)[1,:]
z = getindex.(ppd, 3)

contour(x, y, z, color=:binary, grid=false, size=(400,400),
        xlabel=L"\theta_1", ylabel=L"\theta_2", 
        levels=-2000:30:0)
hline!([θ[1]], color=:black, ls=:dot)
vline!([θ[2]], color=:black, ls=:dot)


# EP inference
T = UInt16
root = T(15)
bsd  = BetaSplitTree(-1.5, cladesize(root))
Sprior = NatMBM(root, bsd)
θprior = BranchModel(root, SmoothTree.gaussian_mom2nat([mean(prior), std(prior)^2]))
data  = MomMBM.(CCD.(Y, Ref(m)), Ref(bsd), 1e-9)
model = MSCModel(Sprior, θprior, m)

alg   = EPABC(data, model, λ=0.1, α=1e-9, target=500, minacc=100, prunetol=0.)
trace = ep!(alg, 10);
smple = ranking(randtree(MomMBM(trace[end][1].S), 10000))
post  = alg.model

# alternative
alg = SmoothTree.EPABCIS(data, model, λ=0.1, target=4000, miness=10,
                         maxsim=5000, prunetol=1e-9)
istrace = ep!(alg, 10);
trace = first.(istrace)
smple = ranking(randtree(MomMBM(trace[end].S), 10000))
post  = alg.model

# posterior approximation for the relevant branch lengths
lm1, V1 = SmoothTree.gaussian_nat2mom(post.q[c1])
lm2, V2 = SmoothTree.gaussian_nat2mom(post.q[c2])
d1 = Normal(lm1, √V1)
d2 = Normal(lm2, √V2)

# try to get a contour plot for the approximation density
function getcontour(d1, d2, prior; q=0.1, steps=50)
    q1, q2 = quantile(prior, [q, 1-q])
    step = (q2-q1)/steps
    [(θ1, θ2, logpdf(d1, θ1) + logpdf(d2, θ2)) for θ1=q1:step:q2, θ2=q1:step:q2]
end

ZZ = getcontour(d1, d2, prior, q=0.1, steps=200)
xx = getindex.(ZZ, 1)[:,1]
yy = getindex.(ZZ, 2)[1,:]
zz = getindex.(ZZ, 3)
zn = log.(lognormalize(z))
zzn = log.(lognormalize(zz))

levels=-500:20:0
pl1 = contourf(xx, yy, zzn, levels=levels, color=:magma,
               alpha=0.6, linewidth=0, xlim=(-2,2), ylim=(-2,2),
               ls=:dot, title="(A)")
contour!(x, y, zn, grid=false, size=(350,350),
        xlabel=L"\phi_1", ylabel=L"\phi_2",
        levels=levels, color=:black, lw=1)
hline!([θ[1]], color=:black)
vline!([θ[2]], color=:black)
hline!([lm1], color=:black, ls=:dot)
vline!([lm2], color=:black, ls=:dot)

# show marginals
ps = lognormalize(z)
p1 = vec(sum(ps, dims=1)) ./ (x[2] - x[1])
p2 = vec(sum(ps, dims=2)) ./ (x[2] - x[1])

pl2 = plot(prior, fill=true, color=:lightgray, legend=true, label=L"p(\phi)",
           alpha=0.5, xlim=(-2,2), xlabel=L"\phi", ylabel="density", yguidefont=8,
           size=(350,270), fg_legend=:transparent, title="(B)")
plot!(x, p1, fill=true, color=:gray, fillalpha=0.5, linealpha=0., label=L"p(\phi|X)")
plot!(y, p2, fill=true, color=:gray, fillalpha=0.5, linealpha=0., label="")
vline!(θ, color=:black, ls=:dot, label="")
plot!(d1, color=:black, label=L"q(\phi)")
plot!(d2, color=:black, label="")

pps = SmoothTree.postpredsim(post, m, N, 1000)

obs = Dict(hash(k)=>v for (k,v) in proportionmap(Y))
comp = [(haskey(obs, k) ? obs[k] : 0, v) for (k,v) in pps]

pl3 = plot(xticks=1:2:15, ylabel=L"P", xlabel=L"G", title="(C)", xtickfont=7)
for (i,x) in enumerate(sort(comp, rev=true))
    plot!(pl3, [i, i], quantile(x[2], [0.025, 0.975]), color=:lightgray, lw=3)
    scatter!(pl3, [(i, x[1])], color=:black)
end
w = 800; h = 0.3w; w1=0.28; w2=(1-w1)/2
plot(pl1, pl2, pl3, size=(w,h), bottom_margin=5mm, left_margin=3mm,
     dpi=300, layout=grid(1,3, widths=[w1,w2,w2]))

final_splits = SmoothTree.allsplits(post.S)
splittrace = mapreduce(x->[MomMBM(x.S)[γ,δ] for (γ, δ) in final_splits], hcat, trace) |> permutedims

all_splits = [final_splits; [(γ,γ-δ) for (γ,δ) in final_splits]]
qs = map(x->SmoothTree.MomBranchModel(x.q), trace)
mutrace = mapreduce(x->[x[(γ,δ)][1] for (γ, δ) in all_splits], hcat, qs) |> permutedims
vtrace  = mapreduce(x->[x[(γ,δ)][2] for (γ, δ) in all_splits], hcat, qs) |> permutedims

kwargs = (xscale=:log10, xticks=[1, 10, 100, 1000])
ps4 = [plot(splittrace, title="(D)", ylabel=L"\theta"; kwargs...), 
     plot(mutrace, ylabel=L"\mu", xlabel="iteration", xguidefont=8; kwargs...), 
     plot(vtrace, ylabel=L"\sigma^2"; kwargs...)]

pl4 = plot(ps4..., layout=(1,3), xticks=[1, 10, 100, 1000])

plot(pl1, pl2, pl3, ps4..., layout=grid(2,3,widths=[1/3,1/3,1/3]), dpi=300)

#savefig("docs/img/fourtaxon.pdf")
#savefig("docs/img/fourtaxon.png")


