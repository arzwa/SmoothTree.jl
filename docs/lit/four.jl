using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, StatsBase, Distributions
using Plots, LaTeXStrings, StatsPlots, Measures
default(gridstyle=:dot, legend=false, framestyle=:box,
        title_loc=:left, titlefont=10, guidefont=10)

# Simulation and numerical posterior for the four-taxon case this calls
# 'hybrid-coal' by Zha & Degnan (https://github.com/hybridLambda/hybrid-coal)
function hybcoal(S)
    fname, io = mktemp()
    close(io)
    cmd = `/home/arzwa/bin/hybrid-coal -sp $(nwstr(S)) -o $fname`
    run(pipeline(cmd, stderr=devnull))
    xs = map(split, readlines(fname * ".prob"))
    run(`rm $fname $fname.prob`)
    Dict(readnw(string(x[1]))=>parse(Float64, x[2]) for x in xs)
end

# Note that the likelihood p(X|θ,S) is a 15-cell multinomial likelihood. When
# we are dealing with a single allele per species, we have two parameters which
# determine the distribution of gene trees.
θ = log.([0.5, 1.])

# There are two possile tree shapes: an asymmetric tree
S = nw"(((A,B),C),D);"
# and a symmetric one
#S = nw"(((A,B),(C,D)));"

SmoothTree.setdistance!(S, Inf)
SmoothTree.setdistance_internal!(S, exp.(θ))

# Simulate a data set
m = clademap(S, UInt16)
M = SmoothTree.MSC(S, m)
N = 200
Y = randtree(M, m, N)
X = countmap(Y)

# prior for branch lengths
prior = Normal(log(1.), √5.)

# check correctness BTW
truedist = hybcoal(S)
sim = proportionmap(randtree(M, m, 10000))

# all possible species trees
allS = collect(keys(truedist))

# marginalize over branch lengths, conditioning on species tree
# ∫p(X,θ|S)dθ
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

# unnormalized posterior for the relevant θ associated with S, conditional on S
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

# takes a while
ppd = posteriorθ(X, S, prior, steps=100, q=0.1)

# get contours
x = getindex.(ppd, 1)[:,1]
y = getindex.(ppd, 2)[1,:]
z = getindex.(ppd, 3)
contour(x, y, z, color=:binary, grid=false, size=(400,400),
        xlabel=L"\theta_1", ylabel=L"\theta_2", 
        levels=-2000:30:0)
hline!([θ[1]], color=:black, ls=:dot)
vline!([θ[2]], color=:black, ls=:dot)


# EP inference
root   = 0x000f
bsd    = BetaSplitTree(-1.5, cladesize(root))
Sprior = CCD(SplitCounts(root), bsd)
tips   = collect(keys(m))
θprior = BranchModel(root, SmoothTree.gaussian_mom2nat([mean(prior), std(prior)^2]), 
                     inftips=tips)
data   = SmoothTree.Locus.(Y, Ref(m), prior=bsd, α=1e-9)
model  = MSCModel(Sprior, θprior)
#alg    = SmoothTree.EPABCIS(data, model, 10000, target=1000, miness=10.)
alg    = SmoothTree.EPABCSIS(data, model, 10000, 5, target=1000, miness=10., prunetol=0.)
trace  = ep!(alg, 10);
post   = alg.model

randtree(alg.model.S, m, 10000) |> ranking

# posterior approximation for the relevant branch lengths
c1 = (0x0007, 0x0003)
c2 = (0x000f, 0x0007)
lm1, V1 = SmoothTree.gaussian_nat2mom(post.ϕ[c1])
lm2, V2 = SmoothTree.gaussian_nat2mom(post.ϕ[c2])
d1 = Normal(lm1, √V1)
d2 = Normal(lm2, √V2)


# 1. contour plot
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


# 2. Marginal density plot
ps = lognormalize(z)
p1 = vec(sum(ps, dims=1)) ./ (x[2] - x[1])
p2 = vec(sum(ps, dims=2)) ./ (x[2] - x[1])

pl2 = plot(prior, fill=true, color=:lightgray, legend=true, label=L"p(\phi)",
           alpha=0.5, xlim=(-2,2), xlabel=L"\phi", ylabel="density", yguidefont=8,
           size=(350,270), fg_legend=:transparent, title="(B)")
plot!(x, p1, fill=true, color=:gray, fillalpha=0.5, linealpha=0., label=L"p(\phi|y)")
plot!(y, p2, fill=true, color=:gray, fillalpha=0.5, linealpha=0., label="")
vline!(θ, color=:black, ls=:dot, label="")
plot!(d1, color=:black, label=L"q(\phi)")
plot!(d2, color=:black, label="")

pl3 = plot(getfield.(trace, :ev), ylabel=L"\hat{Z}_n", xlabel=L"n", title="(C)")

plot(pl1, pl2, pl3, layout=(1,3), size=(700,900/3√2), bottom_margin=5mm, left_margin=3mm)
#savefig("docs/img/fourtaxon-sim.pdf")



# 4. trace plots
xs = SmoothTree.traceback(first.(trace))
p4 = plot(plot(xs.θ, title=L"\theta"), 
          plot(xs.μ, title=L"\mu"), 
          plot(xs.V, title=L"\sigma^2"), 
          plot(getindex.(trace,2), title=L"Z"), 
          xlabel="iteration", xscale=:log10, xticks=[1,10,100,1000])

kwargs = (xscale=:log10, xticks=[1,10,100,1000], xlabel="iteration")
ps4 = [plot(xs.θ, title=L"\theta"; kwargs...), 
       plot(xs.μ, title=L"\mu"; kwargs...), 
       plot(xs.V, title=L"\sigma^2"; kwargs...)]

# 5. combined plot
plot(pl1, pl2, pl3, ps4..., layout=grid(2,3,widths=[1/3,1/3,1/3]), dpi=300, size=(700,500), bottom_margin=4mm)

savefig("docs/img/fourtaxon.pdf")
savefig("docs/img/fourtaxon.png")


