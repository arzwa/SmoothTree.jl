using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree
using StatsBase, Distributions, Plots, StatsPlots, LaTeXStrings, Measures
using Serialization
theme(:wong2)
default(gridstyle=:dot, legend=false, framestyle=:box, title_loc=:left, titlefont=7)

#trees = map(readdir("docs/data/yeast-rokas/mb", join=true)) do f
#    @info f
#    ts = readnw.(readlines(f))
#    ts = SmoothTree.rootall!(ts, "Calb")
#    countmap(ts)
#end
#serialize("docs/data/yeast-rokas/ccd-mb.jls", trees)

mbdata  = deserialize("docs/data/yeast-rokas/mb.jls")
ufbdata = deserialize("docs/data/yeast-rokas/ufboot.jls")
mltrees = readnw.(readlines("docs/data/yeast-rokas/yeast.mltrees"))
tmap  = clademap(mltrees[1])

# we can compare using ML gene trees against acknowledging uncertainty
data1 = Locus.(mbdata, Ref(tmap), 1e-6, -1.5)
#data1 = Locus.(ufbdata, Ref(tmap), 1e-6, -1.5)
data2 = Locus.(mltrees, Ref(tmap), 1e-6, -1.5)

μ, V = log(2.), 5.
root   = UInt16(sum(keys(tmap)))
Sprior = NatMBM(CCD(unique(trees), tmap), BetaSplitTree(-1.5, cladesize(root)), 1.)
θprior = BranchModel(root, gaussian_mom2nat([μ, V]), inftips=collect(keys(tmap)))
model  = MSCModel(Sprior, θprior)

alg1   = EPABCIS(data1, model, 100000, target=2000, miness=10., λ=0.1, α=1e-3)
trace1 = ep!(alg1, 5)

alg2   = EPABCIS(data2, model, 100000, target=2000, miness=10., λ=0.1, α=1e-3)
trace2 = ep!(alg2, 5)

# posterior inspection
smple1 = relabel.(randtree(alg1.model.S, 10000), Ref(tmap)) |> ranking
smple2 = relabel.(randtree(alg2.model.S, 10000), Ref(tmap)) |> ranking

# now should estimate Z for both data sets with map topology of the other
tree   = SmoothTree.getbranches(smple1[1][1], tmap)
alg3   = EPABCIS(data2, tree, θprior, 100000, target=2000, miness=10., λ=0.1, α=1e-3)
trace3 = ep!(alg3, 5)

# and the other way around
tree   = SmoothTree.getbranches(smple2[1][1], tmap)
alg4   = EPABCIS(data0, tree, θprior, 100000, target=2000, miness=10., λ=0.1, α=1e-3)
trace4 = ep!(alg4, 5)

# posterior analysis
algs   = [alg1, alg2, alg3, alg4]
models = [alg1.model, alg2.model, 
          MSCModel(alg1.model.S, alg3.model), 
          MSCModel(alg2.model.S, alg4.model)]
traces = [trace1, trace2, trace3, trace4]
labels = ["CCD", "ML trees", "ML trees (S₁)", "CCD (S₂)"] 

# tree height
function totallength(m, N=10000)
    hs = map(1:N) do _
        sum(last.(filter(x->isfinite(x[end]), randbranches(m))))
    end
    mean(hs), quantile(hs, [0.025, 0.975])
end

map(totallength, models)

species = Dict("Scas"=>"S. castellii", "Smik"=>"S. mikatae", 
               "Spar"=>"S. paradoxus", "Skud"=>"S. kudriavzevii", 
               "Sbay"=>"S. bayanus", "Sklu"=>"S. kluyveri", 
               "Scer"=>"S. cerivisiae", "Calb"=>"C. albicans")
spmap = Dict(v=>species[k] for (k,v) in tmap.m2)

ps = [plot(relabel(maptree(models[i]), spmap), 
           transform=true, pad=1., fs=8, scalebar=10) for i=[1,2,4,3]]
p0 = plot(ps..., layout=(1,4), size=(600,200))

p1 = plot(legend=:right, xlim=(106,Inf), fg_legend=:transparent,
          ylabel=L"\hat{Z}", xlabel="iteration")
for i=1:4
    plot!(p1, getindex.(traces[i], 2), label=labels[i], lw=1.5)
end
p2 = plot(legend=:right, fg_legend=:transparent, ylabel=L"C_i", xlabel="site")
for i=1:4
    plot!(p2, sort(algs[i].siteC), lw=1.5, label=labels[i])
end
plot(p1, p2)

plot(p0, p1, p2, size=(600, 600/√2))

ps = map([1,3]) do i
    alg = algs[i]
    bs = SmoothTree.getbranchapprox(alg.model, randsplits(alg.model.S))
    bs = sort(filter(x->!SmoothTree.isleafclade(x[2]), bs))
    p1 = map(bs) do (γ, δ, d)
        plot(LogNormal(μ, √V), fill=true, color=:lightgray, fillalpha=0.3)
        plot!(LogNormal(d.μ, d.σ), fill=true, fillalpha=0.3, color=:black, xlim=(0,100))
    end |> x->plot(x..., size=(650,350))
    title!(p1[1], labels[i])
end
pl2 = plot(ps..., layout=(2,1), size=(600,600/√2))

map(bs) do (γ, δ, d)
    plot(Normal(μ, √V), fill=true, color=:lightgray, fillalpha=0.3)
    plot!(Normal(d.μ, d.σ), fill=true, fillalpha=0.3, color=:black, xlim=(-4,4))
end |> x->plot(x..., size=(650,350))

alg = algs[1]
pps = SmoothTree.postpredsim(alg.model, data1, 1000)
obs = proportionmap(trees)
comp = [(haskey(obs, k) ? obs[k] : 0, v) for (k,v) in pps]
comp = filter(x->x[1] > 0 || mean(x[2]) > 0.001, comp)
pl3 = plot(ylabel=L"P", xlabel=L"G", xtickfont=7, xticks=0:2:20)
for (i,x) in enumerate(sort(comp, rev=true))
    plot!(pl3, [i, i], quantile(x[2], [0.025, 0.975]), color=:lightgray, lw=4)
    scatter!(pl3, [(i, x[1])], color=:black)
end
plot(pl3, size=(300,200), ylim=(0,1))

#plot(pl1, pl3, layout=grid(1,2, widths=[0.7,0.3]), size=(700,220), bottom_margin=3mm)
lay = @layout [grid(1,4, widths=[0.2,0.8/3,0.8/3,0.8/3]) ; grid(1,3)]
p = plot(ps..., p1, p2, pl3, size=(700,400), layout=lay,
         bottom_margin=3mm, guidefont=8)
for (i,c) in enumerate('A':'G')
    title!(p[i], "($c)") 
end
plot(p)

savefig("docs/img/yeast-panel.pdf")

# the posterior predictive distribution for gene trees differs strongly.
# here we see that when we acknowledge gene tree uncertainty, we
# estimate much less true discordance... posterior predictive
# simulation suggest that almost 90% of the *true* gene trees matches
# the species tree.

# get proportionmap for full posterior over gene trees
X = mbdata[1]
for i=2:length(mbdata)
    for (k,v) in mbdata[i]
        !haskey(X, k) && (X[k] = 0)
        X[k] += v
    end
end
Z = sum(values(X))
Y = Dict(k=>v/Z for (k,v) in X)

alg = algs[1]
pps = SmoothTree.postpredsim(alg.model, data1, 1000)
obs = proportionmap(trees)
comp = [(haskey(obs, k) ? obs[k] : 0, v, (haskey(Y, k) ? Y[k] : 0)) for (k,v) in pps]
comp = filter(x->x[1] > 0 || mean(x[2]) > 0.001, comp)
pl3 = plot(ylabel=L"P", xlabel=L"G", xtickfont=7, xticks=0:2:20)
for (i,x) in enumerate(sort(comp, rev=true))
    plot!(pl3, [i, i], quantile(x[2], [0.025, 0.975]), color=:lightgray, lw=4)
    scatter!(pl3, [(i, x[1])], color=:black)
    scatter!(pl3, [(i, x[3])], color=:lightgray)
end
plot(pl3, size=(300,200), ylim=(0,1))
