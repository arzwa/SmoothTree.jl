using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree
using StatsBase, Distributions
using Plots, StatsPlots, LaTeXStrings
using Serialization
using SmoothTree: EPABCIS, ep!
theme(:wong2)
default(gridstyle=:dot, legend=false, framestyle=:box, title_loc=:left, titlefont=7)

mltrees = readnw.(readlines("docs/data/yeast-rokas/yeast.mltrees"))
mbdata = deserialize("docs/data/yeast-rokas/mb.jls")
spmap  = clademap(first(mbdata[1])[1])
root   = rootclade(spmap)
ntaxa  = cladesize(root)

μ, V = log(2.), 5.
Sprior = CCD(SplitCounts(root), BetaSplitTree(-1.5, ntaxa))
ϕprior = BranchModel(root, gaussian_mom2nat([μ, V]))
model  = MSCModel(Sprior, ϕprior)

data1 = Locus.(mltrees, Ref(spmap), prior=BetaSplitTree(-1.5, ntaxa), α=1e-6)
data2 = Locus.(mbdata,  Ref(spmap), prior=BetaSplitTree(-1.5, ntaxa), α=1e-6)


# A fixed topology
# ================
# The likely true tree:
tree1 = nw"(Calb,(Sklu,(Scas,(Sbay,(Skud,(Smik,(Scer,Spar)))))));"
tree2 = nw"(Calb,(Sklu,(Scas,((Sbay,Skud),(Smik,(Scer,Spar))))));"
bs1 = SmoothTree.getbranches(tree1, spmap)
bs2 = SmoothTree.getbranches(tree2, spmap)

mom  = MvNormal(fill(μ, length(bs1)), √V)
prior = SmoothTree.tonatural(mom)

alg3   = EPABCSIS(data1, bs1, prior, 10000, 5, target=1000, miness=10., λ=0.1, α=1e-4, c=0.95)
trace3 = ep!(alg3, 5)

alg4   = EPABCSIS(data1, bs2, prior, 10000, 5, target=1000, miness=10., λ=0.1, α=1e-4, c=0.95)
trace4 = ep!(alg4, 5)

plot(getfield.(trace3, :ev), legend=:topleft)
plot!(getfield.(trace4, :ev), legend=:topleft)


# Using ordinary importance sampling
# ==================================
alg1   = EPABCIS(data1, model, 50000, target=500, miness=10., λ=0.1, α=1e-4, c=0.95)
trace1 = ep!(alg1, 5)

alg2   = EPABCIS(data2, model, 50000, target=500, miness=10., λ=0.1, α=1e-4, c=0.95)
trace2 = ep!(alg2, 5)

smple1 = randtree(alg1.model.S, spmap, 10000) |> ranking
smple2 = randtree(alg2.model.S, spmap, 10000) |> ranking

tree   = SmoothTree.getbranches(smple2[1][1], spmap)
alg3   = EPABCIS(data1, tree, θprior, 10000, target=200, miness=10., λ=0.1, α=1e-4, c=0.95)
trace3 = ep!(alg3, 5)


plot(getfield.(trace1, :ev), legend=:topleft)
plot!(getfield.(trace2, :ev), legend=:topleft)

using SmoothTree: relabel, maptree
t1 = relabel(maptree(alg1.model), spmap)
t2 = relabel(maptree(alg2.model), spmap)
p1 = plot(t1, transform=true, scalebar=10.)
p2 = plot(t2, transform=true, scalebar=10.)
plot(p1, p2)
    
M  = alg.model
bs = SmoothTree.getbranchapprox(M.ϕ, randsplits(M.S))
ps = map(bs) do x
    plot(Normal(μ, √V), fill=true, color=:lightgray)
    plot!(x[end], color=:black)
end |> xs->plot(xs...)


# Unrooted
# ========
# we cannot easily identify the root from a collection of unrooted trees it
# appears.
data3 = Locus.(mbdata,  Ref(spmap), prior=BetaSplitTree(-1.5, ntaxa), α=1e-6, 
               rooted=false)
Sprior = CCD(SplitCounts(root), BetaSplitTree(-1.99, ntaxa))
ϕprior = BranchModel(root, [0., -0.5])
model  = MSCModel(Sprior, ϕprior)
alg3   = EPABCIS(data3, model, 50000, target=200, miness=10., λ=0.1, α=1e-4, c=0.95)
trace3 = ep!(alg3, 2)


# SIS
# ===
using SmoothTree: EPABCSIS, maptree, relabel

alg1b  = EPABCSIS(data1, model, 10000, 5, target=1000, miness=10., λ=0.1, α=1e-4, c=0.95)
trace1 = ep!(alg1b, 5)
smple1 = randtree(alg1b.model.S, spmap, 10000) |> ranking
maptree1 = maptree(alg1b.model, spmap)

alg2b  = EPABCSIS(data2, model, 10000, 5, target=1000, miness=10., λ=0.1, α=1e-4, c=0.95)
trace2 = ep!(alg2b, 5)
smple2 = randtree(alg2b.model.S, spmap, 10000) |> ranking
maptree2 = maptree(alg2b.model, spmap)

plot(getfield.(trace1, :ev), legend=:topleft)
plot!(getfield.(trace2, :ev), legend=:topleft)

plot(sort(alg1b.siteC), legend=:bottomright)
plot!(sort(alg2b.siteC))

t1 = smple2[1][1]
tree   = SmoothTree.getbranches(t1, spmap)
alg3   = EPABCSIS(data1, tree, ϕprior, 10000, 5, target=500, miness=10., λ=0.1, α=1e-4, c=0.95)
trace3 = ep!(alg3, 5)
maptree3 = SmoothTree.addmapbranches!(t1, spmap, alg3.model)

t2 = smple1[1][1]
tree   = SmoothTree.getbranches(t2, spmap)
alg4   = EPABCSIS(data2, tree, ϕprior, 10000, 5, target=500, miness=10., λ=0.1, α=1e-4, c=0.95)
trace4 = ep!(alg4, 5)
maptree4 = SmoothTree.addmapbranches!(t2, spmap, alg4.model)

# analyze posteriors
algs = [alg2b, alg4, alg1b, alg3]
labels = ["CCD", "CCD (tree 2)", "ML", "ML (tree 1)"]
trees = [maptree2, maptree4, maptree1, maptree3] 

species = Dict("Scas"=>"S. castellii", 
               "Smik"=>"S. mikatae", 
               "Spar"=>"S. paradoxus", 
               "Skud"=>"S. kudriavzevii", 
               "Sbay"=>"S. bayanus", 
               "Sklu"=>"S. kluyveri", 
               "Scer"=>"S. cerivisiae", 
               "Calb"=>"C. albicans")

ps = [plot(relabel(trees[i], species), transform=true, pad=1.5, fs=8,
           fontfamily="helvetica oblique", title=('A':'D')[i], scalebar=10) for i in 1:4]
p0 = plot(ps..., layout=(1,4), size=(600,200), left_margin=3mm)

# Plot the site marginals...
p1 = plot(legend=:bottomright, ylabel=L"C_i", xlabel="site \$i\$", title="E",
          fg_legend=:transparent)
for (label, alg) in zip(labels,algs)
    plot!(p1, sort(alg.siteC), lw=2, label=label)
end
plot(p1)
p2 = plot(xlabel="site \$i\$", title="F")
o = sortperm(algs[1].siteC)
marker = (markerstrokewidth=0.5, markersize=2.5)
for i in [1,2]
    scatter!(p2, algs[i].siteC[o], color=i, markerstrokecolor=i; marker...)
end
plot(p2)
p3 = plot(xlabel="site \$i\$", title="G")
o = sortperm(algs[3].siteC)
for i in [3,4]
    scatter!(p3, algs[i].siteC[o], color=i, markerstrokecolor=i; marker...)
end
plot(p3)
p4 = plot(p1, p2, p3, layout=(1,3), size=(700,700/3√2), guidefont=8,
          bottom_margin=3.5mm, left_margin=2mm)
plot(p0, p4, layout=(2,1), size=(700, 2*700/3√2))
savefig("docs/img/yeast-panel.pdf")


M  = alg2b.model
bs = SmoothTree.getbranchapprox(M.ϕ, randbranches(M))

bs = filter(x->!SmoothTree.isleafclade(x[2]), bs)

ps = []
for (i, (γ, δ, d)) in enumerate(bs)
    p = plot(Normal(log(μ), √V), 
             fill=true, color=:lightgray, fillalpha=0.5,
             title=bitstring(δ), titlefont=6, xlim=(-2,8))
             #xticks=i<8 ? false : :auto, 
             #yticks=(i-1)%7 == 0 ? :auto : false)
    vline!([d.μ], ls=:dot, color=:black)
    plot!(d, color=:black, xlabel=i > 3 ? L"\log \phi" : "")
    push!(ps, p)    
end 
plot(ps...)
plot(ps..., size=(600, 2*600/3√2), bottom_margin=3mm)

savefig("docs/img/yeast-phi.pdf")
