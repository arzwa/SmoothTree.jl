using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, ThreadTools, StatsBase, Serialization
using ProgressBars

treefiles = readdir("docs/data/turtles/BUSCO-686-trees", join=true)
t = readnw.(readlines(treefiles[1]))
@time ts = rootall!(t, "homSap");
spmap = clademap(ts[1])
@time xs = SplitCounts(ts, spmap)

data = typeof(xs)[]
for f in ProgressBar(treefiles)
    ts = readnw.(readlines(f)[1001:2:end])
    ts = rootall!(ts, "homSap")
    xs = SplitCounts(ts, spmap)
    push!(data, xs)
end

serialize("docs/data/turtles/mb5000.jls", data)

# analysis
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, StatsBase, Serialization, Plots
using Distributions, StatsPlots
default(gridstyle=:dot, legend=false, title_loc=:left, titlefont=8, framestyle=:box)

treefiles = readdir("docs/data/turtles/BUSCO-686-trees", join=true);
t = readnw.(readline(treefiles[1]))
spmap = clademap(ts[1])

smple = map(treefiles) do x
    l = rand(readlines(x)[1000:end])
    getroot(set_outgroup(readnw(l), "homSap"))
end

data = deserialize("docs/data/turtles/mb5000.jls")
root = rootclade(spmap)
ntaxa = length(spmap)

loci = map(data) do x
    x = CCD(x, BetaSplitTree(-1.5, ntaxa), 1e-4)
    Locus(x, spmap, Dict(i=>[i] for (i,_) in spmap))
end

μ, V = log(2.), 5.
Sprior = CCD(SplitCounts(smple, spmap), BetaSplitTree(-1., ntaxa))
ϕprior = BranchModel(root, gaussian_mom2nat([μ, V]))
model  = MSCModel(Sprior, ϕprior)

using SmoothTree: EPABCSIS
alg = EPABCSIS(loci, model, 10000, 5, target=500, miness=5., λ=0.1, α=1e-4, c=0.95, prunetol=1e-3)
trace1 = ep!(alg, 2)

M  = alg.model
bs = SmoothTree.getbranchapprox(M.ϕ, randbranches(M.S))
bs = filter(x->SmoothTree.isleafclade(x[2]), bs)
map(bs) do (γ, δ, d)
    plot(Normal(log(μ), √V), fill=true, color=:lightgray)
    plot!(d, color=:black)
    #plot(LogNormal(log(μ), √V), fill=true, color=:lightgray, xlim=(0,10))
    #plot!(LogNormal(d.μ, d.σ), color=:black)
end |> x->plot(x..., size=(1200,800))
