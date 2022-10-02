using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, ThreadTools, StatsBase, Serialization
using ProgressBars


# Data preparation
# ================
treefiles = readdir("docs/data/turtles/chiari-mb", join=true)
ls = unique(mapreduce(tf -> name.(getleaves(readnw(readline(tf)))), vcat, treefiles))
spmap = clademap(ls)
ntaxa = length(spmap)

ts = readnw.(readlines(treefiles[1]))
ts = rootall!(ts, "Xenopus")
ccd = CCD(SplitCounts(ts, spmap))
locus = Locus(ts, spmap, α=1e-4, prior=BetaSplitTree(-1.5, ntaxa))

data = typeof(locus)[]
for f in ProgressBar(treefiles)
    ts = readnw.(readlines(f)[1001:1:end])
    out = "protopterus" ∈ name.(getleaves(ts[1])) ? "protopterus" : "Xenopus" 
    ts = rootall!(ts, out)
    l = Locus(ts, spmap, α=1e-4, prior=BetaSplitTree(-1.5, ntaxa))
    push!(data, l)
end

serialize("docs/data/turtles/chiari-mb10000.jls", (data, spmap))


# Constructing a prior distribution
# =================================
splitdict = Dict{UInt16, Dict{UInt16, Int64}}()
c1 = getclade(spmap, ["protopterus"])
splitdict[root] = Dict(min(c1, root-c1)=>1)
c2 = root - c1
c3 = getclade(spmap, ["Xenopus"])
splitdict[c2] = Dict(min(c3, c2-c3)=>1)
c4 = c2-c3
c5 = getclade(spmap, ["Homo", "Monodelphis", "Ornithorhynchus"])
splitdict[c4] = Dict(min(c5, c4-c5)=>1)
X = SplitCounts(splitdict, root)
ccd = CCD(X, BetaSplitTree(-1., ntaxa), 0.1) 
randtree(ccd, spmap, 10000) |> ranking


trees = []
for tf in treefiles
    length(getleaves(readnw(readline(tf)))) != 16 && continue
    @info tf
    ts = readnw.(readlines(tf)[1001:end])
    ts = rootall!(ts, "protopterus")
    push!(trees, ts...)
end

SmoothTree.topologize!.(trees)
X = SplitCounts(trees, spmap)
Sprior = CCD(X, BetaSplitTree(-1., ntaxa), .0001)

serialize("docs/data/turtles/chiari-splitcounts-complete.jls", X)

ts = first(ranking(randtree(Sprior, spmap, 10000)), 16)
plot([plot(t[1], transform=true) for t in ts]..., size=(1000,1000))



# Analysis
# ========
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, StatsBase, Serialization
using Plots, Distributions, StatsPlots, Measures, LaTeXStrings
default(gridstyle=:dot, legend=false, title_loc=:left, titlefont=8, framestyle=:box)

data, spmap = deserialize("docs/data/turtles/chiari-mb10000.jls")
root = rootclade(spmap)
ntaxa = length(spmap)

μ, V = log(2.), 2.
#Sprior = CCD(SplitCounts(smple, spmap), BetaSplitTree(-1., ntaxa))
Sprior = CCD(SplitCounts(root), BetaSplitTree(-1., ntaxa))
ϕprior = BranchModel(root, gaussian_mom2nat([μ, V]))
model  = MSCModel(Sprior, ϕprior)

using SmoothTree: EPABCSIS
alg = EPABCSIS(data, model, 10000, 5, target=1000, miness=10., λ=0.1, α=1e-4, c=0.95, prunetol=1e-5)
trace1 = ep!(alg, 2)

#alg1 = deserialize("docs/data/turtles/chiari-alg1-10passes.jls")
alg2 = deserialize("docs/data/turtles/chiari-alg1/chiari-alg1-10passes-2.jls")
#alg3 = deserialize("docs/data/turtles/chiari-alg1-10passes-3.jls")
alg4 = deserialize("docs/data/turtles/chiari-alg1/chiari-alg1-10passes-4.jls")

species = Dict("emys_orbicularis"=>"Emys",
               "phrynops"=>"Phrynops",
               "alligator"=>"Alligator",
               "caiman"=>"Caiman",
               "python"=>"Python",
               "chelonoidis_nigra"=>"Chelonoidis",
               "podarcis"=>"Podarcis",
               "caretta"=>"Caretta",
               "protopterus"=>"Protopterus")

M1 = alg2.model
M2 = alg4.model
bs = SmoothTree.getbranchapprox(M1.ϕ, randbranches(M1))
bs = filter(x->!SmoothTree.isleafclade(x[2]), bs)
xs = map(bs) do (γ, δ, d)
    plot(Normal(log(μ), √V), fill=true, color=:lightgray, fillalpha=0.5, tickfont=7)
    plot!(d, color=:black, fill=true, fillalpha=0.2)
    m, s = SmoothTree.gaussian_nat2mom(M2.ϕ[(γ,δ)])
    plot!(Normal(m, √s), title=bitstring(δ), titlefont=7, fill=true, fillalpha=0.3)
    #plot(LogNormal(log(μ), √V), fill=true, color=:lightgray, xlim=(0,10))
    #plot!(LogNormal(d.μ, d.σ), color=:black)
end 
for p in xs[end-4:end]
    xlabel!(p, L"\log \phi")
end
#xs = [xs[1:end-2] ; plot(framestyle=:none) ; xs[end-1:end]]
xs = [xs ; plot(framestyle=:none) ]
plot(xs..., size=(800,350), layout=(3,5))

savefig("docs/img/chiari-phi.pdf")

mt = SmoothTree.maptree(M2, spmap)

for n in getleaves(mt)
    haskey(species, name(n)) && (n.data.name = species[name(n)]) 
end

p1 = plot(mt, transform=true, fontfamily="helvetica oblique", scalebar=5,
          xlabel=L"5N_e", pad=0.5, size=(350,350))
o = sortperm(alg4.siteC)
p2 = scatter(alg4.siteC[o], ylabel=L"C_i", xlabel="site \$i\$",
             legend=false, color=:black, ms=3, left_margin=8mm)
scatter!(alg2.siteC[o], ms=3)
plot(p1,p2, size=(600,300), layout=grid(1,2, widths=[0.33,0.66]), bottom_margin=5mm)


outlier = treefiles[findall(!isfinite, alg.siteC)]
oloci = map(outlier) do f
    ts = readnw.(readlines(f))[1001:end]
    sp = clademap(ts[1])
    out = "protopterus" ∈ name.(getleaves(ts[1])) ? "protopterus" : "Xenopus" 
    ts = rootall!(ts, out)
    l = Locus(ts, sp)
    f, l
end

map(oloci) do (f, l)
    t, p = randtree(l, 10000) |> ranking |> first
    plot(t, title=split(basename(f), ".")[1][4:end] * "\n($p)", transform=true)
end |> x-> plot(x..., layout=(1,3), size=(900,300))




