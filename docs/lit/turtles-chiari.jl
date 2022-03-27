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
    l = Locus(ts, spmap, α=1e-4, prior=BetaSplitTree(-1.5, ntaxa), tag=f)
    push!(data, l)
end

serialize("docs/data/turtles/chiari-mb10000.jls", (data, spmap))


# Constructing a prior distribution
# =================================
# Using some constraints
# ------------------------
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


# Using bootstrap data sets for ASTRAL
# ------------------------------------
ts = map(x->readlines(x)[1001:end], treefiles)

function get_bootstrap_samples(trees, n, N, target)
    smples = Vector{String}[]
    while length(smples) < n
        smple = [rand(rand(trees)) for i=1:N]
        ls = mapreduce(x->name.(getleaves(readnw(x))), vcat, smple)
        length(unique(ls)) == target && push!(smples, smple)
    end
    return smples
end

boots = get_bootstrap_samples(ts, 100, 1000, 16)

function run_astral(trees, astral_path="/home/arzwa/bin/Astral/astral.5.5.3.jar")
    f, _ = mktemp()
    write(f, join(trees, "\n"))
    run(`java -jar $astral_path -i $f -o $f.out`)
    rm(f)
    S = readline("$f.out")
    rm("$f.out")
    return S
end

astral_results = map(run_astral, boots)

ts = rootall!(readnw.(astral_results), "protopterus")
write("docs/data/turtles/chiari-astral-bootstrap1000.nw", join(nwstr.(ts), "\n"))

Xs = SplitCounts(ts, spmap)
Sprior1 = CCD(Xs, BetaSplitTree(-1., ntaxa), 1.)
Sprior2 = CCD(Xs, BetaSplitTree(-1., ntaxa), 10.)

using Printf
rs = first(ranking(randtree(Sprior2, spmap, 10000)), 4)
ps = map(t->plot(SmoothTree.relabel(t[1], species), transform=true,
                 fontfamily="helvetica oblique", pad=1.5,
                 title=@sprintf("p = %.2f", t[2])), rs)
plot(ps..., layout=(1,4), size=(800,250))

savefig("docs/img/chiari-astral-ccd-prior.pdf")


# Analysis
# ========
# On the cluster...
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, StatsBase, Serialization
using Plots, Distributions, StatsPlots, Measures, LaTeXStrings
theme(:wong2)
default(gridstyle=:dot, legend=false, title_loc=:left, titlefont=8, framestyle=:box)

data, spmap = deserialize("docs/data/turtles/chiari-mb10000.jls")
root = rootclade(spmap)
ntaxa = length(spmap)

ts = readnw.(readlines("docs/data/turtles/chiari-astral-bootstrap1000.nw"))
Sprior = CCD(SplitCounts(ts, spmap), BetaSplitTree(-1., ntaxa), 10.)

μ, V = log(2.), 5.
ϕprior = BranchModel(root, gaussian_mom2nat([μ, V]))
model  = MSCModel(Sprior, ϕprior)

using SmoothTree: EPABCSIS
alg = EPABCSIS(data, model, 10000, 5, target=1000, miness=10., λ=0.1, α=1e-4, c=0.95, prunetol=1e-5)
trace1 = ep!(alg, 2)

algs1 = map(deserialize, readdir("docs/data/turtles/chiari-alg4/", join=true))
algs2 = map(deserialize, readdir("docs/data/turtles/chiari-alg4-2/", join=true))

algs = [algs1[end], algs2[end]]

# get species names dictionary
species = Dict("emys_orbicularis"=>"Emys",
               "phrynops"=>"Phrynops",
               "alligator"=>"Alligator",
               "caiman"=>"Caiman",
               "python"=>"Python",
               "chelonoidis_nigra"=>"Chelonoidis",
               "podarcis"=>"Podarcis",
               "caretta"=>"Caretta",
               "protopterus"=>"Protopterus")
for (k,v) in spmap
    !haskey(species, v) && (species[v] = v)
end


# Branch posterior
# ----------------
M1 = algs[1].model
bs = SmoothTree.getbranchapprox(M1.ϕ, randbranches(M1))
bs = filter(x->!SmoothTree.isleafclade(x[2]), bs)
xs = map(bs) do (γ, δ, d)
    p = plot(Normal(log(μ), √V), fill=true, color=:lightgray, fillalpha=0.5, xlim=(-4,6))
    for alg in [algs1[end], algs2[end]]
        m, s = SmoothTree.gaussian_nat2mom(alg.model.ϕ[(γ,δ)])
        plot!(Normal(m, √s), title=bitstring(δ), fill=true, fillalpha=0.2)
    end
    #for alg in algs
    #    m, s = SmoothTree.gaussian_nat2mom(alg.model.ϕ[(γ,δ)])
    #    plot!(Normal(m, √s), title=bitstring(δ))
    #end
    p
end 
for p in xs[end-1:end]
    xlabel!(p, L"\log \phi")
end
xs = [xs[1:end-2] ; plot(framestyle=:none) ; xs[end-1:end]]
plot(xs..., size=(700,550))

savefig("docs/img/chiari-phi.pdf")


# Tree an outliers
# ================
a = algs[2]
b = algs[1]
mt = SmoothTree.relabel(SmoothTree.maptree(a.model, spmap), species)
p1 = plot(mt, transform=true, fontfamily="helvetica oblique", scalebar=10,
          xlabel=L"10N_e", pad=0.5, size=(350,350), title="(A)")
o = sortperm(a.siteC)
p2 = scatter(a.siteC[o], ylabel=L"C_i", xlabel="site \$i\$", title="(B)",
             legend=false, color=2, ms=4, left_margin=8mm, alpha=0.5, markerstrokecolor=2)
scatter!(b.siteC[o], ms=4, color=4, alpha=0.5, markerstrokecolor=4)
plot(p1, p2, size=(600,300), layout=grid(1,2, widths=[0.33,0.66]), bottom_margin=5mm)

savefig("docs/img/chiari-map.pdf")

# an interesting outlier
C40 = algs[2].siteC[40]
treefiles = readdir("docs/data/turtles/chiari-mb", join=true)
treefiles[40]

function getconsensus(infile, outgroup)
    f, _ = mktemp()
    run(`iqtree --contree $infile -o $outgroup --prefix $f`)
    tree = readnw(readline("$f.contree"))
    rm(f); rm("$f.contree")
    return tree
end

t = getconsensus(treefiles[40], "Xenopus")
t = relabel(set_outgroup!(t, "Xenopus"), species)
for n in postwalk(t)
    isleaf(n) && continue
    !isnan(n.data.support) && (n.data.name = @sprintf("%2d", n.data.support))
end

p = plot(p1, p2, size=(600,300), layout=grid(1,2, widths=[0.33,0.66]), bottom_margin=5mm)
plot!(p, inset = (2, bbox(0.25, 0.23, 0.6, 0.75, :top, :left)), subplot = 3,)
insetp = p[3]
p[2].attr[:fontfamily_subplot] = "sans-serif"
p[1].attr[:titlefontfamily] = "sans-serif"
plot!(insetp, t, transform=true, internal=true, fs=7, fontfamily="helvetica oblique")
plot!(p[2], [60,10], [-25, C40], color=:black, arrow=true)
plot(p)

savefig("docs/img/chiari-map.pdf")


# Outliers
# --------
cut = -20
x1 = findall(x->x<cut, algs[1].siteC)
x2 = findall(x->x<cut, algs[2].siteC)
xs = union(x1, x2)

ps = map(xs) do i
    locus = algs[1].data[i]
    mapt = relabel(maptree(locus), species)
    plot(mapt, transform=true, internal=true, fontfamily="helvetica oblique",
         title="locus $i", linealpha=0.4)
end
plot(ps..., layout=(3,4), color=:gray, size=(900,700))

# Brown & Thomson loci...
i = findfirst(x->x == "docs/data/turtles/chiari-mb/my_ENSGALG00000011434.macse_DNA_gb.nex.treesample", treefiles)
j = findfirst(x->x == "docs/data/turtles/chiari-mb/my_ENSGALG00000008916.macse_DNA_gb.nex.treesample", treefiles)
for a in algs
    @show a.siteC[i], a.siteC[j]
end

outliers = treefiles[xs]
ts = map(outliers) do x
    s = split(split(x, "_")[2], ".")[1]
    ts = readline(x)
    out = contains(ts, "protopterus") ? "protopterus" : "Xenopus"
    t = getconsensus(x, out)
    t = relabel(set_outgroup!(t, out), species)
    for n in postwalk(t)
        isleaf(n) && continue
        !isnan(n.data.support) && (n.data.name = @sprintf("%2d", n.data.support))
    end
    t, s
end

ps = [plot(t, transform=true, internal=true, fontfamily="helvetica oblique",
           title=s, linealpha=0.4) for (t,s) in ts]
plot(ps..., layout=(3,4), color=:gray, size=(900,700))

savefig("docs/img/outlierloci.pdf")
