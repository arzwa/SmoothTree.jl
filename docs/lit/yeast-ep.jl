using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree
using StatsBase, Distributions
using Serialization
data_ = deserialize("docs/data/yeast-rokas/ccd-ufboot.jls")
trees = readnw.(readlines("docs/data/yeast.trees"))

Ψprior = SmoothTree.NatBMP(CCD(trees, α=1.))
smple  = SmoothTree.ranking(randtree(SmoothTree.MomBMP(Ψprior), 10000))

θprior = SmoothTree.mom2nat(log(1.5), 2)



# -------------------------------------------------------------------
taxon_map = SmoothTree.taxon_map(name.(getleaves(trees[1])))
invmap = SmoothTree.inverse(taxon_map)

data  = CCD.(data_, α=0.01 * length(data_))
#Ψprior = SmoothTree.UniformBMP(8)
Ψprior = SmoothTree.fitbmp(randtree(CCD(trees, α=1.), 10000), invmap)
#Ψprior = SmoothTree.fitbmp(trees, SmoothTree.inverse(taxon_map))
θprior = SmoothTree.mom2nat(0.5,2)
model = SmoothTree.MSCModel(Ψprior, θprior, taxon_map)

alg = SmoothTree.EPABC(data, model, M=10000, λ=0.1)

trace = map(i->SmoothTree.ep_iteration!(alg, i), 1:length(data))

X = log.(permutedims(hcat(map(x->collect(values(x.Ψ.smap[0x0023])), trace)...)))
plot(X)

maxtree = SmoothTree.ranking(randtree(CCD(trees), 1000))[1][1]
ultimate_clades = map(SmoothTree.getclades(maxtree)) do x
    UInt16(sum([invmap[y] for y in name.(x)]))
end
ultimate_clades = filter(x->!SmoothTree.isleafclade(x), ultimate_clades)

traces = map(ultimate_clades[2:end]) do clade
    ctrace = map(x->collect(values(x.Ψ.smap[clade])), trace)
    X = permutedims(hcat(ctrace...))
    plot(X, legend=false, yscale=:log10, framestyle=:box,
         gridstyle=:dot, title="clade $(bitstring(clade)[end-7:end])",
         title_loc=:left, titlefont=7)
end |> x->plot(x...)



# -------------------------------------------------------------------
data  = CCD.(data_, α=0.01 * length(data_))
taxa = getleaves(data[1])

Ψprior = CCD(trees, α=1.)
SmoothTree.ranking(randtree(Ψprior, 10000))
θprior = SmoothTree.moment2natural(1,1)
model = SmoothTree.MSCModel(Ψprior, θprior)

alg = SmoothTree.EPABC(data, model)

SmoothTree.ep_iteration!(alg, 1)
