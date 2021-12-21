using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree
using StatsBase, Distributions, Plots
using Serialization
data_ = deserialize("docs/data/yeast-rokas/ccd-ufboot.jls")
trees = readnw.(readlines("docs/data/yeast.trees"))
taxon_map = SmoothTree.taxonmap(name.(getleaves(trees[1])))
invmap = SmoothTree.inverse(taxon_map)

Sprior = SmoothTree.NatBMP(CCD(trees, α=5.))
smple  = SmoothTree.ranking(randtree(SmoothTree.MomBMP(Sprior), 10000))

θprior = [SmoothTree.gaussian_mom2nat(log(1.), 5.)...]

data  = CCD.(data_, α=0.01 * length(data_))
model = SmoothTree.MSCModel(Sprior, θprior, taxon_map)
alg = SmoothTree.EPABC(data, model, λ=0.1, α=0.01)

trace = SmoothTree.ep_pass!(alg, maxn=5e4)

trace = map(i->SmoothTree.ep_iteration!(alg, i, maxn=5e4), 1:length(data))
trace = [trace; map(i->SmoothTree.ep_iteration!(alg, i), 1:length(data))]

mtrace = map(x->SmoothTree.MomBMP(x.S), trace)

traces = map(ultimate_clades[2:end]) do clade
    splits = keys(mtrace[end].smap[clade].splits)
    xs = map(mtrace) do bmp
        [haskey(bmp.smap[clade].splits, x) ? 
         bmp.smap[clade].splits[x] : bmp.smap[clade].η0 for x in splits]
    end 
    X = permutedims(hcat(xs...))
    plot(X, legend=false, yscale=:log10, framestyle=:box,
         gridstyle=:dot, title="clade $(bitstring(clade)[end-7:end])",
         title_loc=:left, titlefont=7)#, xscale=:log10)
end |> x->plot(x...)

smple  = SmoothTree.ranking(randtree(mtrace[end], 10000))


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


