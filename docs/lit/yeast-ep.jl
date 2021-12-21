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
alg = SmoothTree.EPABC(data, model, λ=0.1, α=1e-3)

trace = SmoothTree.ep!(alg, 2, maxn=5e4, mina=5, target=20)

function traceback(trace)
    clades = keys(trace[end].S.smap)
    splits = Dict(γ=>collect(keys(trace[end].S.smap[γ].splits)) for γ in clades)
    traces = Dict(γ=>Vector{Float64}[] for γ in clades)
    for i=length(trace):-1:1
        bmp = SmoothTree.MomBMP(trace[i].S)
        for γ in clades
            x = map(δ->haskey(bmp, γ) ? bmp[γ][δ] : NaN, splits[γ])
            push!(traces[γ], x)
        end
    end
    return Dict(γ=>permutedims(hcat(reverse(xs)...)) for (γ, xs) in traces)
end

X = traceback(trace)

traces = map(ultimate_clades[2:end]) do clade
    plot(X[clade], legend=false, yscale=:log10, framestyle=:box,
         gridstyle=:dot, title="clade $(bitstring(clade)[end-7:end])",
         title_loc=:left, titlefont=7)#, xscale=:log10)
end |> x->plot(x...)

smple  = SmoothTree.ranking(randtree(mtrace[end], 10000))

maxtree = SmoothTree.ranking(randtree(CCD(trees), 1000))[1][1]
ultimate_clades = map(SmoothTree.getclades(maxtree)) do x
    UInt16(sum([invmap[y] for y in name.(x)]))
end
ultimate_clades = filter(x->!SmoothTree.isleafclade(x), ultimate_clades)

