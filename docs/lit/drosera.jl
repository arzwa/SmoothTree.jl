
using NewickTree, SmoothTree, Serialization

#basedir = "/home/arzwa/dev/DroseraGenomeEvolution/data/loci-perfect/trimmed"
#trees = map(readdir("$basedir/ufboot", join=true)) do f
#    @info f
#    trees = readnw.(readlines(f))
#    trees = SmoothTree.rootall!(trees, "bvu_1")
#    SmoothTree.topologize!.(trees)
#    countmap(trees)
#end

#serialize("$basedir/trees.jls", trees)
trees = deserialize("$basedir/trees.jls")

tmap = taxonmap(first(trees[1])[1])
data = CCD.(trees, lmap=tmap, α=0.01)

speciesmap = Dict("bvu"=>["bvu_1"], 
                  "dca"=>["dca_1", "dca_2", "dca_3"], 
                  "dre"=>["dre_1", "dre_2", "dre_3"])

mtrees = first.(first.(trees))
Sprior = NatBMP(CCD(mtrees, α=1.))
smple  = ranking(randtree(MomBMP(Sprior), 1000))
θprior = [SmoothTree.gaussian_mom2nat(log(1.), 5.)...]

model = MSCModel(Sprior, θprior, tmap)
alg   = SmoothTree.MULEPABC(data, model, speciesmap, λ=0.5, α=1e-6)

trace = ep!(alg, 2, maxn=1e5, mina=10, target=50)

X, Y = traceback(trace)

xs = filter(x->size(x[2], 2) > 1, collect(X))
map(xs) do (k, x)
    p1 = plot(x, title="clade $(bitstring(k)[end-7:end])")
    p2 = plot(Y[k])
    plot(p1, p2)
end |> x-> plot(x..., size=(1400,1400))

smple  = SmoothTree.ranking(randtree(SmoothTree.MomBMP(alg.model.S), 10000))
SmoothTree.relabel(first(smple)[1], tmap)
# (((((dre_1,dca_3),(dca_2,dre_3)),dre_2),dca_1),bvu_1);
# ((((dre_1,((dca_3,dca_2),dre_3)),dre_2),dca_1),bvu_1);

obs = ranking(trees)
pps = map(1:1000) do rep
    S = SmoothTree.randsptree(trace[end])
    M = SmoothTree.MSC(S, SmoothTree.randinit(data[1], tmap, speciesmap))
    pps = proportionmap(randtree(M, tmap, length(trees)))
    xs = map(x->haskey(pps, x[1]) ? pps[x[1]] : 0., obs)
end |> x->permutedims(hcat(x...))

boxplot(pps, linecolor=:black, fillcolor=:lightgray, outliers=false)
scatter!(last.(obs), color=:black)

