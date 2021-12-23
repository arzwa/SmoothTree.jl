using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree
using StatsBase, Distributions, Plots
using Serialization
default(gridstyle=:dot, legend=false, framestyle=:box, title_loc=:left, titlefont=7)

#trees = map(readdir("docs/data/mammals-song/424genes", join=true)) do dname
#    fpath = joinpath(dname, "raxmlboot.gtrgamma/RAxML_bootstrap.all")
#    @info fpath
#    ts = readnw.(readlines(fpath))
#    ts = SmoothTree.rootall!(ts, "GAL")
#    countmap(ts)
#end
#
#serialize("docs/data/mammals-song/424genes-raxmlboot.jls", trees)

trees = readnw.(readlines("docs/data/mammals-song/mltrees.nw"))

deserialize("docs/data/mammals-song/424genes-raxmlboot.jls")

tmap = taxonmap(first(trees[1])[1], UInt64)

data = CCD.(trees, lmap=tmap, α=1e-11)

Sprior = NatBMP(CCD(trees, lmap=tmap, α=1e-20))
#smple  = ranking(randtree(MomBMP(Sprior), 10000))
θprior = [SmoothTree.gaussian_mom2nat(log(1.), 5.)...]

model = MSCModel(Sprior, θprior)
alg   = EPABC(data, model, λ=0.2, α=1e-15)

trace = SmoothTree.ep_iteration!(alg, 1, maxn=1e5, mina=5, target=50)
