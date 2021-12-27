using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, Test, NewickTree
using StatsBase, Distributions, Plots
using Serialization
default(gridstyle=:dot, legend=false, framestyle=:box, title_loc=:left, titlefont=7)

fnames = readdir("docs/data/mammals-song/424genes.ufboot", join=true)
boots = map(fnames) do fpath
    @info fpath
    ts = readnw.(readlines(fpath))
    ts = SmoothTree.rootall!(ts, "Gal")
    countmap(ts)
end

boots = SmoothTree.readtrees("docs/data/mammals-song/424genes-ufboot.trees")

trees = readnw.(readlines("docs/data/mammals-song/mltrees.nw"))
tmap = taxonmap(first(trees), UInt64)
collect(keys(tmap)) |> sort .|> bitstring

data = CCD.(boots, lmap=tmap, α=1e-11)

Sprior = NatBMP(CCD(trees, lmap=tmap, α=1e-20))
#Sprior = NatBMP(data[1])
smple  = ranking(randtree(MomBMP(Sprior), 10000))
θprior = [SmoothTree.gaussian_mom2nat(log(1.), 5.)...]

model = MSCModel(Sprior, θprior, tmap)
alg   = EPABC(data[1:100], model, λ=0.1, α=1e-20)

trace = SmoothTree.ep_iteration!(alg, 1, maxn=1e4, mina=10, target=50, adhoc=true)

trace = SmoothTree.ep!(alg, 1, maxn=1e4, mina=10, target=50, adhoc=true)

# it does seem to lead somewhere, two passes on the first 10 loci give
# 15% posterior mass to
# ((((Mac,Mon),((((((((New,((Gor,(Hom,Pan)),Pon)),Cal),(Oto,Mic)),(Tup,Tar)),((((Spe,(Mus,Rat)),Dip),Cav),(Ory,Och))),(Sor,(((Myo,Pte),((((Tur,Bos),Vic),Sus),((Fel,Can),Eri))),Equ))),((Lox,Pro),Ech)),(Das,Cho))),Orn),Gal);
# which is not likely the correct tree, but not too far...
# if we find a good way to modify the kernel, it should work.

X, Y = traceback(trace)

xs = filter(x->size(x[2], 2) > 1, collect(X))
map(xs[1:32]) do (k, x)
    p1 = plot(x, title="clade $(bitstring(k)[end-7:end])")
    p2 = plot(Y[k])
    plot(p1, p2)
end |> x-> plot(x..., size=(1200,500))

smplepost = SmoothTree.ranking(randtree(SmoothTree.MomBMP(trace[end].S), 10000))

SmoothTree.relabel(last(smplepost)[1], tmap)

