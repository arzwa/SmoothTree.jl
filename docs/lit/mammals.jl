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
#smple  = ranking(randtree(MomBMP(Sprior), 10000))
θprior = [SmoothTree.gaussian_mom2nat(log(1.), 5.)...]

model = MSCModel(Sprior, θprior)
alg   = EPABC(data[1:10], model, λ=0.2, α=1e-20)

trace = SmoothTree.ep_iteration!(alg, 1, maxn=2e4, mina=5, target=50, noisy=true)

trace = SmoothTree.ep!(alg, 1, maxn=2e4, mina=5, target=50, noisy=true, adhoc=80.)

# it does seem to lead somewhere, two passes on the first 10 loci give
# 15% posterior mass to
# ((((Mac,Mon),((((((((New,((Gor,(Hom,Pan)),Pon)),Cal),(Oto,Mic)),(Tup,Tar)),((((Spe,(Mus,Rat)),Dip),Cav),(Ory,Och))),(Sor,(((Myo,Pte),((((Tur,Bos),Vic),Sus),((Fel,Can),Eri))),Equ))),((Lox,Pro),Ech)),(Das,Cho))),Orn),Gal);
# which is not likely the correct tree, but not too far...
# if we find a good way to modify the kernel, it should work.
