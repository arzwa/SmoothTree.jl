
using NewickTree, SmoothTree, Serialization

basedir = "/home/arzwa/dev/DroseraGenomeEvolution/data/loci-perfect/trimmed"
trees = map(readdir("$basedir/ufboot", join=true)) do f
    @info f
    trees = readnw.(readlines(f))
    trees = SmoothTree.rootall!(trees, "bvu_1")
    SmoothTree.topologize!.(trees)
    countmap(trees)
end

serialize("$basedir/trees.jls", trees)

tmap = taxonmap(first(trees[1])[1])
data = CCD.(trees, lmap=tmap, α=0.01)

mtrees = first.(first.(trees))
Sprior = NatBMP(CCD(mtrees, α=1.))
smple  = ranking(randtree(MomBMP(Sprior), 1000))
θprior = [SmoothTree.gaussian_mom2nat(log(1.), 5.)...]

model = MSCModel(Sprior, θprior)
alg   = EPABC(data, model, λ=0.2, α=1e-20)

trace = SmoothTree.ep!(alg, 1, maxn=2e4, mina=5, target=50)

