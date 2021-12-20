
using NewickTree, SmoothTree, Serialization

basedir = "/home/arzwa/dev/DroseraGenomeEvolution/data/loci-perfect/trimmed"
trees = map(readdir("$basedir/ufboot", join=true)) do f
    trees = readnw.(readlines(f))
    trees = SmoothTree.rootall!(trees, "bvu_1")
    SmoothTree.topologize!.(trees)
    countmap(trees)
end
serialize("$basedir/trees.jls", trees)

data = CCD.(trees, αroot=0., α=0.01)

a = SmoothTree.n_internal(S) - 1
treeprior = CCD(randtree(data[1], 1), α=100., αroot=0.)
θprior = MvNormal(zeros(m), 1.)

T = randtree(treeprior)
x = rand(θprior)
SmoothTree.setdistance_internal_rooted!(T, exp.(x))
simm = SmoothTree.MSC(T)
sims = CCD(simm, randsplits(simm, 10000), α=0.01, αroot=0.)


# swap leaf label clade numbers
function relabel(ccd::CCD, lmap)
    
end
