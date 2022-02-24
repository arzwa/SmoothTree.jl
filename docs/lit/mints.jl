using NewickTree, BioTools

# get the data
nepetoideae = readnw(readline("docs/data/mints/nepetoideae.nw"))
taxa = name.(getleaves(nepetoideae))

# get alignments from the larger alignments
outdir = mkpath("docs/data/mints/nepetoideae-aln/")
dir = "/home/arzwa/research/msc/doi_10.5061_dryad.9p8cz8wc5__v7/2_Shen_et_al_Empirical_datasets/3_plant/data_matrix/gene_aligments"
alns = map(readdir(dir, join=true)) do f
    aln = BioTools.readfasta(f)
    aln = Dict(k=>v for (k,v) in aln if k ∈ taxa)
    @info length(aln)
    of = joinpath(outdir, basename(f))
    BioTools.writefasta(of, aln)
end

# get the input tree data
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, ThreadTools, StatsBase, Serialization
nepetoideae = readnw(readline("docs/data/mints/nepetoideae.nw"))
taxa = name.(getleaves(nepetoideae))
smap = clademap(taxa, UInt32)

treefiles = readdir("docs/data/mints/nepetoideae-mb/trees", join=true)
data = ThreadTools.tmap(treefiles) do f
    ts = readnw.(readlines(f)[1001:10:end])
    ts = SmoothTree.rootall(ts)
    SmoothTree.topologize!.(ts)
    cm = countmap(ts)
    @info f
    cm
end

serialize("docs/data/mints/mb1000.jls", data)

# analysis
using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, ThreadTools, StatsBase, Serialization, Plots
default(titleloc=:left, titlefont=9)
nepetoideae = readnw(readline("docs/data/mints/nepetoideae.nw"))
taxa = name.(getleaves(nepetoideae))
smap = clademap(taxa, UInt32)
data = deserialize("docs/data/mints/mb1000.jls")
root = UInt32(sum(keys(smap)))

loci = Locus.(data, Ref(smap), 1e-6, -1.0, rooted=false)

# now, in the unrooted case, and with missing taxa, it is less straightforward
# to obtain an informative species tree prior from the gene trees.
#complete = filter(x->length(x.lmap) == 24, loci)
#smple = mapreduce(x->randtree(x, 100), vcat, complete)
#smple = SmoothTree.rootall(smple)
#Sprior = NatMBM(CCD(smple, smap, rooted=false), BetaSplitTree(-1.0, 24), 0.001)

# or derived from the concat tree
Sprior = NatMBM(CCD(nepetoideae, smap), BetaSplitTree(-1.0, 24), 10.)

# a sample from the prior
maprior = ranking(relabel.(randtree(Sprior, 10000), Ref(smap)))

# the rest of the model
θprior = BranchModel(root, SmoothTree.gaussian_mom2nat([0.,3]))
model  = MSCModel(Sprior, θprior)

# EP-ABC
alg1   = EPABCIS(loci[1:10], model, 10000, target=2000, miness=10., λ=0.1,
                 α=1e-3, prunetol=1e-4)

trace1 = ep!(alg1, 3, traceit=10)

smple1 = relabel.(randtree(alg1.model.S, 10000), Ref(smap)) |> ranking

plot([plot(k, transform=true, title=v) for (k,v) in smple1[1:9]]..., size=(900,1200))
