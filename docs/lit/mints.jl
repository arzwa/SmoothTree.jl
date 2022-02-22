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
using SmoothTree, NewickTree
nepetoideae = readnw(readline("docs/data/mints/nepetoideae.nw"))
taxa = name.(getleaves(nepetoideae))
tmap = clademap(taxa, UInt32)

treefiles = readdir("docs/data/mints/nepetoideae-mb/trees", join=true)
loci = map(treefiles[1:10]) do f
    @info f
    ts = readnw.(readlines(f)[1001:10:end])
    ts = SmoothTree.rootall!(ts)
    proportionmap(ts)
end

locus = Locus(loci[1], tmap, rooted=false)

root = UInt32(sum(keys(tmap)))
Sprior = NatMBM(root, BetaSplitTree(-1.5, 24))
θprior = BranchModel(root, SmoothTree.gaussian_mom2nat([0.,5]))
model  = MSCModel(Sprior, θprior)
bs = randbranches(model)
ss = randsplits(bs, locus.init)
