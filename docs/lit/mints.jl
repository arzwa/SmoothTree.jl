using NewickTree, BioTools

# get the data
nepetoideae = readnw(readline("docs/data/mints/nepetoideae.nw"))
taxa = name.(getleaves(nepetoideae))

outdir = mkpath("docs/data/mints/nepetoideae-aln/")
dir = "/home/arzwa/research/msc/doi_10.5061_dryad.9p8cz8wc5__v7/2_Shen_et_al_Empirical_datasets/3_plant/data_matrix/gene_aligments"
alns = map(readdir(dir, join=true)) do f
    aln = BioTools.readfasta(f)
    aln = Dict(k=>v for (k,v) in aln if k ∈ taxa)
    @info length(aln)
    of = joinpath(outdir, basename(f))
    BioTools.writefasta(of, aln)
end
