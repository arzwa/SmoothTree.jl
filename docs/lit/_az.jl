using Plots, StatsPlots, LaTeXStrings, Parameters, Distributions, Measures
default(legend=false, gridstyle=:dot, framestyle=:box, titlefont=9, title_loc=:left)
@unpack μ, V = config

Ms = first.(trace)

mapS = first(smple)[1]
nodes = filter(n->!isleaf(n) && !isroot(n), postwalk(mapS))
clades = map(n->(id(parent(n)), id(n)), nodes) 
pls = map(clades) do g
    p = plot(Normal(log(μ), √V),color=:black, xlim=(-4.5,4.5), 
             grid=false, ylabel=L"p(\phi)")
    xs = Ms[1:200:end]
    for (i,model) in enumerate(xs)
        lm, VV = SmoothTree.gaussian_nat2mom(model.q[g])
        i == length(xs) && plot!(Normal(lm, √VV), color=:lightgray, fill=true,
                                 fillalpha=0.5)
        plot!(Normal(lm, √VV), color=:black)
    end
    vline!([log(1/(ntaxa-1))], color=:black, ls=:dot, xlabel=L"\phi")
    p
end
title!(pls[1], "(A)")
p1 = plot(pls..., size=(800,150))

final_splits = SmoothTree.allsplits(alg.model.S)
all_splits = [final_splits; [(γ,γ-δ) for (γ,δ) in final_splits]]
qs = map(x->x.q, Ms)
splittrace = mapreduce(x->[MomMBM(x.S)[γ,δ] for (γ, δ) in final_splits], hcat, Ms) |> permutedims
mutrace = mapreduce(x->[x[(γ,δ)][1] for (γ, δ) in all_splits], hcat, qs) |> permutedims
vtrace  = mapreduce(x->[x[(γ,δ)][2] for (γ, δ) in all_splits], hcat, qs) |> permutedims

kwargs = (xlabel="iteration", xguidefont=8)
ps4 = [
       plot(splittrace, title="(B)", ylabel=L"\theta"; kwargs...), 
       plot(mutrace, ylabel=L"\mu", xguidefont=8; kwargs...), 
       plot(vtrace, ylabel=L"\sigma^2"; kwargs...)
      ]
pl4 = plot(ps4..., layout=(1,3), size=(900,200))

plot(p1, pl4, layout=(2,1), size=(700,300), bottom_margin=3mm, left_margin=4mm)

savefig("docs/img/anomaly-5taxa-500trees.pdf")
