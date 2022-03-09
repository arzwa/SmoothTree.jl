using Pkg; Pkg.activate(@__DIR__)
using NewickTree, SmoothTree, Plots, LaTeXStrings, Measures
default(gridstyle=:dot, framestyle=:box, titlefont=9, title_loc=:left)

bs = [-2, -1.5, -1., 0., 2., 10., Inf]
xs = [SmoothTree.betasplitpmf(β, 10) for β=bs]

p = plot(title="(A)")
for (x, b) in zip(xs,bs)
    isnan(x[1]) && (x[1] = 1.)
    y = [x ./ 2 ; reverse(x[1:end-1]) ./ 2]
    y[5] *= 2
    plot!(y, lw=2, marker=0, alpha=0.5, markerstyle=2, 
          label="\$\\beta = $b\$", legend=false)
    annotate!((5, 0.02+y[5], text("\$\\beta = $b \$", 9)))
end
p = plot(p, ylim=(-0.05, 0.55), size=(330,300), xticks=1:9, ylabel=L"q_{10}(i)",
     xlabel=L"i")

n = 50
ps = map(bs) do b
    bsd = BetaSplitTree(b, n)
    m = NatMBM(SmoothTree.rootclade(n), bsd)
    t = randtree(m)
    for n in getleaves(t)
        n.data.name=" "
    end
    plot(t, transform=true, pad=0, title="\$\\beta = $b\$")
end
p2 = plot(ps..., layout=(1,7))

title!(p2, "(B)")

plot(p, p2, layout=grid(1,2,widths=[0.2,0.8]), size=(950,250), left_margin=4mm, bottom_margin=5mm)

savefig("docs/img/betasplit.pdf")


