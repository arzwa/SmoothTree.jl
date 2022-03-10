using Pkg; Pkg.activate(@__DIR__)
using SmoothTree, NewickTree, StatsBase, BenchmarkTools, Distributions
import SmoothTree: CCD, clademap, SplitCounts, rootall, randsplits

trees = readnw.(readlines("test/mammals-example.nw"))
trees = rootall(trees, "Gal")
spmap = clademap(trees[1], UInt64)

ccd = CCD(SmoothTree.SplitCounts(trees, spmap))
@benchmark randsplits($ccd)

#@benchmark randtree($ccd, $spmap)
x = randsplits(ccd);
@benchmark logpdf($ccd, $x)

#=
Previous version

julia> @benchmark randsplits($ccd)
BenchmarkTools.Trial: 10000 samples with 8 evaluations.
 Range (min … max):  3.658 μs … 714.721 μs  ┊ GC (min … max):  0.00% … 98.85%
 Time  (median):     3.961 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   5.003 μs ±  21.550 μs  ┊ GC (mean ± σ):  16.58% ±  3.83%

  ▁▅▇███▇▆▆▄▄▄▃▂▂▁ ▁                ▁▁▁ ▁▁▁▁ ▁                ▂
  ██████████████████████▇▇▆▆▇▇▇▇▇████████████████▇▆▆▅▆▄▅▄▅▅▄▄ █
  3.66 μs      Histogram: log(frequency) by time      6.93 μs <

 Memory estimate: 9.19 KiB, allocs estimate: 111.

julia> @benchmark randtree($ccd, $spmap)
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  10.493 μs …   6.161 ms  ┊ GC (min … max):  0.00% … 99.58%
 Time  (median):     11.702 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   15.332 μs ± 127.799 μs  ┊ GC (mean ± σ):  18.59% ±  2.23%

      ▃▅▇█▆▃▁
  ▁▂▃▇███████▇▅▄▃▃▃▂▂▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▁▁▁▁▁▁▁▁▁▁▁ ▂
  10.5 μs         Histogram: frequency by time         18.6 μs <

 Memory estimate: 26.80 KiB, allocs estimate: 408.

julia> x = randsplits(ccd);

julia> @benchmark logpdf($ccd, $x)
BenchmarkTools.Trial: 10000 samples with 10 evaluations.
 Range (min … max):  1.776 μs …   3.835 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     1.789 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   1.844 μs ± 165.062 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ██  ▃▂  ▁  ▁   ▁    ▁   ▁▁   ▂    ▁                 ▁     ▁ ▂
  ███████▇█▇▆██▅▅██▅▄▇█▅▃▁██▄▃▁██▄▄▄█▇▃▃▄██▃▃▃▁▇▇▃▄▄▄██▆▆▅▄▃█ █
  1.78 μs      Histogram: log(frequency) by time      2.51 μs <

 Memory estimate: 0 bytes, allocs estimate: 0.
=#

