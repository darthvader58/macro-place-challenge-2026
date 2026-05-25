# Audit — `submissions/will_seed/placer.py`

_Generated 2026-05-19. Phase 1, Sub-agent B (executed inline by lead after sub-agent permission denial)._

## TL;DR (5 bullets)

1. **will_seed is a wirelength-only refiner over a RePlAce seed.** It legalizes the initial `.plc` placement, then runs 3000 SA iters that optimize an unweighted L1-pair wirelength proxy. Density and congestion — which together are ~95% of the real proxy cost — are never seen by the optimizer.
2. **No orientation handling.** Klein-4 (N/FN/FS/S) is never explored. All macros stay in their initial orientation. CLAUDE.md flags this as a major leave-on-the-table.
3. **No soft-macro co-optimization.** Soft macros are explicitly kept at initial positions. After hard macros move, soft positions become stale and degrade density + WL.
4. **Runtime is massively under-budget.** Across ibm01/09/12/18 the placer finishes in 0.9–2.4 s wall-clock. The 1-hour-per-benchmark cap leaves a **1500–4000× compute multiplier** unused.
5. **Avg proxy 1.5336 ⇒ −5.2% vs RePlAce.** It loses to RePlAce on 14 of 17 benchmarks and only beats it on ibm02/ibm10/ibm12 (where RePlAce happens to be weakest). The cost decomposition shows congestion (1.3–2.6) dwarfing density (0.75–1.05) dwarfing wirelength (0.05–0.08). **All gains are in congestion/density; WL is already saturated.**

## 1. Strategy walk-through

### Initialization
`benchmark.macro_positions` comes from `loader.py`, which calls `plc.restore_placement(plc_file, ifInital=True, …)`. For IBM benchmarks this is the RePlAce initial.plc (the same one whose avg proxy is 1.4578). So will_seed inherits a strong seed and only has to refine it.

### Legalization (`_legalize`)
Greedy min-displacement spiral search:
- Order macros by descending area.
- For each: if no overlap with already-placed macros (with 0.05 µm safety gap), keep position.
- Otherwise, spiral outward in concentric rings (step = 25% of max(w,h)) up to radius 150, picking the closest legal point.
- O(N) overlap check per probed cell ⇒ worst case O(N · R⁴) for one macro.

This works for the RePlAce seed which is mostly legal, but it has no notion of channel structure or PDN spacing — it just plops each macro at its nearest legal position.

### Refinement (`_sa_refine`) — **WL-only**
A geometric-cooling SA (T_start = 0.15·canvas, T_end = 0.001·canvas, 3000 steps):
- Move types: SHIFT (50%, Gaussian step ∝ T), SWAP-with-neighbor (30%), MOVE-TOWARD-NEIGHBOR (20%).
- Cost = Σₑ wₑ · (|Δxₑ| + |Δyₑ|) where edges are derived by collapsing each net to its set of hard-macro members, with clique weights wₑ = 1/(|members|−1). Pin offsets and I/O ports are not in the cost.
- Acceptance: standard Metropolis on Δcost / T.
- Overlap check: per-move single-macro O(N) numpy broadcast with 0.05 µm gap.

### Orientation
**Not searched.** Macros keep whatever orientation the initial `.plc` gave them.

### Soft macros
**Not moved.** Source comment: _"they were already optimized for the initial hard macro layout and minimal legalization preserves this."_ But once hard macros move (legalize + SA), the soft positions are no longer a good fit.

### Plc-loading inside `place()`
`_load_plc` re-parses the netlist from disk just to walk `plc.nets` for edge extraction. This is redundant — the evaluator already has the `plc` object — but the placer interface (`place(benchmark)`) doesn't expose it. Cost: 0.3–1.1 s per benchmark (see timings).

## 2. Where it spends its time

Instrumented copy at `_archive/will_seed_instrumented.py` (identical logic + `time.perf_counter()` around each stage). Ran on the four representative benchmarks:

| benchmark | n_hard | n_edges | setup | edges (parse+build) | legalize | sa (3000) | emit | **TOTAL** |
|-----------|-------:|--------:|------:|--------------------:|---------:|----------:|-----:|----------:|
| ibm01     |   246  |    644  | 0.020 | 0.265               | 0.908    | 0.037     | 0.003| **1.23 s** |
| ibm09     |   253  |    241  | 0.012 | 0.480               | 0.361    | 0.035     | 0.000| **0.89 s** |
| ibm12     |   651  |   1813  | 0.012 | 1.112               | 1.271    | 0.046     | 0.000| **2.44 s** |
| ibm18     |   285  |    719  | 0.013 | 0.950               | 0.285    | 0.040     | 0.000| **1.29 s** |

Read these as: edge-extraction is the second-biggest cost (Python string-split per pin), legalization dominates when the seed isn't already legal (ibm01, ibm12), and **SA at 3000 iters is essentially free**. There is no benchmark where will_seed is anywhere near the 1-hour-per-benchmark cap; the smallest measured headroom is ibm12 at **1480×**.

## 3. Per-benchmark proxy decomposition (full sweep, from `_archive/log_will_seed.txt`)

```
benchmark   proxy     wl       den      cong    notes
ibm01      1.2920   0.074    1.047    1.390   small, densest of small set
ibm02      1.6798   0.077    0.839    2.366   
ibm03      1.4043   0.080    0.830    1.818   
ibm04      1.4478   0.073    0.907    1.842   
ibm06      1.7965   0.069    0.819    2.636   high-cong outlier
ibm07      1.5903   0.067    0.978    2.069   
ibm08      1.5877   0.071    0.925    2.108   
ibm09      1.1625   0.059    0.889    1.318   
ibm10      1.4116   0.071    0.752    1.929   
ibm11      1.2547   0.055    0.921    1.479   
ibm12      1.6528   0.060    0.811    2.374   biggest n_hard, beats RePlAce
ibm13      1.4113   0.054    0.920    1.794   
ibm14      1.6515   0.053    1.013    2.184   high-density
ibm15      1.6379   0.059    0.976    2.181   
ibm16      1.5484   0.050    0.893    2.103   
ibm17      1.7493   0.054    0.956    2.434   
ibm18      1.7921   0.054    1.042    2.435   high-density, high-cong
```

WL contribution to proxy: 3–6% across the board. Density: 25–35%. Congestion: 60–70%. **Optimizing WL alone explains why will_seed plateaus at −5% vs RePlAce.**

## 4. Klein-4 orientations

Searched `submissions/will_seed/placer.py` for `orient`, `R0`, `R90`, `FN`, `FS`, `flip`, `rotation` — **zero hits**. The orientation each macro lands in is whatever PlacementCost reports from the initial `.plc`. Since pin offsets rotate with orientation, this leaves an estimated 1–3% proxy improvement on the table, larger on pin-heavy / SRAM-dominated designs (= the NG45 designs).

## 5. Soft macro co-optimization

`benchmark.num_soft_macros` is referenced only through `num_hard_macros` (used to slice the position tensor). The placer explicitly comments _"Keep soft macros at initial positions — they were already optimized for the initial hard macro layout."_ For Tier 1 this affects density (soft cells contribute to the density grid) and the routing congestion estimate; for Tier 2 it would also hurt detailed-place WL. CLAUDE.md notes that the SA baseline interleaves FD on soft macros — will_seed is leaving that off.

## 6. Top-3 improvements ranked by expected proxy-gain × (1 / effort)

1. **Make the SA cost the actual proxy, not unweighted L1.** Replace `wl_cost()` with periodic `compute_proxy_cost(pos, benchmark, plc)` (e.g., every K=50 moves) or with a fast surrogate that adds density + congestion bounds — even a coarse density-and-congestion approximation evaluated every move. Since WL is already 3–6% of proxy, the SA is currently optimizing the wrong objective. **Expected: 10–25% proxy reduction** (we're presently chasing an objective that only sees ~5% of the real signal). **Effort: 2–4 hours** (need a fast incremental density/cong update; full `compute_proxy_cost` per move is too slow). On benchmarks like ibm06/ibm17/ibm18 where congestion is the killer, this is the single biggest lever.

2. **Klein-4 orientation pass after positions settle.** One sweep over hard macros: for each, try all 4 K4 orientations (N/FN/FS/S), keep whichever lowers `compute_proxy_cost`. Use `plc.modules_w_pins[idx].set_orientation(...)` style API if it exists, otherwise reflect pin offsets in-place. Re-run for 2–3 sweeps until no improvement. **Expected: 1–4% proxy reduction**, larger on benchmarks with pin-heavy macros (NG45 SRAMs). **Effort: 2–3 hours**, including verifying we never emit a non-K4 orientation.

3. **Soft-macro FD pass at the end** (or even interleaved). One `plc.optimize_stdcells` call after legalization+SA, with the documented "spread-then-attract" schedule from CLAUDE.md §5. This pulls soft cells off congested rows and improves both density and congestion. **Expected: 3–8% proxy reduction** (largest on ibm14/ibm17/ibm18 where density is already near 1.0). **Effort: 1–2 hours** to wire and budget; reportedly minutes-per-call so we'd cache plc state and skip on benchmarks where it doesn't help.

Honorable mentions (smaller, but cheap):
- **Crank SA iterations 100×.** From 3 000 → 300 000 still finishes in under 5 s. Adds a few % even on the existing WL-only objective. Trivial — change a constant. Should be combined with (1) above to matter.
- **Multi-start with 2–3 seeds**, pick best by full `compute_proxy_cost`. Budget allows ~1 minute total. Hedges against bad starts on ibm01 (the worst-vs-RePlAce design).
- **Plc-pass-through.** Stop re-loading `plc` inside `place()` (saves 0.3–1.1 s/benchmark) by caching across calls or accepting it from a constructor — only matters because the savings can be spent on SA.

## 7. Cross-cuts to remember

- **Float32 precision**: will_seed uses float64 in the SA core but emits float32 (`torch.tensor(pos, dtype=torch.float32)`). The 0.05 µm legalization gap is generous enough to survive the float32 quantization that `validate_placement` will apply.
- **Determinism**: seeded (`torch`, `random`, `numpy`) — good. SA acceptance uses Python `random.random()` which is deterministic given the seed.
- **Fixed macros**: `movable` mask is honored throughout `_legalize` and `_sa_refine`. Fixed macros are never proposed in `random.choice(movable_idx)`.
- **Bounded moves**: every move uses `np.clip(..., half_w, cw-half_w)` so canvas violations are impossible by construction.
