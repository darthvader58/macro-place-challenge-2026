# SOTA Macro-Placement Research — 48-Hour Adoption Survey

_Generated 2026-05-19. Phase 1, Sub-agent A (executed inline by lead after sub-agent permission denial). Constraints: must run on RTX 6000 Ada within 1 h/benchmark; air-gapped at submission; no from-scratch training; Python 3.11 + PyTorch 2.5.1 default container._

## Survey table

Columns: **Year/Venue**, **idea (1 sentence)**, **Code URL**, **Reported gain (and on which benchmarks)**, **Fits RTX 6000 in 1 h?**, **Usable pretrained checkpoint?**, **Tier-2 (post-route WNS/TNS) reported?**

| Method | Year/Venue | Algorithmic idea | Code | Reported gain | Fits 1 h on 1 GPU? | Pretrained? | Tier-2 metrics? |
|---|---|---|---|---|---|---|---|
| **DREAMPlace 4.0 / 4.1** | 2023+ (TCAD); active to 2025 | GPU-accelerated analytical placer with timing-driven net-weighting and Lagrangian refinement; 4.x adds a 2-stage macro-place flow | github.com/limbo018/DREAMPlace | Used as the strongest non-ML baseline in nearly every paper since 2023 | **Yes** (designed for single GPU) | No pretrained — it's analytical | Yes — used as backbone in AutoDMP, ReMaP for ORFS WNS/TNS |
| **AutoDMP** | ISPD 2023 (NVlabs) | DREAMPlace global+detail + MOBO hyperparameter search over mixed-size placement | github.com/NVlabs/AutoDMP | Best open-source WL & timing on TILOS bench (~30× speedup vs commercial) | Tested on DGX A100; **borderline** — needs MOBO budget cut to fit 1 h | No model; the BBO loop is the "training" | Yes — full ORFS WNS/TNS |
| **Hier-RTLMP** | TCAD 2024 (UCSD/OpenROAD) | Hierarchical SA on macro clusters derived from RTL hierarchy; production placer in OpenROAD `src/mpl` | github.com/The-OpenROAD-Project/OpenROAD/tree/master/src/mpl | Substantial gains over RTL-MP on commercial designs (TCAD 2024) | Yes — designed for CPU production use | No model | Yes — ORFS-integrated |
| **ChiPFormer** | ICML 2023 | Offline Decision-Transformer policy trained on a multi-design trajectory bank; one-shot inference + brief fine-tune per chip | github.com/laiyao1/ChiPFormer | 10–30× speedup vs online RL, beats RL baselines on ISPD05 | Yes (small transformer; trained on RTX-class GPUs) | **Yes** — `save_models/` contains ICML checkpoints | Proxy only (HPWL/congestion); no post-route in paper |
| **Macro Regulator (MaskRegulate)** | NeurIPS 2024 | RL policy that *adjusts* an existing placement (regulator framing) with regularity-mask reward | github.com/lamda-bbo/macro-regulator | Improves on prior RL methods on ISPD05/ICCAD15 (regularity + HPWL) | Yes (single GPU); inference is tiny | Likely yes (per-design fine-tune is brief; ISPD05 ckpt in repo) | Proxy + regularity only |
| **LaMPlace** | ICLR 2025 | Polynomial mask predictor of cross-stage metrics (WNS/TNS/area); greedy mask-guided placement | github.com/MIRALab-USTC/AI4EDA-LaMPlace | +9.6% avg cross-stage; **+43% WNS / +30.4% TNS** on their bench | Yes (predictor is small; greedy place is fast) | Yes (predictor weights ship with repo for their bench; need adapter for IBM/NG45) | **Yes** — explicit Tier-2 framing |
| **WireMask-BBO** | NeurIPS 2023 | Greedy placement guided by a per-cell "wire mask" estimating WL contribution; BBO outer loop optionally tunes order | github.com/lamda-bbo/WireMask-BBO | Up to **50% HPWL** improvement when *fine-tuning existing placements* (their "warm-start" mode) | Yes (CPU-friendly; mask compute is GPU-accelerated) | No model — search-based; warm-start uses our seed | HPWL only |
| **EGPlace** | ICML 2025 | Evolutionary search with greedy-repositioning-guided mutation operator | (paper; code expected at lamda-bbo, not yet verified) | −10.8% WL vs WireMask-EA; −9.3% WL vs EfficientPlace; **2.8–7.8× faster** | Yes (designed faster than WireMask) | No model — pure search | HPWL only on ISPD05 + Ariane |
| **EfficientPlace** | ICML 2024 | RL policy + Monte-Carlo tree search; constructive | not yet located on GitHub | Strong proxy results on ISPD05; sample-efficient | Yes (single GPU) | Likely yes (paper code) | Proxy only |
| **ReMaP** | DAC 2025 | Recursive prototyping + periphery-guided relocation; recovers expert-quality manual layouts | github.com/lamda-bbo/DAC25-ReMaP (mirror: xiaosinju/DAC25-ReMaP) | **+34.15% WNS / +65.39% TNS** vs OpenROAD baseline on 8 ORFS designs | Yes — runs on CPU + light GPU; was a contest entry | No model | **Yes** — full ORFS WNS/TNS/Power/DRC reported |
| **Re²MaP** | arXiv 2025-11 | ReMaP + packing-tree relocation; supersedes ReMaP on its own benchmarks | (paper-stage; code may not be released yet) | +22.22% WNS / +97.91% TNS over Hier-RTLMP on ORFS designs | Yes | No model | Yes |
| **IncreMacro** | ISPD 2024 (TCAD 2025) | Kd-tree macro diagnosis + gradient shift + LP-based legalization; designed as a refiner | (paper-only; cited as integrated into AutoDMP / DREAMPlace 4.0) | Improves AutoDMP / DREAMPlace 4.0 routability & timing | Yes (refiner) | No model | Yes — ORFS metrics |

## Top 3 techniques to actually try in 48 hours

The three picks below are ordered by **(realistic effort) × (expected proxy gain on our 17 IBM + 4 NG45 set)**, given our starting point (will_seed @ avg proxy 1.5336, RePlAce seed in hand, no time to train anything large).

### 1. WireMask-style refinement on top of the RePlAce seed (highest leverage)

**Why.** WireMask-BBO explicitly supports a "fine-tune existing placement" mode that reports up to 50% HPWL improvement when fed a non-trivial seed. Our seed *is* a non-trivial RePlAce placement. The wire-mask construction (compute a per-cell wire-length-delta heatmap; greedily relocate macros toward low-mask cells) is mostly a Python+NumPy/PyTorch routine — we can either install the repo or **port the core mask routine into our placer**, skipping the BBO outer loop (which costs hours per design and chases ISPD05 protocol).  
**Effort.** ~6 h to port the mask compute + a 1–2-sweep greedy relocator. ~3 h more if we want the BBO outer loop, but it's optional.  
**Expected gain.** The published 50% HPWL is on raw ISPD05 (no good seed). On top of a will-seed-class baseline we should still net **5–15% proxy reduction** because the mask captures WL contribution at fine grain — particularly when used as a *cost function inside SA* (replaces the current naive L1 sum).  
**Tier-2 caveat.** WireMask only reports HPWL. We'd want to retain ≥12 µm channel spacing as a hard constraint on top.

### 2. Klein-4 orientation + soft-macro refinement (lowest hanging fruit)

**Why.** Neither will_seed nor any of the surveyed RL/transformer methods explicitly address Klein-4 orientations — but the challenge requires us to *emit* orientations and forbids R90/R270/FE/FW. A simple after-the-fact pass that, for each hard macro, tries all four allowed orientations and keeps whichever lowers `compute_proxy_cost` is **directly aligned with the proxy** and trivially correct. CLAUDE.md §7 lists this as a high-leverage win (a few % per design). Combine with one pass of `plc.optimize_stdcells` to refresh soft macros after the hard-macro placement has moved.  
**Effort.** ~3 h orientation pass + ~2 h soft-FD wiring (the API call is documented).  
**Expected gain.** **2–6% proxy reduction**, larger on density-heavy designs (ibm14, ibm17, ibm18) where soft cells are crowding the same rows as macros.  
**Tier-2 caveat.** Orientation correctness *is* a Tier-2 hard constraint (fakeram SRAMs forbid 90° rotations). A pass that only ever emits N/FN/FS/S is the safest possible legalization.

### 3. Macro Regulator / LaMPlace as a *final-pass* surrogate ranker (hedging play)

**Why.** Macro Regulator and LaMPlace are both designed as refiners *over* an existing placement. Macro Regulator's RL policy and LaMPlace's polynomial cross-stage predictor are both small enough that their pretrained weights ship with the repos and can be evaluated in ≪1 minute per design. Concretely we'd:
- generate 4–8 candidate placements (RePlAce seed + WireMask refinement + a couple of multi-start SA seeds + soft-FD variants),
- score them with LaMPlace's cross-stage predictor (predicts WNS/TNS rather than only proxy),
- pick the candidate that's best on the *combined* (proxy, predicted-WNS, predicted-TNS).

LaMPlace's 43% WNS / 30.4% TNS gains are exactly the Tier-2 metric we will be ranked on if we make top-7. Even if its predictor doesn't transfer cleanly to NG45, using it as a *ranker* (relative ordering) is robust to absolute calibration error.  
**Effort.** ~5 h to install the LaMPlace inference path (PyTorch model, polynomial features, no training); +2 h to hook it as a selector over candidates. Skipping this if it doesn't load cleanly costs us nothing — we still have the WireMask + K4 + soft-FD stack.  
**Expected gain.** Hard to quantify on Tier 1; the value is **Tier-2 hedging** (likely 5–15% relative improvement on the Grand-Prize geo-mean). For Tier 1 proxy: small (≤2%).

## Methods we explicitly de-prioritize

- **ChiPFormer / Macro Regulator from scratch / EfficientPlace training**: any approach that requires more than minutes of fine-tune time is out (we don't have the trajectory bank, and our benchmarks are IBM ICCAD04 + NG45 — different from ISPD05). Pretrained-weights inference *is* on the table (item 3 above).
- **AutoDMP MOBO outer loop**: each MOBO iteration is a DREAMPlace run; budget cap at 1 h/benchmark with ~12 iters per design is feasible but install is complex (custom CUDA ops in DREAMPlace; CUDA 12.4 / cuDNN 9 / PyTorch 2.5.1 alignment is fragile). Worth exploring only if items 1–2 land early.
- **Hier-RTLMP from OpenROAD**: production CPU placer in C++. Installing OpenROAD inside our Docker image is non-trivial. Skip — DREAMPlace already covers the analytical-baseline slot.
- **ReMaP / Re²MaP**: highest reported Tier-2 gains, but the DAC'25 code is a full-flow refiner that consumes ORFS-style inputs. Adapter cost is high; ReMaP is the *judges'* benchmark to beat in Tier 2, not necessarily what we run.

## Open questions worth a follow-up search if time allows

- Verify EGPlace code release URL (paper from ICML 2025; likely at `lamda-bbo` org).
- Find ChiPFormer pretrained checkpoint that has been retargeted to ICCAD04-style benchmarks (their default save_models is ISPD05).
- Confirm whether DREAMPlace 4.1 builds inside the default `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` image; their build instructions historically use CUDA 11.x.

## Sources

- [DREAMPlace (limbo018)](https://github.com/limbo018/DREAMPlace) — analytical placer w/ macro flow
- [AutoDMP (NVlabs)](https://github.com/NVlabs/AutoDMP) — DREAMPlace + MOBO
- [AutoDMP NVIDIA blog](https://developer.nvidia.com/blog/autodmp-optimizes-macro-placement-for-chip-design-with-ai-and-gpus/) — engineering details
- [ChiPFormer (laiyao1)](https://github.com/laiyao1/ChiPFormer) — offline DT; pretrained ckpts
- [ChiPFormer ICML'23 paper](https://arxiv.org/abs/2306.14744)
- [Macro Regulator (lamda-bbo)](https://github.com/lamda-bbo/macro-regulator) — NeurIPS'24 refiner; arxiv [2412.07167](https://arxiv.org/abs/2412.07167)
- [LaMPlace (MIRALab-USTC)](https://github.com/MIRALab-USTC/AI4EDA-LaMPlace) — ICLR'25 cross-stage predictor; [OpenReview](https://openreview.net/forum?id=YLIsIzC74j)
- [WireMask-BBO (lamda-bbo)](https://github.com/lamda-bbo/WireMask-BBO) — NeurIPS'23 wire-mask + BBO; arxiv [2306.16844](https://arxiv.org/abs/2306.16844)
- [EGPlace ICML'25 paper](https://proceedings.mlr.press/v267/deng25g.html) — evolutionary + greedy mutation
- [ReMaP (DAC'25)](https://github.com/lamda-bbo/DAC25-ReMaP) — recursive prototyping + relocation
- [Re²MaP arxiv 2025-11](https://arxiv.org/abs/2511.08054) — packing-tree relocation
- [IncreMacro TCAD'25 paper](https://www.cse.cuhk.edu.hk/~byu/papers/J137-TCAD2025-IncreMacro.pdf) — kd-tree + LP refinement
- [Hier-RTLMP in OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD/tree/master/src/mpl) — production hierarchical placer
- [TILOS MacroPlacement evaluator](https://github.com/TILOS-AI-Institute/MacroPlacement) — the proxy-cost engine used by the challenge
- [Partcl challenge repo](https://github.com/partcleda/macro-place-challenge-2026) — rules + leaderboard
