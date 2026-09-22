# Agent notes for this repo

Read `README.md` (status + doc map) and `REPORT.md` (the current write-up) first.
This file records the pins and traps that are easy to get wrong when resuming work
on a fresh machine.

## Non-negotiable pins

1. **TF-IDF lexical null**: `char_wb`, ngram (3,5) — pinned in `src/lexical.py`.
   Plain `char` gives 0.7608, not the documented 0.7715.
2. **Layer 0 is excluded from headline layer selection** (`recs[1:]` in
   `src/run_panel.py` and `src/specificity.py`) — it is the token-identity control
   layer, and deepseek-coder leaks lexical signal there that TF-IDF can't
   residualize. Keep it that way.
3. **bf16 hidden states differ across GPU generations in the 3rd decimal.**
   Never mix EQ numbers extracted on different machines — re-extract the whole
   panel on one box, or use the fp32 caches from the HF archive as-is.
   (Phases 0.5–3 ran on a single RTX 5090; Phase 0 on an RTX 3070.)
4. **Parameter counts come from weights** (`results/raw/*.params`, measured at
   load), never from model cards (Qwen3-0.6B is really 0.752B).
5. **Eval harnesses are pinned** in `src/eval_*.py` and run in-house for every
   model. Never import published capability numbers into an analysis.
6. `results/panel.json` entries carry `n_params` only after a `run_panel` pass
   over the cached npz — if you add models, run `run_panel` for them before
   `analyze`.

## Raw-data archive

Everything regenerates from
[`toolazyhhh123/representation-of-equivalent-math-raw`](https://huggingface.co/datasets/toolazyhhh123/representation-of-equivalent-math-raw)
(fp32 activation caches for MELD+PAWS, per-item eval outputs, aggregate results)
plus the pinned code here. Activations also cache locally to `results/raw/`
(gitignored).

## Agreed framing (user decision — respect it in any REPORT edit)

- The **correlation** (ρ(EQ_resid, MATH-500) with size/lexical/general-quality
  controls) is the confirmatory tier, stated with confidence.
- **Necessity-without-sufficiency** is a generative hypothesis: one hedged
  paragraph plus a standing falsifiable prediction — not a proven claim. The
  wedge is also what a monotone relation plus a floor-bounded DV produces; the
  0.65 threshold is descriptive on 23 points, not fitted.

## Next steps (agreed order)

**Latest user decision, 2026-09-22:** prioritize a direct test of whether better
hidden-state geometry accompanies better independent reasoning performance, within
or across models. See **[NEXT_STEPS.md](NEXT_STEPS.md)** for the next-container
handoff. Geometry is primary; supervised probes are auxiliary. OOD probe degradation
can reflect probe transfer failure, worse model task performance, or both, so it
cannot establish geometric degradation alone. Define the metric and evaluation
design next; no consequence-geometry metric or checkpoint family is selected yet.
This priority supersedes the earlier roadmap below.

**2026-09-22 handoff update:** H2 now has completed pilot/follow-up runs and pinned HF
snapshots; see `ARCHIVES.md`, `H2_RUNNING.md`, and `results/H2_FOLLOWUP_READOUT.md`.
The user then requested the logical-consequence investigation. That first pilot is
in `LOGIC_RUNNING.md` and `results/LOGIC_READOUT.md`: limited directional transfer,
and a post-hoc syntax rule perfectly predicts the auxiliary premise-role labels.
Do not present those role scores as semantic indispensability evidence. The next
logic version needs counterfactual remote-proof changes and matched surface features.
All frozen experiment files are hash checked: create new versions rather than edit
the scientific code/data and silently reuse old caches. Earlier result files remain
historical; the list below records the pre-H2 roadmap, not unfinished completed work.

1. **H2 invariance arm** — rewrite-invariance DV (PLAN.md §5, GSM-Symbolic-style);
   two-sided advance predictions already registered in REPORT §4.8–4.9.
2. H3 causal arm — ablate the equivalence subspace (~10 dims expected; PLAN §7).
3. Instrument hardening: fitted necessity boundary with uncertainty, frozen
   sentence encoder as stronger partialled regressor, topic-matched negatives,
   better language-side mirror than PAWS.
4. Contact tutor with the repo link (roadmap feedback, arXiv endorsement).
