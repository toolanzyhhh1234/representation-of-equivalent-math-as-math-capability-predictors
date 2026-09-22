# Logical-consequence representation pilot

**Exploratory decoding study, not causal evidence of a reasoning mechanism.**
Ground truth was checked by forward chaining and exhaustive Boolean model enumeration.
300 theory groups; English training/development only; all reversals, paraphrases and premise-role cases remain grouped.
There are no gold answers or proof traces in the encoded prompts. Not-entailed is not the same as false.
Protocol: `2da216b0990aa692ff6526ef2e68c83ad35b92f0b509536d64883c457b8eafbd`. [Design](../prereg/LOGIC_PILOT.md).

## Primary: directed consequence (AUROC)

| Model | Layer | IID English | Depth 4–5 | New structures | IID symbolic |
|---|---:|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 27 | 0.580 | 0.501 | 0.492 | 0.545 |
| Qwen/Qwen2.5-0.5B | 2 | 0.486 | 0.466 | 0.508 | 0.477 |
| microsoft/phi-1_5 | 10 | 0.498 | 0.519 | 0.590 | 0.516 |
| Qwen/Qwen2.5-1.5B | 19 | 0.645 | 0.559 | 0.563 | 0.517 |
| Qwen/Qwen2.5-Math-1.5B | 28 | 0.509 | 0.453 | 0.494 | 0.520 |

## Controls and transfer diagnostics (IID English unless specified)

| Model | Layer 0 AUC | Random-label AUC | Ordered-node AUC | Symmetric cosine AUC | Unrelated false positives | Native answer AUC |
|---|---:|---:|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 0.500 | 0.492 | 0.616 | 0.500 | 0.700 | 0.520 |
| Qwen/Qwen2.5-0.5B | 0.500 | 0.493 | 0.641 | 0.500 | 0.190 | 0.423 |
| microsoft/phi-1_5 | 0.500 | 0.500 | 0.439 | 0.500 | 0.500 | 0.488 |
| Qwen/Qwen2.5-1.5B | 0.500 | 0.434 | 0.787 | 0.500 | 0.320 | 0.505 |
| Qwen/Qwen2.5-Math-1.5B | 0.500 | 0.424 | 0.733 | 0.500 | 0.490 | 0.466 |

### Lexical/position baselines

| Baseline | IID AUC | Depth AUC | Structure AUC |
|---|---:|---:|---:|
| char | 0.500 | 0.500 | 0.500 |
| word_bigrams | 0.500 | 0.500 | 0.500 |
| positions | 0.596 | 0.560 | 0.529 |

## Secondary: premise roles (balanced accuracy; chance 1/3)

| Model | Layer | IID | Depth | Early-gate structure | Native IID |
|---|---:|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 23 | 0.587 | 0.433 | 0.407 | 0.333 |
| Qwen/Qwen2.5-0.5B | 24 | 0.540 | 0.433 | 0.320 | 0.333 |
| microsoft/phi-1_5 | 14 | 0.547 | 0.473 | 0.533 | 0.333 |
| Qwen/Qwen2.5-1.5B | 22 | 0.760 | 0.787 | 0.767 | 0.327 |
| Qwen/Qwen2.5-Math-1.5B | 20 | 0.713 | 0.547 | 0.487 | 0.333 |

## Limits

Layers are selected on development data only, excluding layer 0. Probes have fixed C=1 and
a fixed random linear projection to 128 dimensions, with scaling fit only on training rows.
These are low-capacity supervised readouts, not proof of native ordered geometry or causal use.
Symmetric cosine cannot distinguish reversed pairs and is a construction-level control.
Ordered-node probes can recover a ranking without representing every non-entailment; unrelated-query
false positives must be considered. Necessity templates may expose structural shortcuts; the early-gate
holdout tests one such shift. This first pilot contains positive Horn logic, not negation or unrestricted FOL.
All detailed test partitions, controls, layer-selection curves, behavior and cache provenance are in
`results/logic/pilot_v1/analysis.json` and the per-model files. No cross-model capability correlation
or mechanistic claim is made from five models and this small synthetic grammar.
