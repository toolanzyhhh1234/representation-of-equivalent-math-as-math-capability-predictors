---
license: apache-2.0
language:
- en
task_categories:
- sentence-similarity
- feature-extraction
- text-retrieval
tags:
- mathematics
- mathlib
- theorem-embedding
- retrieval
- contrastive-learning
- hard-negatives
- adversarial
- benchmark
- evaluation
size_categories:
- n<1K
pretty_name: "MELD: Mathematical Equivalence under Linguistic Diversity"
configs:
- config_name: theorem_pairs
  data_files:
  - split: test
    path: adversarial_theorem_pairs_2.json
- config_name: distractors
  data_files:
  - split: test
    path: distractors_all.json
---

# MELD — Mathematical Equivalence under Linguistic Diversity

<!-- TODO(confirm): the expansion of "MELD" above is a placeholder. Replace it with the
     lab's official expansion before publishing. -->

**MELD** is a small, hand-curated **evaluation benchmark** for math-aware text embedding
models. It tests one specific capability: does the model recognize that two statements
describing the **same** mathematical fact are equivalent **even when they are written in
the vocabulary, notation, and conventions of different mathematical subfields**?

MELD was originally part of
[`uw-math-ai/Math2Vec-embedding-dataset`](https://huggingface.co/datasets/uw-math-ai/Math2Vec-embedding-dataset)
and is released here as a standalone benchmark. Unlike the main Math2Vec corpus (a large
training set of Mathlib theorems with informalizations and mined hard negatives), MELD is
an **adversarial, evaluation-only** set: it is deliberately small, balanced across domains,
and constructed so that surface-level lexical overlap is a *misleading* signal.

## Why "adversarial"?

Each positive pair restates one concept twice, under two different **framings** (the dialect
of a particular subfield). For example, "a subset spans a vector space" and "the smallest
submodule containing a set is the whole module" are the *same* idea phrased in
*vector-space* language vs. *module-theory* language. A model that keys on shared tokens
will be fooled, because:

- **Positives look different.** The two members of a true pair share the underlying meaning
  but often share very few surface tokens.
- **Distractors look similar.** The hard-negative pool contains statements that are
  lexically close to true statements but mathematically wrong (e.g. a sign flipped, a
  quantifier weakened, `injective` swapped for `surjective`, `closed` for `open`).

A strong math embedder should place the two framings of a pair close together while keeping
the distractors far away.

## Dataset structure

The benchmark ships as two JSON files.

### 1. `adversarial_theorem_pairs_2.json`

A JSON object with two top-level keys:

```jsonc
{
  "pairs":       [ /* 270 pair objects */ ],
  "distractors": { /* framing -> list[str]; identical to distractors_all.json */ }
}
```

Each element of `pairs` has the schema:

| field      | type   | description                                                        |
| ---------- | ------ | ------------------------------------------------------------------ |
| `id`       | int    | Unique pair id, 1–270.                                             |
| `domain`   | string | Source domain (one of 9, see below).                              |
| `topic`    | string | The shared concept the two entries express (e.g. "Spanning / Generation"). |
| `entry_1`  | object | `{ "framing": str, "statement": str }`                            |
| `entry_2`  | object | `{ "framing": str, "statement": str }`                            |

`entry_1` and `entry_2` are the two **equivalent** restatements. `framing` names the
mathematical dialect; `statement` is the LaTeX-formatted statement in that dialect.

Example:

```json
{
  "id": 1,
  "domain": "algebra",
  "topic": "Spanning / Generation",
  "entry_1": {
    "framing": "vector spaces",
    "statement": "A subset $S \\subseteq V$ \\textbf{spans} $V$ if every $v \\in V$ can be expressed as a finite sum $v = \\sum_i \\lambda_i s_i$ with $\\lambda_i \\in F$, $s_i \\in S$."
  },
  "entry_2": {
    "framing": "module theory",
    "statement": "Assume $R$ is a field and $M$ is a left $R$-module. The smallest $R$-submodule of $M$ containing $S$ is all of $M$ precisely when no proper $R$-submodule contains $S$."
  }
}
```

### 2. `distractors_all.json`

A JSON object mapping each **framing** to a list of distractor statements (hard negatives):

```jsonc
{
  "vector spaces": [ "A linear map $T : V \\to W$ is \\textbf{injective} if and only if $\\ker T = \\{0\\}$.", ... ],
  "module theory": [ ... ],
  ...
}
```

These are *unpaired* statements meant to populate the candidate pool during retrieval /
ranking evaluation. The same dictionary is also embedded as the `distractors` field of
`adversarial_theorem_pairs_2.json`, so you can use either file as the source of truth.

## Statistics

| Quantity                         | Value |
| -------------------------------- | ----- |
| Positive pairs                   | 270   |
| Domains                          | 9 (30 pairs each) |
| Distinct framings                | 18    |
| Pairs per framing                | 30    |
| Distractor statements (total)    | 541   |
| Framings in distractor pool      | 18 (~30 each) |
| Language                         | English (LaTeX math) |

**Domains** (9): `algebra`, `probability`, `foundations`, `algebraic_geometry`,
`algebraic_topology`, `spectral_graph_theory`, `discrete_math`, `representation_theory`,
`algebraic_combinatorics`.

**Framings** (18): vector spaces, module theory, probability, measure theory, set theory,
category theory, geometry, commutative algebra, topology, algebra, graph theory, linear
algebra, discrete math, complex analysis, representation theory, Fourier analysis,
symmetric functions, tableaux.

## Loading

The files are nested JSON objects (not one-record-per-line), so the simplest and most
reliable path is to read them directly:

```python
import json
from huggingface_hub import hf_hub_download

repo = "uw-math-ai/MELD-dataset"  

pairs_path = hf_hub_download(repo, "adversarial_theorem_pairs_2.json", repo_type="dataset")
distr_path = hf_hub_download(repo, "distractors_all.json",            repo_type="dataset")

data        = json.load(open(pairs_path))
pairs       = data["pairs"]            # list of 270 pair dicts
distractors = json.load(open(distr_path))  # framing -> list[str]

print(len(pairs), "pairs")
print(sum(len(v) for v in distractors.values()), "distractors")
```

## Suggested evaluation protocols

MELD is format-agnostic; a few natural ways to use it:

- **Paired similarity / AUC.** For each pair, embed `entry_1` and `entry_2`; a true pair
  should score higher than a (`entry_1`, distractor) mismatch. Report ROC-AUC or accuracy
  of "is this a true equivalence?".
- **Retrieval.** Treat `entry_1` as the query and build a candidate pool from all
  `entry_2` statements plus the distractor pool; report Recall@k / MRR for retrieving the
  matching `entry_2`. Drawing distractors from the **same framing** as the target yields
  the hardest setting.
- **Contrastive / triplet evaluation.** Use (`entry_1`, `entry_2`, distractor) triplets to
  measure margin between positives and hard negatives.

## Dataset construction

The positive pairs are written so that one mathematical concept appears in two distinct
subfield framings, holding the meaning fixed while varying notation and terminology. The
distractor pool consists of statements that are superficially close to correct statements
in a given framing but altered to be mathematically false. The set is small and balanced by
design (9 domains × 30 pairs) so that it functions as a clean diagnostic rather than a
training corpus.

<!-- TODO(confirm): add details the lab wants on record — generation method (model used,
     prompting, human review), and whether statements derive from / are inspired by Mathlib. -->

## Limitations and intended use

- **Evaluation only.** With 270 pairs MELD is far too small for training; it is a
  diagnostic benchmark.
- **English + LaTeX.** Statements are natural-language math with LaTeX; there are no Lean
  or other formal representations in this set.
- **Synthetic / adversarial.** Statements are curated to stress a specific failure mode and
  are not a representative sample of real-world theorem text.
- **Distractor duplication.** The distractor dictionary appears both as its own file and
  embedded inside the pairs file; keep them in sync if you edit either.

## Licensing

Released under the Apache-2.0 license.

## Acknowledgements

MELD originates from the
[Math2Vec](https://huggingface.co/datasets/uw-math-ai/Math2Vec-embedding-dataset)
project of the [UW Math AI Lab](https://ai.math.uw.edu).