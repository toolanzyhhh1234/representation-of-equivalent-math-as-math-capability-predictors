"""MELD loading and stimulus assembly.

MELD (uw-math-ai/MELD-dataset): 270 positive pairs across 9 domains / 18 framings,
plus 541 framing-matched hard negatives (lexically close, mathematically wrong).
The distractors are the sec 4.1 control -- without them EQ measures topical clustering.
"""
import json
from dataclasses import dataclass
from .config import DATA


@dataclass(frozen=True)
class Stimuli:
    texts: list          # every unique statement, deduped, stable order
    index: dict          # text -> row in the activation matrix
    pairs: list          # (idx_a, idx_b, framing_a, framing_b, domain, pair_id)
    distractors: dict    # framing -> [row indices]


def load_meld() -> Stimuli:
    raw = json.loads((DATA / "adversarial_theorem_pairs_2.json").read_text())
    dis = json.loads((DATA / "distractors_all.json").read_text())

    texts, index = [], {}

    def add(t):
        t = t.strip()
        if t not in index:
            index[t] = len(texts)
            texts.append(t)
        return index[t]

    pairs = []
    for p in raw["pairs"]:
        ia = add(p["entry_1"]["statement"])
        ib = add(p["entry_2"]["statement"])
        pairs.append((ia, ib, p["entry_1"]["framing"], p["entry_2"]["framing"],
                      p["domain"], p["id"]))

    distractors = {f: [add(s) for s in ss] for f, ss in dis.items()}

    # A distractor that is textually identical to a real statement would be a silent
    # label error: it would appear as both positive and negative. Check, do not assume.
    dset = {i for v in distractors.values() for i in v}
    pset = {i for pr in pairs for i in pr[:2]}
    overlap = dset & pset
    if overlap:
        raise ValueError(f"{len(overlap)} statements appear as both positive and distractor")

    return Stimuli(texts=texts, index=index, pairs=pairs, distractors=distractors)
