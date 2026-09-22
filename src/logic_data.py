"""Finite positive Horn theories with independently checked entailment labels."""
from itertools import combinations
import json

import numpy as np

from .config import ROOT
from .h2_data import digest

DEST = ROOT / "data/logic/pilot.json"
SEED = 20260924


def closure(rules, facts):
    known = set(facts)
    while True:
        expanded = known | {q for premises, q in rules if set(premises) <= known}
        if expanded == known: return known
        known = expanded


def truth_worlds(n, rules):
    """Independent exhaustive model checker, not another call to forward chaining."""
    worlds = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)
    valid = np.ones(len(worlds), dtype=bool)
    for premises, q in rules:
        valid &= ~worlds[:, list(premises)].all(axis=1) | worlds[:, q]
    return worlds[valid]


def entailment(worlds, facts, q):
    compatible = worlds[worlds[:, list(facts)].all(axis=1)] if facts else worlds
    if not len(compatible): raise ValueError("Inconsistent theory; explosion excluded")
    witnesses = compatible[~compatible[:, q]]
    return len(witnesses) == 0, witnesses[0].astype(int).tolist() if len(witnesses) else None


def minimal_supports(rules, facts, q):
    supports = []
    for k in range(len(facts)+1):
        for subset in combinations(facts, k):
            if any(set(s) <= set(subset) for s in supports): continue
            if q in closure(rules, subset): supports.append(list(subset))
    return supports


def graph(family, depth):
    if family == "diamond":
        u = 2; next_node = 3; paths = []
        for _ in range(2):
            path = list(range(next_node, next_node+depth-1)); next_node += depth-1
            paths.append(path)
        v = next_node; next_node += 1
        rules = [([0], u), ([1], u)]
        for path in paths:
            seq = [u]+path+[v]
            rules += [([a], b) for a, b in zip(seq, seq[1:])]
        rules += [([v], next_node), ([v], next_node+1)]
        n = next_node+2; facts = []
    else:
        n = depth+3; u, v = 1, depth+1
        rules = [([i], i+1) for i in range(n-1)]; facts = []
        if family == "conjunction":
            guard = n; n += 1; facts = [guard]
            rules[1] = ([1, guard], 2)
    z, w = n, n+1; n += 2
    rules += [([z], w), ([w], z)]  # disconnected component; unrelated is not contradiction
    return n, rules, facts, u, v, z


def necessity_graph(depth, early_gate=False):
    # Two alternative supporting facts u/a, an indispensable guard g, and irrelevant z.
    u, a, g, z = 0, 1, 2, 3
    chain = list(range(4, 4+depth)); q = 4+depth; n = q+3
    if early_gate:
        rules = [([u, g], chain[0]), ([a, g], chain[0])]
        rules += [([x], y) for x, y in zip(chain, chain[1:]+[q])]
    else:
        rules = [([u], chain[0]), ([a], chain[0])]
        rules += [([x], y) for x, y in zip(chain, chain[1:])]
        rules += [([chain[-1], g], q)]
    rules += [([z], q+1), ([q+2], z)]
    return n, rules, [u, a, g, z], q, [g, u, z]


def render(rules, facts, names, style):
    if style == "english":
        lines = ["If " + " and ".join(f"{names[p]} is true" for p in ps) + f", then {names[q]} is true."
                 for ps, q in rules]
        fs = " ".join(f"{names[p]} is true." for p in facts) or "No facts are given."
    else:
        lines = [" & ".join(names[p] for p in ps) + " -> " + names[q] + "." for ps, q in rules]
        fs = ", ".join(names[p] for p in facts) or "none"
    return "Rules:\n"+"\n".join(lines)+"\nFacts: "+fs+"\n"


def build():
    rng = np.random.default_rng(SEED)
    splits = [("train",120,0),("dev",30,200),("test_iid",50,300),
              ("test_depth",50,400),("test_structure",50,500)]
    theories, items = [], []
    for split, count, start in splits:
        for local in range(count):
            tid = f"{split}-{local:03d}"
            family = ("diamond" if local % 2 else "conjunction") if split == "test_structure" else "chain"
            depth = (4+local%2) if split == "test_depth" else (2+local%2 if family == "diamond" else 1+local%3)
            n, rules, facts, p, q, z = graph(family, depth)
            rng.shuffle(rules)
            names = [f"p{i:03d}" for i in rng.choice(np.arange(start, start+100), n, replace=False)]
            worlds = truth_worlds(n, rules)
            queries = [("forward",p,q,1),("reverse",q,p,0),("unrelated_forward",p,z,0),("unrelated_reverse",z,p,0)]
            checked = []
            for kind, source, target, label in queries:
                fc = target in closure(rules, facts+[source])
                truth, witness = entailment(worlds, facts+[source], target)
                if fc != truth or int(truth) != label: raise ValueError("Solver disagreement or invalid direction")
                checked.append({"kind":kind,"source":source,"target":target,"label":label,"countermodel":witness})
            nn, nrules, nfacts, nq, candidates = necessity_graph(depth, split == "test_structure")
            rng.shuffle(nrules); rng.shuffle(nfacts)
            nnames = [f"p{i:03d}" for i in rng.choice(np.arange(start,start+100),nn,replace=False)]
            nworlds = truth_worlds(nn,nrules); supports = minimal_supports(nrules,nfacts,nq)
            if not entailment(nworlds,nfacts,nq)[0]: raise ValueError("Necessity query must initially follow")
            needed = []
            for expected, candidate in enumerate(candidates):
                reduced = [f for f in nfacts if f != candidate]
                after, witness = entailment(nworlds,reduced,nq)
                if after != (nq in closure(nrules,reduced)): raise ValueError("Deletion solver disagreement")
                role = 0 if not after else (1 if any(candidate in s for s in supports) else 2)
                if role != expected: raise ValueError("Incorrect premise role")
                needed.append({"candidate":candidate,"label":role,"countermodel_after_deletion":witness})
            theories.append({"id":tid,"split":split,"family":family,"depth":depth,"n":n,"rules":rules,
                             "facts":facts,"names":names,"queries":checked,
                             "necessity":{"n":nn,"rules":nrules,"facts":nfacts,"names":nnames,
                                          "query":nq,"minimal_supports":supports,"candidates":needed}})
            for style in ("english","symbolic"):
                context = render(rules,facts,names,style)
                for node in (p,q,z):
                    items.append({"id":f"{tid}/{style}/node/{node}","theory":tid,"split":split,"style":style,
                                  "task":"node","prompt":context+f"Statement: {names[node]}.\nReadout:"})
                for qr in checked:
                    source, target = qr["source"], qr["target"]
                    prompt = context+f"Additional premise: {names[source]}.\nConclusion: {names[target]}.\n"
                    prompt += "Does the conclusion necessarily follow? A = yes; B = no.\nAnswer:"
                    items.append({"id":f"{tid}/{style}/{qr['kind']}","theory":tid,"split":split,"style":style,
                                  "task":"direction" if qr["kind"] in ("forward","reverse") else "unrelated",
                                  "label":qr["label"],"source_node":f"{tid}/{style}/node/{source}",
                                  "target_node":f"{tid}/{style}/node/{target}","prompt":prompt})
                context = render(nrules,nfacts,nnames,style)
                for nr in needed:
                    prompt = context+f"Conclusion: {nnames[nq]}.\nCandidate fact: {nnames[nr['candidate']]}.\n"
                    prompt += ("Classify the candidate for deriving the conclusion. "
                               "A = indispensable (removing it prevents derivation); "
                               "B = redundant (used in some minimal proof, but an alternative proof omits it); "
                               "C = irrelevant (used in no minimal proof).\nAnswer:")
                    items.append({"id":f"{tid}/{style}/necessity/{nr['label']}","theory":tid,"split":split,
                                  "style":style,"task":"necessity","label":nr["label"],"prompt":prompt})
    bundle = {"schema":1,"seed":SEED,"theories":theories,"items":items,
              "semantics":"classical finite positive Horn implications; no negation; not-entailed does not mean false"}
    bundle["sha256"] = digest(bundle)
    DEST.parent.mkdir(parents=True,exist_ok=True)
    if DEST.exists() and digest(json.loads(DEST.read_text())) != digest(bundle):
        raise ValueError("Refusing changed logic dataset")
    DEST.write_text(json.dumps(bundle,indent=2)+"\n")
    print(f"Verified {len(theories)} theories, {len(items)} prompts with two independent solvers; {bundle['sha256']}")
    return bundle


if __name__ == "__main__": build()
