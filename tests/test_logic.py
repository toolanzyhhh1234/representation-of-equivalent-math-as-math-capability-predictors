"""Logical labels, grouping and shortcut controls must hold before model extraction."""
import json
import unittest

import numpy as np

from src.logic_data import DEST,closure,entailment,graph,minimal_supports,necessity_graph,truth_worlds
from src.h2_data import digest


class LogicTests(unittest.TestCase):
    def test_consequence_is_directional(self):
        rules=[([0],1),([1],2)]
        w=truth_worlds(3,rules)
        self.assertTrue(entailment(w,[0],2)[0])
        yes,counter=entailment(w,[2],0)
        self.assertFalse(yes)
        self.assertEqual(counter[2],1);self.assertEqual(counter[0],0)

    def test_equivalent_reversals_are_not_negative(self):
        w=truth_worlds(2,[([0],1),([1],0)])
        self.assertTrue(entailment(w,[0],1)[0]);self.assertTrue(entailment(w,[1],0)[0])

    def test_conjunction_needs_both_facts(self):
        rules=[([0,1],2)];w=truth_worlds(3,rules)
        self.assertFalse(entailment(w,[0],2)[0]);self.assertTrue(entailment(w,[0,1],2)[0])

    def test_redundant_is_not_indispensable(self):
        for early in (False,True):
            n,r,f,q,c=necessity_graph(3,early)
            supports=minimal_supports(r,f,q)
            self.assertEqual({frozenset(s) for s in supports},{frozenset([0,2]),frozenset([1,2])})
            self.assertNotIn(q,closure(r,[x for x in f if x!=c[0]]))
            self.assertIn(q,closure(r,[x for x in f if x!=c[1]]))
            self.assertFalse(any(c[2] in s for s in supports))

    def test_solvers_agree_on_random_horn_systems(self):
        rng=np.random.default_rng(7)
        for _ in range(40):
            r=[(rng.choice(6,size=int(rng.integers(1,3)),replace=False).tolist(),int(rng.integers(6))) for _ in range(8)]
            facts=rng.choice(6,size=2,replace=False).tolist();w=truth_worlds(6,r)
            for q in range(6):self.assertEqual(q in closure(r,facts),entailment(w,facts,q)[0])

    def test_direction_nodes_have_equal_local_degrees(self):
        for family in ('chain','conjunction','diamond'):
            for depth in (2,3):
                n,r,f,u,v,z=graph(family,depth)
                degrees=lambda p:(sum(q==p for _,q in r),sum(p in ps for ps,_ in r))
                self.assertEqual(degrees(u),degrees(v))

    def test_dataset_identity_and_split_balance(self):
        b=json.loads(DEST.read_text())
        self.assertEqual(b['sha256'],digest({k:v for k,v in b.items() if k!='sha256'}))
        self.assertEqual(len(b['theories']),300);self.assertEqual(len(b['items']),6000)
        byid={r['id']:r for r in b['items']};vocab={}
        for t in b['theories']:
            vocab.setdefault(t['split'],set()).update(t['names'])
            for style in ('english','symbolic'):
                a=byid[f"{t['id']}/{style}/forward"];c=byid[f"{t['id']}/{style}/reverse"]
                self.assertEqual(a['source_node'],c['target_node']);self.assertEqual(a['target_node'],c['source_node'])
                self.assertEqual(a['label']+c['label'],1)
                self.assertEqual(len(a['prompt']),len(c['prompt']))
        for a,va in vocab.items():
            for c,vc in vocab.items():
                if a!=c:self.assertFalse(va & vc)

    def test_json_roundtrip_does_not_change_rule_identity(self):
        x={'rules':[([0,1],2)]}
        self.assertEqual(digest(x),digest(json.loads(json.dumps(x))))


if __name__=='__main__':unittest.main()
