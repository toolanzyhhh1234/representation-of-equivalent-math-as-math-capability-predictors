"""Post-hoc syntax-shortcut diagnostic; never changes registered probe results."""
from collections import defaultdict
import json
import re

from .logic_data import DEST
from .run_logic import OUT


def main():
    b=json.loads(DEST.read_text());counts=defaultdict(lambda:[0,0])
    for r in b['items']:
        if r['task']!='necessity':continue
        candidate=re.search(r'Candidate fact: (p\d+)\.',r['prompt']).group(1)
        rules=r['prompt'].split('Facts:')[0]
        if r['style']=='english':
            conjunction=f'and {candidate} is true' in rules
            consequent=f'then {candidate} is true' in rules
        else:
            conjunction=f'& {candidate} ->' in rules
            consequent=f'-> {candidate}.' in rules
        pred=0 if conjunction else (2 if consequent else 1)
        c=counts[r['split']+'/'+r['style']];c[0]+=pred==r['label'];c[1]+=1
    out={'type':'post-hoc construct-validity diagnostic, not the registered primary probe',
         'rule':'indispensable if candidate occurs after and/&; else irrelevant if a rule consequent; else redundant',
         'uses':'prompt text only; neither proof search nor model states nor item IDs',
         'scores':{k:{'correct':a,'total':n,'accuracy':a/n} for k,(a,n) in counts.items()}}
    (OUT/'premise_shortcut_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
