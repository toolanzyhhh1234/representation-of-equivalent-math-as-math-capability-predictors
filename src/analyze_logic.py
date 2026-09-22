"""Train probes on training theories, select layers on development, evaluate once."""
import hashlib
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from threadpoolctl import threadpool_limits

from .config import ROOT
from .run_h2 import atomic_json
from .run_logic import OUT, inputs

SEED=20260924


def probe(d):
    return make_pipeline(StandardScaler(),GaussianRandomProjection(n_components=min(128,d),random_state=SEED),
                         StandardScaler(),LogisticRegression(C=1.0,max_iter=2000,solver='lbfgs'))


def save_probe(path,clf,X):
    s1,projection,s2,linear=[step[1] for step in clf.steps]
    np.savez_compressed(path,input_mean=s1.mean_,input_scale=s1.scale_,projection=projection.components_,
                        projected_mean=s2.mean_,projected_scale=s2.scale_,coef=linear.coef_,
                        intercept=linear.intercept_,classes=linear.classes_,probabilities=clf.predict_proba(X))


def metrics(y,prob):
    return {'balanced_accuracy':float(balanced_accuracy_score(y,np.argmax(prob,axis=1))),
            'auroc':float(roc_auc_score(y,prob[:,1])) if len(np.unique(y))==2 else
                    float(roc_auc_score(y,prob,multi_class='ovr'))}


def masks(items,task,style='english',split='train'):
    return np.array([r['task']==task and r['style']==style and r['split']==split for r in items])


def evaluate(clf,X,items,y,task):
    out={}
    for split in ('test_iid','test_depth','test_structure'):
        for style in ('english','symbolic'):
            mask=masks(items,task,style,split)
            out[split+'/'+style]=metrics(y[mask],clf.predict_proba(X[mask]))
            if task=='direction':
                unrelated=masks(items,'unrelated',style,split)
                out[split+'/'+style]['unrelated_false_positive_rate']=float((clf.predict(X[unrelated])==1).mean())
    return out


def position_features(b):
    theories={t['id']:t for t in b['theories']};out=[]
    for r in b['items']:
        if r['task'] not in ('direction','unrelated'):out.append([0]*6);continue
        context=r['prompt'].split('Additional premise:')[0];t=theories[r['theory']]
        p=t['names'][int(r['source_node'].rsplit('/',1)[1])];q=t['names'][int(r['target_node'].rsplit('/',1)[1])]
        out.append([context.count(p),context.count(q),context.find(p)/len(context),context.find(q)/len(context),
                    context.rfind(p)/len(context),context.rfind(q)/len(context)])
    return np.asarray(out,float)


def analyze():
    p,b=inputs();items=b['items'];index={r['id']:i for i,r in enumerate(items)}
    y=np.array([r.get('label',-1) for r in items]);train=masks(items,'direction');dev=masks(items,'direction',split='dev')
    texts=[r['prompt'] for r in items]
    lexical={}
    for name,vec in [('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5))),
                     ('word_bigrams',TfidfVectorizer(ngram_range=(1,2)))]:
        X=vec.fit_transform([texts[i] for i in np.flatnonzero(train)])
        clf=LogisticRegression(C=1,max_iter=2000).fit(X,y[train])
        lexical[name]=evaluate(clf,vec.transform(texts),items,y,'direction')
    X=position_features(b);clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000)).fit(X[train],y[train])
    lexical['positions']=evaluate(clf,X,items,y,'direction')
    output={'protocol_sha256':p['sha256'],'baselines':lexical,'models':{}}
    for mid in p['models']:
        folder=OUT/mid.replace('/','__');meta=json.loads((folder/'manifest.json').read_text())
        if meta['protocol_sha256']!=p['sha256']:raise ValueError('Mixed cache')
        path=folder/'states.npz'
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=meta['states_sha256']:raise ValueError('Cache checksum mismatch')
        z=np.load(path);H=z['states'];L=z['choice_logits'];result={}
        source=np.array([index.get(r.get('source_node'),i) for i,r in enumerate(items)])
        target=np.array([index.get(r.get('target_node'),i) for i,r in enumerate(items)])
        for kind,task in [('context','direction'),('ordered_nodes','direction'),('premise_role','necessity')]:
            tr=masks(items,task);dv=masks(items,task,split='dev');curves=[];best=None
            def features(layer):
                h=H[:,layer,:]
                return np.concatenate([h[source],h[target],h[target]-h[source]],axis=1) if kind=='ordered_nodes' else h
            for layer in range(H.shape[1]):
                X=features(layer);clf=probe(X.shape[1]).fit(X[tr],y[tr]);scores=metrics(y[dv],clf.predict_proba(X[dv]))
                value=scores['auroc'] if task=='direction' else scores['balanced_accuracy']
                curves.append({'layer':layer,**scores})
                if layer==0:layer0=evaluate(clf,X,items,y,task)
                elif best is None or value>best[0]:best=(value,layer,clf)
            _,layer,clf=best;X=features(layer)
            save_probe(folder/f'{kind}_probe.npz',clf,X)
            result[kind]={'selected_layer':layer,'development':curves,'test':evaluate(clf,X,items,y,task),'layer0':layer0}
            if task=='direction':
                # Random orientation per whole theory: group/paraphrase labels remain coupled.
                control=y.copy()
                for i,r in enumerate(items):
                    if r['task']=='direction':
                        flip=int(hashlib.sha256((str(SEED)+r['theory']).encode()).hexdigest(),16)%2
                        control[i]=y[i]^flip
                null=probe(X.shape[1]).fit(X[tr],control[tr])
                save_probe(folder/f'{kind}_shuffled_probe.npz',null,X)
                result[kind]['shuffled_control']=evaluate(null,X,items,control,task)
                if kind=='ordered_nodes':
                    a=H[source,layer];c=H[target,layer]
                    cos=(a*c).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(c,axis=1)).clip(1e-12)
                    result[kind]['symmetric_cosine_auroc']={}
                    for split in ('test_iid','test_depth','test_structure'):
                        for style in ('english','symbolic'):
                            m=masks(items,'direction',style,split)
                            result[kind]['symmetric_cosine_auroc'][split+'/'+style]=float(roc_auc_score(y[m],cos[m]))
            print(mid,kind,'selected layer',layer,flush=True)
        result['behavior']={}
        for split in ('test_iid','test_depth','test_structure'):
            for style in ('english','symbolic'):
                m=masks(items,'direction',style,split);margin=L[m,0]-L[m,1]
                n=masks(items,'necessity',style,split)
                u=masks(items,'unrelated',style,split)
                result['behavior'][split+'/'+style]={'direction_auroc':float(roc_auc_score(y[m],margin)),
                    'direction_balanced_accuracy':float(balanced_accuracy_score(y[m],margin>0)),
                    'unrelated_false_positive_rate':float((L[u,0]>L[u,1]).mean()),
                    'premise_role_balanced_accuracy':float(balanced_accuracy_score(y[n],L[n].argmax(axis=1)))}
        output['models'][mid]=result;atomic_json(folder/'probe_results.json',result)
        del H,L,z
    atomic_json(OUT/'analysis.json',output)
    lines=['# Logical-consequence representation pilot','',
           '**Exploratory decoding study, not causal evidence of a reasoning mechanism.**',
           'Ground truth was checked by forward chaining and exhaustive Boolean model enumeration.',
           '300 theory groups; English training/development only; all reversals, paraphrases and premise-role cases remain grouped.',
           'There are no gold answers or proof traces in the encoded prompts. Not-entailed is not the same as false.',
           f"Protocol: `{p['sha256']}`. [Design](../prereg/LOGIC_PILOT.md).",'',
           '## Primary: directed consequence (AUROC)','',
           '| Model | Layer | IID English | Depth 4–5 | New structures | IID symbolic |',
           '|---|---:|---:|---:|---:|---:|']
    for mid,r in output['models'].items():
        v=r['context'];t=v['test'];lines.append(f"| {mid} | {v['selected_layer']} | {t['test_iid/english']['auroc']:.3f} | {t['test_depth/english']['auroc']:.3f} | {t['test_structure/english']['auroc']:.3f} | {t['test_iid/symbolic']['auroc']:.3f} |")
    lines+=['','## Controls and transfer diagnostics (IID English unless specified)','',
            '| Model | Layer 0 AUC | Random-label AUC | Ordered-node AUC | Symmetric cosine AUC | Unrelated false positives | Native answer AUC |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for mid,r in output['models'].items():
        key='test_iid/english';v=r['context'];n=r['ordered_nodes']
        lines.append(f"| {mid} | {v['layer0'][key]['auroc']:.3f} | {v['shuffled_control'][key]['auroc']:.3f} | {n['test'][key]['auroc']:.3f} | {n['symmetric_cosine_auroc'][key]:.3f} | {v['test'][key]['unrelated_false_positive_rate']:.3f} | {r['behavior'][key]['direction_auroc']:.3f} |")
    lines+=['','### Lexical/position baselines','', '| Baseline | IID AUC | Depth AUC | Structure AUC |','|---|---:|---:|---:|']
    for name,v in lexical.items():lines.append(f"| {name} | {v['test_iid/english']['auroc']:.3f} | {v['test_depth/english']['auroc']:.3f} | {v['test_structure/english']['auroc']:.3f} |")
    lines+=['','## Secondary: premise roles (balanced accuracy; chance 1/3)','',
            '| Model | Layer | IID | Depth | Early-gate structure | Native IID |','|---|---:|---:|---:|---:|---:|']
    for mid,r in output['models'].items():
        v=r['premise_role'];t=v['test'];lines.append(f"| {mid} | {v['selected_layer']} | {t['test_iid/english']['balanced_accuracy']:.3f} | {t['test_depth/english']['balanced_accuracy']:.3f} | {t['test_structure/english']['balanced_accuracy']:.3f} | {r['behavior']['test_iid/english']['premise_role_balanced_accuracy']:.3f} |")
    lines+=['','## Limits','',
            'Layers are selected on development data only, excluding layer 0. Probes have fixed C=1 and',
            'a fixed random linear projection to 128 dimensions, with scaling fit only on training rows.',
            'These are low-capacity supervised readouts, not proof of native ordered geometry or causal use.',
            'Symmetric cosine cannot distinguish reversed pairs and is a construction-level control.',
            'Ordered-node probes can recover a ranking without representing every non-entailment; unrelated-query',
            'false positives must be considered. Necessity templates may expose structural shortcuts; the early-gate',
            'holdout tests one such shift. This first pilot contains positive Horn logic, not negation or unrestricted FOL.',
            'All detailed test partitions, controls, layer-selection curves, behavior and cache provenance are in',
            '`results/logic/pilot_v1/analysis.json` and the per-model files. No cross-model capability correlation',
            'or mechanistic claim is made from five models and this small synthetic grammar.','']
    (ROOT/'results/LOGIC_PILOT.md').write_text('\n'.join(lines));print(ROOT/'results/LOGIC_PILOT.md')


if __name__=='__main__':
    with threadpool_limits(limits=1):analyze()
