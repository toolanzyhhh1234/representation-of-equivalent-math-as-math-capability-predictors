"""Frozen forward-pass pilot for directional consequence and premise roles."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ROOT
from .h2_data import digest
from .logic_data import DEST, build
from .run_h2 import atomic_json, environment
from .run_h2_followup import MODELS

PROTOCOL = ROOT/"prereg/logic_pilot.json"
OUT = ROOT/"results/logic/pilot_v1"
FILES = ["src/logic_data.py","src/run_logic.py","src/analyze_logic.py","prereg/LOGIC_PILOT.md",
         "src/h2_data.py","src/run_h2.py","src/run_h2_followup.py","src/eval_gsm8k.py",
         "src/hf_data.py","src/config.py","data/hf_sources.json","pyproject.toml","uv.lock"]


def fingerprints():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in FILES}


def prepare():
    bundle = build()
    value = {"schema":1,"models":MODELS,"dataset_sha256":bundle["sha256"],
             "settings":{"batch_size":16,"dtype":"bfloat16","pooling":"last_real_token",
                         "projection_dimensions":128,"regularization_C":1.0,"seed":20260924},
             "environment":environment(),"code_sha256":fingerprints()}
    value["sha256"] = digest(value)
    if PROTOCOL.exists():
        old=json.loads(PROTOCOL.read_text())
        if {k:v for k,v in old.items() if k!="created_utc"} != value: raise ValueError("Changed frozen protocol")
        return
    value["created_utc"]=datetime.now(timezone.utc).isoformat()
    atomic_json(PROTOCOL,value); print('Frozen logic protocol:',value['sha256'])


def inputs(runtime=False):
    p=json.loads(PROTOCOL.read_text()); b=json.loads(DEST.read_text())
    if p['sha256']!=digest({k:v for k,v in p.items() if k not in ('sha256','created_utc')}): raise ValueError('Protocol changed')
    if b['sha256']!=digest({k:v for k,v in b.items() if k!='sha256'}) or b['sha256']!=p['dataset_sha256']: raise ValueError('Dataset changed')
    if p['code_sha256']!=fingerprints(): raise ValueError('Frozen code changed')
    if runtime and p['environment']!=environment(): raise ValueError('Runtime changed')
    return p,b


@torch.no_grad()
def forward(model,tok,texts,choice_ids):
    enc=tok(texts,return_tensors='pt',padding=True,truncation=False).to('cuda')
    last=enc['attention_mask'].sum(1)-1; rows=torch.arange(len(texts),device='cuda')
    # Decoder only: do not allocate vocabulary logits at every input position.
    output=model.model(**enc,output_hidden_states=True,use_cache=False)
    states=torch.stack([h[rows,last].float() for h in output.hidden_states],dim=1)
    logits=model.get_output_embeddings()(output.last_hidden_state[rows,last])[:,choice_ids].float()
    return states.cpu().numpy(),logits.cpu().numpy()


def extract_model(p,b,mid):
    folder=OUT/mid.replace('/','__');folder.mkdir(parents=True,exist_ok=True)
    cache=folder/'states.npz';meta=folder/'manifest.json'
    expected={'protocol_sha256':p['sha256'],'model':mid,'revision':p['models'][mid],'dataset_sha256':b['sha256']}
    if cache.exists():
        m=json.loads(meta.read_text())
        if any(m[k]!=v for k,v in expected.items()):raise ValueError('Mixed activation cache')
        with cache.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=m['states_sha256']:raise ValueError('Activation checksum mismatch')
        print('Verified cached',mid,flush=True);return
    tok=AutoTokenizer.from_pretrained(mid,revision=p['models'][mid]);tok.padding_side='right'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    choices=[tok.encode(' '+s,add_special_tokens=False) for s in ('A','B','C')]
    if any(len(x)!=1 for x in choices):raise ValueError('Choice must be one continuation token')
    # Verify the continuation IDs in actual answer-marker context.
    prefix=tok.encode('Answer:',add_special_tokens=False)
    if any(tok.encode('Answer: '+s,add_special_tokens=False)!=prefix+x for s,x in zip(('A','B','C'),choices)):
        raise ValueError('Continuation tokenization mismatch')
    choice_ids=[x[0] for x in choices]
    model=AutoModelForCausalLM.from_pretrained(mid,revision=p['models'][mid],dtype=torch.bfloat16,
                                               attn_implementation='sdpa').cuda().eval()
    texts=[r['prompt'] for r in b['items']]
    lengths=[len(x) for x in tok(texts)['input_ids']]
    if max(lengths)>model.config.max_position_embeddings:raise ValueError('Context overflow')
    probe=sorted(range(len(texts)),key=lambda i:lengths[i])[::max(1,len(texts)//8)][:8]
    batched,_=forward(model,tok,[texts[i] for i in probe],choice_ids)
    single=np.concatenate([forward(model,tok,[texts[i]],choice_ids)[0] for i in probe])
    a=batched.reshape(8,-1);c=single.reshape(8,-1)
    cosine=(a*c).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(c,axis=1))
    if cosine.min()<.999:raise ValueError('Logic padding invariance failed')
    order=sorted(range(len(texts)),key=lambda i:lengths[i]);states=None;logits=np.empty((len(texts),3),np.float32)
    start=time.monotonic()
    for offset in range(0,len(order),p['settings']['batch_size']):
        idx=order[offset:offset+p['settings']['batch_size']]
        h,l=forward(model,tok,[texts[i] for i in idx],choice_ids)
        if states is None:states=np.empty((len(texts),*h.shape[1:]),np.float32)
        states[idx]=h;logits[idx]=l
        if offset%256==0:print(f'{mid}: {min(offset+len(idx),len(texts))}/{len(texts)} states ({time.monotonic()-start:.0f}s)',flush=True)
    tmp=folder/'states.tmp.npz';np.savez_compressed(tmp,states=states,choice_logits=logits);tmp.replace(cache)
    with cache.open('rb') as f: checksum=hashlib.file_digest(f,'sha256').hexdigest()
    atomic_json(meta,{**expected,'states_sha256':checksum,'shape':list(states.shape),'dtype':'float32',
                     'forward_dtype':'bfloat16','padding_min_cosine':float(cosine.min()),'max_prompt_tokens':max(lengths),
                     'n_params':sum(x.numel() for x in model.parameters()),'choice_token_ids':choice_ids})
    del model,states;gc.collect();torch.cuda.empty_cache()
    print('Saved',mid,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--models',nargs='+',choices=list(MODELS),default=list(MODELS));a=parser.parse_args()
    if a.prepare:prepare();return
    p,b=inputs(runtime=True)
    for mid in a.models:extract_model(p,b,mid)


if __name__=='__main__':main()
