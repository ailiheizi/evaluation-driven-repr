"""Test the CORE EDRL claim: does evaluation signal make structured dimensions EMERGE?
================================================================
The purest test of Liu's hypothesis:
  "Train with evaluation signal -> embedding space auto-differentiates into
   interpretable dimensions (like 咸甜酸苦 from 好吃/难吃)"

Design:
1. Content -> BGE embedding (initial)
2. LLM labels each paragraph on KNOWN latent dims (金句/情感/动作/悬念/描写)
   — these are the "咸甜酸苦" we hope will emerge
3. Train transform T using ONLY review-count as supervision (the "好吃/难吃")
4. Test: does T(embedding) separate the KNOWN dims BETTER than raw BGE?
   — if training on review-count makes 金句/情感/动作 axes MORE separable,
     evaluation signal DID induce structured representation.

KILL: if T doesn't improve separability of known dims over BGE, claim fails.

Usage:
  export DEEPSEEK_API_KEY=xxx; export HF_HUB_OFFLINE=1
  python emergence_test.py
"""
import os, sys, json, time, random
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "D:/windows/code/project/research/memory/memory-engine/memory_engine")

def main():
    import torch, torch.nn as nn
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from deepseek_client import DeepSeekClient
    random.seed(42); torch.manual_seed(42); np.random.seed(42)

    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    client = DeepSeekClient()
    data = json.load(open(os.path.join(_here,'..','data','qidian','shengxu_reviews.json'),encoding='utf-8'))
    segs=[]
    for ci,ch in enumerate(data['chapters']):
        for s in ch['segments']:
            if len(s['reviews'])>=3 and len(s['text'])>=20:
                segs.append({'text':s['text'],'nreviews':len(s['reviews']),'ch':ci})
    print(f'Segments: {len(segs)}')

    # === Step 1: LLM labels KNOWN content dimensions (the "咸甜酸苦") ===
    # For each paragraph, rate on 5 known dims (0/1): 金句 action 情感 悬念 描写
    DIMS=['金句','动作','情感','悬念','描写']
    labels=[]  # list of dict dim->0/1
    print('Labeling known content dimensions via LLM...')
    for i in range(0,len(segs),12):
        batch=segs[i:i+12]
        lines=[f'{j+1}. {s["text"][:80]}' for j,s in enumerate(batch)]
        prompt=('为每个小说段落判断是否具有以下特征(有=1无=0):\n'
                '金句=有哲理/名言/警句, 动作=打斗/激烈动作, 情感=情感强烈/煽情, '
                '悬念=悬念/反转/揭秘, 描写=环境/外貌细致描写\n\n'
                +'\n'.join(lines)+
                '\n\n每行JSON: {"id":N,"金句":0/1,"动作":0/1,"情感":0/1,"悬念":0/1,"描写":0/1}. 只输出JSON.')
        try:
            r=client.chat([{"role":"user","content":prompt}],temperature=0,max_tokens=900)
            got={}
            for line in r['content'].strip().split('\n'):
                line=line.strip()
                if line.startswith('{'):
                    try:
                        o=json.loads(line); got[o['id']-1]=o
                    except: pass
            for j in range(len(batch)):
                labels.append(got.get(j, {d:0 for d in DIMS}))
        except Exception as e:
            for j in range(len(batch)): labels.append({d:0 for d in DIMS})
        if i%48==0: print(f'  {i}/{len(segs)}',flush=True)
        time.sleep(0.3)

    for s,l in zip(segs,labels):
        s['dims']={d:int(l.get(d,0)) for d in DIMS}

    from collections import Counter
    for d in DIMS:
        print(f'  {d}: {sum(s["dims"][d] for s in segs)} positive')

    # === Step 2: embeddings + review-count supervision ===
    emb = model.encode([s['text'] for s in segs], normalize_embeddings=True, show_progress_bar=False)
    counts=np.array([s['nreviews'] for s in segs])
    # review-count as binary supervision (high/low reaction)
    hi_thresh=np.percentile(counts,60)
    react_y=(counts>=hi_thresh).astype(int)

    # === Step 3: separability of KNOWN dims — BGE vs trained ===
    def dim_separability(X):
        """avg 5-fold CV accuracy predicting each known dim from embedding X"""
        accs={}
        for d in DIMS:
            y=np.array([s['dims'][d] for s in segs])
            if y.sum()<10 or (len(y)-y.sum())<10:
                accs[d]=None; continue
            accs[d]=cross_val_score(LogisticRegression(max_iter=1000),X,y,cv=5).mean()
        return accs

    print('\n=== BEFORE training (raw BGE): can we read known dims? ===')
    base_sep=dim_separability(emb)
    for d,a in base_sep.items(): print(f'  {d}: {a:.3f}' if a else f'  {d}: (too few)')

    # Train transform using ONLY review-count supervision (contrastive on reaction)
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    X=torch.tensor(emb,dtype=torch.float32).to(dev)
    Y=torch.tensor(react_y,dtype=torch.long).to(dev)
    class ResT(nn.Module):
        def __init__(s):
            super().__init__(); s.d=nn.Sequential(nn.Dropout(0.3),nn.Linear(512,128),nn.ReLU(),nn.Linear(128,512)); s.a=nn.Parameter(torch.tensor(0.3))
        def forward(s,x): return nn.functional.normalize(x+s.a*s.d(x),dim=1)
    T=ResT().to(dev)
    opt=torch.optim.Adam(T.parameters(),lr=5e-4,weight_decay=1e-3)
    def supcon(z,y,t=0.1):
        z=nn.functional.normalize(z,dim=1); sim=z@z.T/t; sim.fill_diagonal_(-1e9)
        y=y.view(-1,1); pos=(y==y.T).float(); pos.fill_diagonal_(0)
        if pos.sum()==0: return torch.tensor(0.0,device=dev)
        logp=sim-torch.logsumexp(sim,dim=1,keepdim=True)
        return -((pos*logp).sum(1)/pos.sum(1).clamp(min=1)).mean()
    T.train()
    for ep in range(60):
        loss=supcon(T(X),Y); opt.zero_grad(); loss.backward(); opt.step()
    T.eval()
    with torch.no_grad(): emb_t=T(X).cpu().numpy()

    print('\n=== AFTER training (review-count supervised): known dims MORE separable? ===')
    learn_sep=dim_separability(emb_t)
    print(f'  {"Dim":8s} {"BGE":>7s} {"Trained":>8s} {"Delta":>7s}')
    deltas=[]
    for d in DIMS:
        if base_sep[d] and learn_sep[d]:
            dl=learn_sep[d]-base_sep[d]; deltas.append(dl)
            print(f'  {d:8s} {base_sep[d]:>7.3f} {learn_sep[d]:>8.3f} {dl:>+7.3f}')
    print()
    md=np.mean(deltas) if deltas else 0
    if md>0.02:
        print(f'  *** EMERGENCE: review-count training made known dims MORE separable (avg Δ={md:+.3f})')
        print(f'      Evaluation signal induced structured representation!')
    else:
        print(f'  ✗ No emergence: review-count training did NOT improve dim separability (avg Δ={md:+.3f})')
        print(f'      Evaluation signal insufficient to induce structure.')

    json.dump({'base':base_sep,'learned':learn_sep,'mean_delta':float(md),'dims':DIMS},
              open(os.path.join(_here,'..','results','narrative','emergence.json'),'w'),ensure_ascii=False,indent=2)
    print('\n  Saved: results/narrative/emergence.json')

if __name__=='__main__':
    main()
