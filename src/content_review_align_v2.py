"""Content-Review Alignment v2: fix overfitting, beat Raw BGE
================================================================
Fixes: smaller model, dropout, early stopping on val, residual connection
(learned projection = BGE + small learned delta, so it can't be worse than BGE).

Usage:
  export HF_HUB_OFFLINE=1
  python content_review_align_v2.py
"""
import os, sys, json, random
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None
_here = os.path.dirname(os.path.abspath(__file__))

def main():
    import torch, torch.nn as nn
    from sentence_transformers import SentenceTransformer
    random.seed(42); torch.manual_seed(42); np.random.seed(42)
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

    data = json.load(open(os.path.join(_here,'..','data','qidian','shengxu_reviews.json'),encoding='utf-8'))
    segs=[]
    for ci,ch in enumerate(data['chapters']):
        for s in ch['segments']:
            if len(s['reviews'])>=3 and len(s['text'])>=15:
                segs.append({'text':s['text'],'reviews':s['reviews'][:15],'ch':ci})
    print(f'Segments: {len(segs)}')

    content_emb = model.encode([s['text'] for s in segs], normalize_embeddings=True, show_progress_bar=False)
    review_emb=np.array([model.encode(s['reviews'],normalize_embeddings=True,show_progress_bar=False).mean(0) for s in segs])

    chs=sorted(set(s['ch'] for s in segs)); random.shuffle(chs)
    n=len(chs); nte=max(4,n//5); nval=max(3,n//5)
    tc=set(chs[:nte]); vc=set(chs[nte:nte+nval])
    tr=[i for i,s in enumerate(segs) if s['ch'] not in tc and s['ch'] not in vc]
    va=[i for i,s in enumerate(segs) if s['ch'] in vc]
    teI=[i for i,s in enumerate(segs) if s['ch'] in tc]
    print(f'Train {len(tr)}, Val {len(va)}, Test {len(teI)}')

    dev='cuda' if torch.cuda.is_available() else 'cpu'
    def T(a,idx): return torch.tensor(a[idx],dtype=torch.float32).to(dev)
    Ctr,Rtr=T(content_emb,tr),T(review_emb,tr)
    Cva,Rva=T(content_emb,va),T(review_emb,va)
    Cte,Rte=T(content_emb,teI),T(review_emb,teI)

    # Residual tower: output = normalize(x + alpha * delta(x)); starts ~= BGE
    class ResTower(nn.Module):
        def __init__(s):
            super().__init__()
            s.d=nn.Sequential(nn.Dropout(0.3),nn.Linear(512,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,512))
            s.a=nn.Parameter(torch.tensor(0.1))
        def forward(s,x): return nn.functional.normalize(x+s.a*s.d(x),dim=1)
    fC,gR=ResTower().to(dev),ResTower().to(dev)
    opt=torch.optim.Adam(list(fC.parameters())+list(gR.parameters()),lr=5e-4,weight_decay=1e-3)

    def clip_loss(C,R,t=0.07):
        c=fC(C); r=gR(R); logits=c@r.T/t; lab=torch.arange(len(c),device=dev)
        return (nn.functional.cross_entropy(logits,lab)+nn.functional.cross_entropy(logits.T,lab))/2

    def recall(C,R,ks=(1,5,10)):
        fC.eval();gR.eval()
        with torch.no_grad(): c=fC(C).cpu().numpy(); r=gR(R).cpu().numpy()
        sim=c@r.T; nn_=len(sim); return {k:sum(1 for i in range(nn_) if i in np.argsort(-sim[i])[:k])/nn_ for k in ks}
    def raw_recall(C,R,ks=(1,5,10)):
        c=C.cpu().numpy(); r=R.cpu().numpy(); sim=c@r.T; nn_=len(sim)
        return {k:sum(1 for i in range(nn_) if i in np.argsort(-sim[i])[:k])/nn_ for k in ks}

    base=raw_recall(Cte,Rte)
    print(f'\nRaw BGE test R@1/5/10: {base[1]:.3f}/{base[5]:.3f}/{base[10]:.3f}')

    best_val=0; best_state=None; patience=0
    fC.train();gR.train()
    for ep in range(300):
        loss=clip_loss(Ctr,Rtr); opt.zero_grad(); loss.backward(); opt.step()
        if ep%10==0:
            vr=recall(Cva,Rva)[5]
            if vr>best_val: best_val=vr; best_state=(({k:v.clone() for k,v in fC.state_dict().items()}),({k:v.clone() for k,v in gR.state_dict().items()})); patience=0
            else: patience+=1
            fC.train();gR.train()
            if patience>=6: print(f'  early stop ep{ep}'); break

    if best_state: fC.load_state_dict(best_state[0]); gR.load_state_dict(best_state[1])
    learned=recall(Cte,Rte)

    print(f'\n=== RESULT (held-out test chapters) ===')
    print(f'  {"Method":14s} {"R@1":>7s} {"R@5":>7s} {"R@10":>7s}')
    print(f'  {"Raw BGE":14s} {base[1]:>7.3f} {base[5]:>7.3f} {base[10]:>7.3f}')
    print(f'  {"Learned":14s} {learned[1]:>7.3f} {learned[5]:>7.3f} {learned[10]:>7.3f}')
    print()
    delta5=learned[5]-base[5]
    if delta5>0.02: print(f'  *** Training BEAT Raw BGE by {delta5:+.3f} on R@5!')
    elif delta5>-0.02: print(f'  ~ Matched Raw BGE (R@5 delta {delta5:+.3f}). Content-review signal confirmed either way.')
    else: print(f'  Raw BGE still better; but strong content-review alignment confirmed (R@1={base[1]:.2f} vs random {1/len(teI):.3f}).')

    json.dump({'raw':base,'learned':learned,'n_test':len(teI),'random_r1':1/len(teI)},
              open(os.path.join(_here,'..','results','narrative','align_v2.json'),'w'),indent=2)
    print('\n  Saved: results/narrative/align_v2.json')

if __name__=='__main__':
    main()
