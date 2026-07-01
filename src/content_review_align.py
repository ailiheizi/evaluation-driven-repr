"""Content-Review Alignment via Contrastive Learning
================================================================
The RIGHT experiment (anchor test confirmed signal exists, d=0.6).

Train two projections:
  f: paragraph content embedding -> shared space
  g: aggregated review embedding -> shared space
Contrastive loss: f(content_i) should match g(reviews_i), not g(reviews_j).

This learns "what in the content drives the reviews" WITHOUT predefined
categories. Evaluation: on held-out chapters, can f(content) retrieve the
correct review-set among candidates? (Recall@k vs random baseline)

Usage:
  export HF_HUB_OFFLINE=1
  python content_review_align.py
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
            if len(s['reviews'])>=4 and len(s['text'])>=15:
                segs.append({'text':s['text'],'reviews':s['reviews'][:15],'ch':ci})
    print(f'Segments: {len(segs)}')

    # Embed content and aggregated reviews (mean of review embeddings)
    print('Encoding content and reviews...')
    content_emb = model.encode([s['text'] for s in segs], normalize_embeddings=True, show_progress_bar=False)
    review_emb = []
    for s in segs:
        re = model.encode(s['reviews'], normalize_embeddings=True, show_progress_bar=False)
        review_emb.append(re.mean(axis=0))
    review_emb = np.array(review_emb)

    chs=sorted(set(s['ch'] for s in segs)); random.shuffle(chs)
    nt=max(4,len(chs)//4); tc=set(chs[:nt])
    tr_idx=[i for i,s in enumerate(segs) if s['ch'] not in tc]
    te_idx=[i for i,s in enumerate(segs) if s['ch'] in tc]
    print(f'Train: {len(tr_idx)} segs, Test: {len(te_idx)} segs ({nt} chapters held out)')

    dev='cuda' if torch.cuda.is_available() else 'cpu'
    Ctr=torch.tensor(content_emb[tr_idx],dtype=torch.float32).to(dev)
    Rtr=torch.tensor(review_emb[tr_idx],dtype=torch.float32).to(dev)
    Cte=torch.tensor(content_emb[te_idx],dtype=torch.float32).to(dev)
    Rte=torch.tensor(review_emb[te_idx],dtype=torch.float32).to(dev)

    # Two-tower projections
    def tower(): return nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,128)).to(dev)
    fC, gR = tower(), tower()
    opt=torch.optim.Adam(list(fC.parameters())+list(gR.parameters()),lr=1e-3)

    def clip_loss(c,r,t=0.07):
        c=nn.functional.normalize(fC(c),dim=1); r=nn.functional.normalize(gR(r),dim=1)
        logits=c@r.T/t
        labels=torch.arange(len(c),device=dev)
        return (nn.functional.cross_entropy(logits,labels)+nn.functional.cross_entropy(logits.T,labels))/2

    # Recall@k eval: does f(content) rank its true reviews high?
    def recall_at_k(C,R,ks=(1,5,10)):
        fC.eval(); gR.eval()
        with torch.no_grad():
            c=nn.functional.normalize(fC(C),dim=1); r=nn.functional.normalize(gR(R),dim=1)
            sim=(c@r.T).cpu().numpy()
        n=len(sim); out={}
        for k in ks:
            hit=sum(1 for i in range(n) if i in np.argsort(-sim[i])[:k])
            out[k]=hit/n
        return out

    # Baseline: raw BGE (no training) recall
    def raw_recall(Ci,Ri,ks=(1,5,10)):
        c=Ci.cpu().numpy(); r=Ri.cpu().numpy()
        c=c/np.linalg.norm(c,axis=1,keepdims=True); r=r/np.linalg.norm(r,axis=1,keepdims=True)
        sim=c@r.T; n=len(sim); out={}
        for k in ks:
            out[k]=sum(1 for i in range(n) if i in np.argsort(-sim[i])[:k])/n
        return out

    base=raw_recall(Cte,Rte)
    rand={k:k/len(te_idx) for k in (1,5,10)}
    print(f'\nBaseline (raw BGE) Recall@1/5/10: {base[1]:.3f}/{base[5]:.3f}/{base[10]:.3f}')
    print(f'Random chance Recall@1/5/10: {rand[1]:.3f}/{rand[5]:.3f}/{rand[10]:.3f}')

    print('\nTraining two-tower alignment...')
    fC.train(); gR.train()
    for ep in range(100):
        loss=clip_loss(Ctr,Rtr); opt.zero_grad(); loss.backward(); opt.step()
        if ep%25==0: print(f'  ep{ep}: loss={loss.item():.4f}')

    learned=recall_at_k(Cte,Rte)
    print(f'\n=== RESULT: content->review retrieval on held-out chapters ===')
    print(f'  {"Method":16s} {"R@1":>7s} {"R@5":>7s} {"R@10":>7s}')
    print(f'  {"Random":16s} {rand[1]:>7.3f} {rand[5]:>7.3f} {rand[10]:>7.3f}')
    print(f'  {"Raw BGE":16s} {base[1]:>7.3f} {base[5]:>7.3f} {base[10]:>7.3f}')
    print(f'  {"Learned (ours)":16s} {learned[1]:>7.3f} {learned[5]:>7.3f} {learned[10]:>7.3f}')
    print()
    if learned[5]>base[5]+0.03:
        print(f'  *** Training IMPROVED content->review alignment!')
        print(f'      The model learned content features that predict reviews.')
    elif base[5]>rand[5]*2:
        print(f'  ~ Raw BGE already aligns (content-review linked), training marginal.')
    else:
        print(f'  ✗ No meaningful alignment.')

    json.dump({'random':rand,'raw_bge':base,'learned':learned,'n_test':len(te_idx)},
              open(os.path.join(_here,'..','results','narrative','content_review_align.json'),'w'),indent=2)
    print('\n  Saved: results/narrative/content_review_align.json')

if __name__=='__main__':
    main()
