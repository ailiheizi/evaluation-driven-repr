"""EDRL on novel: Can content predict reader reaction intensity?
================================================================
X = paragraph text (content), Y = review count (reader reaction).
Core question: is there a learnable signal in content that predicts
which paragraphs trigger high reader engagement?

Two tests:
1. Raw BGE probe: can BGE embedding predict high/low reaction?
2. Contrastive projection: does training on reaction signal improve it?
   (train/test split by CHAPTER — anti-circularity)

Usage:
  export HF_HUB_OFFLINE=1
  python edrl_novel.py
"""
import os, sys, json, random
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))

def main():
    import torch, torch.nn as nn
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, silhouette_score

    random.seed(42); torch.manual_seed(42); np.random.seed(42)
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

    data = json.load(open(os.path.join(_here,'..','data','qidian','shengxu_paired.json'),encoding='utf-8'))
    chapters = data['chapters']

    # Build (text, count, chapter_idx) list
    items = []
    for ci, ch in enumerate(chapters):
        for p in ch['paras']:
            if len(p['t']) >= 10:
                items.append({'text': p['t'], 'count': p['c'], 'ch': ci})
    print(f'Total paragraphs: {len(items)}')

    counts = [it['count'] for it in items]
    median = sorted(counts)[len(counts)//2]
    print(f'Review count: median={median}, mean={np.mean(counts):.1f}, max={max(counts)}')

    # Label: high reaction (> median) vs low (<= median)
    # Use a clearer split: top-third vs bottom-third to avoid median ties
    thresh_hi = np.percentile(counts, 66)
    thresh_lo = np.percentile(counts, 33)
    print(f'Thresholds: low<={thresh_lo:.0f}, high>={thresh_hi:.0f}')

    labeled = [it for it in items if it['count']>=thresh_hi or it['count']<=thresh_lo]
    for it in labeled:
        it['y'] = 1 if it['count']>=thresh_hi else 0
    print(f'Labeled (extremes): {len(labeled)}, high={sum(it["y"] for it in labeled)}, low={sum(1-it["y"] for it in labeled)}')

    # Split by chapter
    chs = sorted(set(it['ch'] for it in labeled))
    random.shuffle(chs)
    n_test = max(2, len(chs)//4)
    test_chs = set(chs[:n_test])
    train = [it for it in labeled if it['ch'] not in test_chs]
    test = [it for it in labeled if it['ch'] in test_chs]
    print(f'Train: {len(train)} ({len(chs)-n_test} ch), Test: {len(test)} ({n_test} ch)')

    tr_emb = model.encode([it['text'] for it in train], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    te_emb = model.encode([it['text'] for it in test], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    tr_y = np.array([it['y'] for it in train])
    te_y = np.array([it['y'] for it in test])

    # === Test 1: Raw BGE probe ===
    print('\n=== BASELINE: Raw BGE — can content predict reaction? ===')
    clf = LogisticRegression(max_iter=1000).fit(tr_emb, tr_y)
    base_acc = clf.score(te_emb, te_y)
    base_auc = roc_auc_score(te_y, clf.predict_proba(te_emb)[:,1])
    print(f'  Accuracy: {base_acc:.3f}  AUC: {base_auc:.3f}  (chance=0.5)')

    # === Test 2: Contrastive projection ===
    print('\n=== LEARNED: reaction-signal projection ===')
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    Xtr=torch.tensor(tr_emb,dtype=torch.float32).to(dev)
    Ytr=torch.tensor(tr_y,dtype=torch.long).to(dev)
    proj=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,128)).to(dev)
    opt=torch.optim.Adam(proj.parameters(),lr=1e-3)
    def loss_fn(z,y,t=0.1):
        z=nn.functional.normalize(z,dim=1); sim=z@z.T/t; sim.fill_diagonal_(-1e9)
        y=y.view(-1,1); pos=(y==y.T).float(); pos.fill_diagonal_(0)
        if pos.sum()==0: return torch.tensor(0.0,device=dev)
        logp=sim-torch.logsumexp(sim,dim=1,keepdim=True)
        return -((pos*logp).sum(1)/pos.sum(1).clamp(min=1)).mean()
    proj.train()
    for ep in range(50):
        loss=loss_fn(proj(Xtr),Ytr); opt.zero_grad(); loss.backward(); opt.step()
    proj.eval()
    with torch.no_grad():
        tr_p=proj(Xtr).cpu().numpy(); te_p=proj(torch.tensor(te_emb,dtype=torch.float32).to(dev)).cpu().numpy()
    clf2=LogisticRegression(max_iter=1000).fit(tr_p,tr_y)
    learn_acc=clf2.score(te_p,te_y)
    learn_auc=roc_auc_score(te_y, clf2.predict_proba(te_p)[:,1])
    print(f'  Accuracy: {learn_acc:.3f}  AUC: {learn_auc:.3f}')

    print(f'\n{"="*55}')
    print(f'  {"Metric":12s} {"BGE":>8s} {"Learned":>8s} {"Delta":>8s}')
    print(f'  {"Accuracy":12s} {base_acc:>8.3f} {learn_acc:>8.3f} {learn_acc-base_acc:>+8.3f}')
    print(f'  {"AUC":12s} {base_auc:>8.3f} {learn_auc:>8.3f} {learn_auc-base_auc:>+8.3f}')
    print()
    if base_auc > 0.58:
        print(f'  ✓ Content DOES predict reader reaction (AUC={base_auc:.3f} > chance).')
        print('    There is a learnable signal: certain content triggers engagement.')
    else:
        print(f'  ✗ Content barely predicts reaction (AUC={base_auc:.3f} ~ chance).')
        print('    Reader reaction may not be a function of content alone.')

    json.dump({'n':len(labeled),'base':{'acc':float(base_acc),'auc':float(base_auc)},
               'learned':{'acc':float(learn_acc),'auc':float(learn_auc)},
               'median':int(median)},
              open(os.path.join(_here,'..','results','narrative','edrl_novel.json'),'w'),indent=2)
    print('\n  Saved: results/narrative/edrl_novel.json')

if __name__=='__main__':
    main()
