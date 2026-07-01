"""Leave-One-Type-Out: the HARD test against lexical leakage
========================================================================
Concern: the +0.57 silhouette gain might be trivial keyword memorization
(projection learns "pull same-keyword sentences together").

HARD TEST: train the projection on 5 eval-types, then evaluate structure
on the HELD-OUT 6th type (whose keywords were NEVER in training labels).
If the learned space ALSO better separates the unseen type's sub-structure,
the projection learned GENERAL evaluative representation, not keyword lookup.

Two sub-tests:
  A) Leave-one-type-out transfer: does the projection improve silhouette on
     the held-out type's clustering vs raw BGE? (structure generalizes)
  B) Non-lexical signal: use DANMAKU DENSITY (burst vs quiet) as the eval
     signal — no keywords at all. Does density-contrast still shape the space?

Usage:
  export HF_HUB_OFFLINE=1
  python edrl_hard_test.py
"""
import os, sys, glob, random
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from step1b_filter_and_recluster import load_danmaku, rule_filter

EVAL_LEXICON = {
    'excitement': ['高能','爽','燃','牛','厉害','强','帅','炸','热血','震撼'],
    'foreshadow': ['伏笔','细节','原来','这里','注意','发现','呼应','铺垫','暗示'],
    'emotion':    ['感动','泪','哭','破防','emo','心疼','难过','温暖','治愈'],
    'humor':      ['哈哈','笑','搞笑','梗','乐','逗','沙雕','好笑'],
    'negative':   ['无聊','尬','烂','差','拉胯','难看','无语','离谱','强行'],
    'plot':       ['剧情','逻辑','设定','反转','结局','故事','情节','悬念'],
}

def label_by_keyword(text):
    scores = {e: sum(1 for kw in kws if kw in text) for e,kws in EVAL_LEXICON.items()}
    scores = {e:c for e,c in scores.items() if c>0}
    return max(scores, key=scores.get) if scores else None

def strip_keywords(text):
    """Remove the eval keywords so the projection can't rely on them at test time"""
    for kws in EVAL_LEXICON.values():
        for kw in kws:
            text = text.replace(kw, '')
    return text

def main():
    import torch, torch.nn as nn
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    random.seed(42); torch.manual_seed(42); np.random.seed(42)
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

    videos = [v for v in sorted(glob.glob(os.path.join(_here,'../data/narrative','BV*.json'))) if 'asr' not in v]
    data = []
    for vf in videos:
        dm,_,_ = load_danmaku(vf)
        for d in rule_filter(dm):
            lbl = label_by_keyword(d['text'])
            if lbl: data.append((d['text'], lbl, os.path.basename(vf)))

    types = sorted(EVAL_LEXICON.keys())
    tid = {t:i for i,t in enumerate(types)}

    def sup_con_loss(z, y, temp=0.1):
        z = nn.functional.normalize(z, dim=1)
        sim = z @ z.T / temp
        sim.fill_diagonal_(-1e9)
        y = y.view(-1,1)
        pos = (y==y.T).float(); pos.fill_diagonal_(0)
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        return -((pos*logp).sum(1)/pos.sum(1).clamp(min=1)).mean()

    def train_proj(Xtr, Ytr, dev):
        proj = nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,128)).to(dev)
        opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
        proj.train()
        for ep in range(30):
            perm = torch.randperm(len(Xtr))
            for i in range(0,len(Xtr),512):
                idx = perm[i:i+512]
                if len(idx)<10: continue
                loss = sup_con_loss(proj(Xtr[idx]), Ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        proj.eval()
        return proj

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ============ TEST A: Leave-One-Type-Out ============
    print("="*60)
    print("  TEST A: Leave-One-Type-Out (keyword-stripped test)")
    print("="*60)
    print("  Train projection on 5 types, test structure on held-out 6th.")
    print("  Test comments have their eval-keywords STRIPPED (no lexical leak).")
    print()

    results_a = []
    for held in types:
        train = [(t,l,v) for t,l,v in data if l != held]
        # held-out: use the held type's comments, but we test whether they
        # separate from a random sample of OTHER comments in learned space
        held_pos = [(t,l,v) for t,l,v in data if l == held]
        held_neg = random.sample([(t,l,v) for t,l,v in data if l != held], min(len(held_pos), 800))

        # Strip keywords from ALL test comments (so projection can't cheat)
        test_texts = [strip_keywords(t) for t,_,_ in held_pos] + [strip_keywords(t) for t,_,_ in held_neg]
        test_lbls = [1]*len(held_pos) + [0]*len(held_neg)

        tr_emb = model.encode([t for t,_,_ in train], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        tr_y = torch.tensor([tid[l] for _,l,_ in train], dtype=torch.long).to(dev)
        Xtr = torch.tensor(tr_emb, dtype=torch.float32).to(dev)

        te_emb = model.encode(test_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        te_y = np.array(test_lbls)

        # Baseline: raw BGE separability of held type (probe)
        base = cross_val_score(LogisticRegression(max_iter=1000), te_emb, te_y, cv=5).mean()

        # Learned projection (trained WITHOUT the held type)
        proj = train_proj(Xtr, tr_y, dev)
        with torch.no_grad():
            te_proj = proj(torch.tensor(te_emb,dtype=torch.float32).to(dev)).cpu().numpy()
        learned = cross_val_score(LogisticRegression(max_iter=1000), te_proj, te_y, cv=5).mean()

        results_a.append((held, base, learned))
        print(f"  held={held:12s}: BGE probe={base:.3f}  learned={learned:.3f}  delta={learned-base:+.3f}")

    mean_base = np.mean([r[1] for r in results_a])
    mean_learn = np.mean([r[2] for r in results_a])
    print(f"\n  MEAN: BGE={mean_base:.3f}  learned={mean_learn:.3f}  delta={mean_learn-mean_base:+.3f}")
    print(f"  -> If learned > BGE on UNSEEN types with keywords stripped,")
    print(f"     the projection learned generalizable evaluative structure.")

    import json
    json.dump({'leave_one_type_out':[{'held':h,'bge':float(b),'learned':float(l)} for h,b,l in results_a],
               'mean_bge':float(mean_base),'mean_learned':float(mean_learn)},
              open(os.path.join(_here,'../results/narrative/hard_test.json'),'w'), indent=2)
    print(f"\n  Saved: results/narrative/hard_test.json")


if __name__ == "__main__":
    main()
