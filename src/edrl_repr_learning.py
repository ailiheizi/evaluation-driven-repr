"""EDRL Minimal Verification: Can evaluation signal SHAPE representations?
========================================================================
Core hypothesis (Liu): natural evaluation signals contain enough information
to learn meaningful representations WITHOUT explicit feature engineering.

This is NOT clustering pre-trained embeddings (that only tests BGE priors).
This TRAINS a projection using evaluation signal as supervision, then tests
whether the learned space is MORE structured than the untrained baseline.

Design:
- Input: danmaku text -> frozen BGE embedding (512-dim)
- Evaluation signal: auto-extracted evaluation TYPE from keywords
  (高能/爽 excitement, 伏笔/细节 foreshadowing, 感动/泪 emotion,
   无聊/尬 negative, 笑/搞笑 humor, 剧情/逻辑 plot)
- Train: contrastive projection head — same eval-type pull together, diff push apart
- Verify 1 (structure): silhouette score of eval-types AFTER vs BEFORE training
- Verify 2 (downstream): linear probe accuracy predicting eval-type, learned vs BGE

Anti-circularity: train/test split by VIDEO (train on some videos, test on held-out).

Usage:
  export HF_HUB_OFFLINE=1
  python edrl_repr_learning.py
"""
import os, sys, glob, random, re
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from step1b_filter_and_recluster import load_danmaku, rule_filter

# Evaluation-type keyword lexicon (weak supervision, NOT the representation itself)
EVAL_LEXICON = {
    'excitement': ['高能', '爽', '燃', '牛', '厉害', '强', '帅', '炸', '热血', '震撼'],
    'foreshadow': ['伏笔', '细节', '原来', '这里', '注意', '发现', '呼应', '铺垫', '暗示'],
    'emotion':    ['感动', '泪', '哭', '破防', 'emo', '心疼', '难过', '温暖', '治愈'],
    'humor':      ['哈哈', '笑', '搞笑', '梗', '乐', '逗', '沙雕', '好笑'],
    'negative':   ['无聊', '尬', '烂', '差', '拉胯', '难看', '无语', '离谱', '强行'],
    'plot':       ['剧情', '逻辑', '设定', '反转', '结局', '故事', '情节', '悬念'],
}

def label_by_keyword(text):
    """Assign eval-type by keyword match; None if no match"""
    scores = {}
    for etype, kws in EVAL_LEXICON.items():
        c = sum(1 for kw in kws if kw in text)
        if c > 0:
            scores[etype] = c
    if not scores:
        return None
    return max(scores, key=scores.get)


def main():
    import torch
    import torch.nn as nn
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

    # Collect labeled comments from all videos
    videos = sorted(glob.glob(os.path.join(_here, '../data/narrative', 'BV*.json')))
    videos = [v for v in videos if 'asr' not in v]

    data = []  # (text, eval_type, video_id)
    for vf in videos:
        dm, _, _ = load_danmaku(vf)
        vid = os.path.basename(vf)
        for d in rule_filter(dm):
            lbl = label_by_keyword(d['text'])
            if lbl:
                data.append((d['text'], lbl, vid))

    from collections import Counter
    print(f"Labeled comments: {len(data)}")
    print(f"Eval-type distribution: {dict(Counter(x[1] for x in data))}")

    # Train/test split BY VIDEO (anti-circularity)
    all_vids = sorted(set(x[2] for x in data))
    random.shuffle(all_vids)
    n_test = max(3, len(all_vids)//4)
    test_vids = set(all_vids[:n_test])
    train = [x for x in data if x[2] not in test_vids]
    test = [x for x in data if x[2] in test_vids]
    print(f"Train: {len(train)} ({len(all_vids)-n_test} videos), Test: {len(test)} ({n_test} videos)")

    types = sorted(EVAL_LEXICON.keys())
    tid = {t:i for i,t in enumerate(types)}

    # Encode with frozen BGE
    print("Encoding with frozen BGE...")
    train_emb = model.encode([x[0] for x in train], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    test_emb = model.encode([x[0] for x in test], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    train_y = np.array([tid[x[1]] for x in train])
    test_y = np.array([tid[x[1]] for x in test])

    # ===== BASELINE: raw BGE embedding =====
    print("\n=== BASELINE (raw frozen BGE) ===")
    base_sil = silhouette_score(test_emb, test_y, metric='cosine', sample_size=min(3000,len(test_emb)))
    base_probe = cross_val_score(LogisticRegression(max_iter=1000), test_emb, test_y, cv=5).mean()
    print(f"  Silhouette (eval-types): {base_sil:.4f}")
    print(f"  Linear-probe accuracy:   {base_probe:.4f}")

    # ===== TRAIN evaluation-driven projection =====
    print("\n=== TRAINING evaluation-driven projection ===")
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xtr = torch.tensor(train_emb, dtype=torch.float32).to(dev)
    Ytr = torch.tensor(train_y, dtype=torch.long).to(dev)

    # Projection head
    proj = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128)).to(dev)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)

    # Supervised contrastive loss: same eval-type close, diff far
    def sup_con_loss(z, y, temp=0.1):
        z = nn.functional.normalize(z, dim=1)
        sim = z @ z.T / temp
        sim.fill_diagonal_(-1e9)
        y = y.view(-1,1)
        pos_mask = (y == y.T).float()
        pos_mask.fill_diagonal_(0)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos_count = pos_mask.sum(1).clamp(min=1)
        loss = -(pos_mask * log_prob).sum(1) / pos_count
        return loss.mean()

    proj.train()
    batch = 512
    for epoch in range(30):
        perm = torch.randperm(len(Xtr))
        losses = []
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i+batch]
            if len(idx) < 10: continue
            z = proj(Xtr[idx])
            loss = sup_con_loss(z, Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"  epoch {epoch}: loss={np.mean(losses):.4f}")

    # ===== EVALUATE learned projection on HELD-OUT videos =====
    print("\n=== LEARNED (evaluation-driven projection, held-out videos) ===")
    proj.eval()
    with torch.no_grad():
        test_proj = proj(torch.tensor(test_emb, dtype=torch.float32).to(dev)).cpu().numpy()

    learn_sil = silhouette_score(test_proj, test_y, metric='cosine', sample_size=min(3000,len(test_proj)))
    learn_probe = cross_val_score(LogisticRegression(max_iter=1000), test_proj, test_y, cv=5).mean()
    print(f"  Silhouette (eval-types): {learn_sil:.4f}")
    print(f"  Linear-probe accuracy:   {learn_probe:.4f}")

    # ===== RESULTS =====
    print(f"\n{'='*55}")
    print(f"  RESULT: Does evaluation signal shape representations?")
    print(f"{'='*55}")
    print(f"  {'Metric':22s} {'BGE':>10s} {'Learned':>10s} {'Delta':>10s}")
    print(f"  {'Silhouette':22s} {base_sil:>10.4f} {learn_sil:>10.4f} {learn_sil-base_sil:>+10.4f}")
    print(f"  {'Probe accuracy':22s} {base_probe:>10.4f} {learn_probe:>10.4f} {learn_probe-base_probe:>+10.4f}")
    print()
    if learn_sil > base_sil and learn_probe > base_probe:
        print("  ✓ Evaluation-driven training IMPROVES both structure and predictability.")
        print("    -> Evaluation signal CAN shape representations (hypothesis supported).")
    else:
        print("  ✗ No clear improvement. Evaluation signal insufficient to shape representation,")
        print("    or the projection just memorizes the keyword lexicon.")

    print("\n  NOTE: test is on HELD-OUT videos, so improvement is not memorization")
    print("        of specific comments. But keyword-derived labels may leak lexical")
    print("        patterns — a stronger test would use non-lexical evaluation signal.")

    import json
    out = {'baseline':{'silhouette':float(base_sil),'probe':float(base_probe)},
           'learned':{'silhouette':float(learn_sil),'probe':float(learn_probe)},
           'n_train':len(train),'n_test':len(test),'n_test_videos':n_test}
    json.dump(out, open(os.path.join(_here,'../results/narrative/repr_learning.json'),'w'), indent=2)
    print(f"\n  Saved: results/narrative/repr_learning.json")


if __name__ == "__main__":
    main()
