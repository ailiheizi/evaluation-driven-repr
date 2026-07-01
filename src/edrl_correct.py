"""EDRL Correct Design: Evaluation signal shapes CONTENT representation
========================================================================
THE KEY INSIGHT: X ≠ Y. Content (ASR) is the OBJECT, danmaku is the SIGNAL.

Design:
  X = ASR text segments (what's being said in the video) -> embedding
  Y = same-timebin danmaku evaluation type (audience reaction to that content)
  Train: projection on X, supervised by Y
  Verify: does X's embedding become structured by evaluation types?

  If YES: evaluation signal shapes content representation (hypothesis confirmed)
  The content embedding now encodes "what kind of audience reaction this triggers"
  WITHOUT ever seeing the evaluation keywords — only the video content.

Anti-circularity:
  - Train/test split by TIME (first 70% of video = train, last 30% = test)
  - At test time, we embed ONLY ASR (no danmaku visible), predict eval type
  - If learned projection predicts eval type from ASR alone better than baseline
    → content representation was shaped to encode "what evaluation it triggers"

Usage:
  export HF_HUB_OFFLINE=1
  python edrl_correct.py
"""
import os, sys, glob, random, json
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


def main():
    import torch, torch.nn as nn
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from collections import Counter

    random.seed(42); torch.manual_seed(42); np.random.seed(42)
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

    # Load ASR + danmaku for the video we have ASR for
    asr_path = os.path.join(_here, '../data/narrative/BV1LSoyYqEuU_asr.json')
    dm_path = os.path.join(_here, '../data/narrative/BV1LSoyYqEuU.json')

    # Copy ASR from paper repo if not present
    if not os.path.exists(asr_path):
        import shutil
        src = 'D:/windows/code/project/research/paper/repo/data/narrative/BV1LSoyYqEuU_asr.json'
        shutil.copy(src, asr_path)

    with open(asr_path, 'r', encoding='utf-8') as f:
        asr_data = json.load(f)
    asr_segs = asr_data['segments']

    danmaku, title, duration = load_danmaku(dm_path)
    filtered = rule_filter(danmaku)
    print(f"Video: {title}")
    print(f"ASR segments: {len(asr_segs)}")
    print(f"Filtered danmaku: {len(filtered)}")

    # Aggregate into 30s bins
    bin_size = 30
    n_bins = int(np.ceil(duration / bin_size))

    bins = []
    for b in range(n_bins):
        start, end = b*bin_size, (b+1)*bin_size

        # ASR text for this bin (concatenate segments overlapping this window)
        asr_text = ' '.join(s['text'] for s in asr_segs
                           if s['start'] < end and s['end'] > start)

        # Danmaku eval labels for this bin
        dm_in_bin = [d for d in filtered if start <= d['t'] < end]
        labels = [label_by_keyword(d['text']) for d in dm_in_bin]
        labels = [l for l in labels if l is not None]

        if not asr_text.strip() or len(labels) < 3:
            continue

        # Majority eval type for this bin
        majority = Counter(labels).most_common(1)[0][0]
        bins.append({'asr': asr_text, 'eval_type': majority, 'time': start})

    types = sorted(EVAL_LEXICON.keys())
    tid = {t:i for i,t in enumerate(types)}

    # Filter to bins with enough representation
    type_counts = Counter(b['eval_type'] for b in bins)
    print(f"\nBins with eval labels: {len(bins)}")
    print(f"Eval type distribution: {dict(type_counts)}")

    # Keep types with >= 3 bins
    valid_types = {t for t,c in type_counts.items() if c >= 3}
    bins = [b for b in bins if b['eval_type'] in valid_types]
    print(f"After filtering (>=3 bins/type): {len(bins)} bins, types={valid_types}")

    if len(bins) < 15:
        print("Not enough data for this video alone. Need more ASR videos.")
        return

    # Encode ASR text (this is X — the CONTENT, no danmaku keywords here)
    asr_texts = [b['asr'] for b in bins]
    asr_emb = model.encode(asr_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    labels_arr = np.array([tid[b['eval_type']] for b in bins])

    # Train/test split by TIME (temporal generalization)
    split_idx = int(len(bins) * 0.7)
    train_emb, test_emb = asr_emb[:split_idx], asr_emb[split_idx:]
    train_y, test_y = labels_arr[:split_idx], labels_arr[split_idx:]

    print(f"\nTrain: {len(train_emb)} bins (first 70% of video)")
    print(f"Test:  {len(test_emb)} bins (last 30% — temporal generalization)")

    # === BASELINE: raw BGE of ASR text ===
    print("\n=== BASELINE (raw BGE of ASR text) ===")
    if len(set(test_y)) > 1:
        base_sil = silhouette_score(test_emb, test_y, metric='cosine') if len(test_emb) > len(set(test_y)) else -1
        base_probe = LogisticRegression(max_iter=1000).fit(train_emb, train_y).score(test_emb, test_y)
    else:
        base_sil = -1; base_probe = 0
        print("  Only one class in test set — single-video too small")
    print(f"  Silhouette: {base_sil:.4f}")
    print(f"  Probe accuracy: {base_probe:.4f}")

    # === TRAIN evaluation-driven projection on ASR content ===
    print("\n=== TRAINING: eval signal shapes ASR content embedding ===")
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xtr = torch.tensor(train_emb, dtype=torch.float32).to(dev)
    Ytr = torch.tensor(train_y, dtype=torch.long).to(dev)

    proj = nn.Sequential(nn.Linear(512,256), nn.ReLU(), nn.Linear(256,128)).to(dev)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)

    def sup_con_loss(z, y, temp=0.1):
        z = nn.functional.normalize(z, dim=1)
        sim = z @ z.T / temp
        sim.fill_diagonal_(-1e9)
        y = y.view(-1,1)
        pos = (y==y.T).float(); pos.fill_diagonal_(0)
        if pos.sum() == 0: return torch.tensor(0.0, device=dev)
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        return -((pos*logp).sum(1) / pos.sum(1).clamp(min=1)).mean()

    proj.train()
    for ep in range(50):
        z = proj(Xtr)
        loss = sup_con_loss(z, Ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0:
            print(f"  epoch {ep}: loss={loss.item():.4f}")

    # === EVALUATE on held-out TIME SEGMENT ===
    print("\n=== LEARNED (ASR content shaped by eval signal, future time) ===")
    proj.eval()
    with torch.no_grad():
        test_proj = proj(torch.tensor(test_emb, dtype=torch.float32).to(dev)).cpu().numpy()

    if len(set(test_y)) > 1 and len(test_proj) > len(set(test_y)):
        learn_sil = silhouette_score(test_proj, test_y, metric='cosine')
        learn_probe = LogisticRegression(max_iter=1000).fit(
            proj(Xtr).detach().cpu().numpy(), train_y
        ).score(test_proj, test_y)
    else:
        learn_sil = -1; learn_probe = 0

    print(f"  Silhouette: {learn_sil:.4f}")
    print(f"  Probe accuracy: {learn_probe:.4f}")

    # === RESULT ===
    print(f"\n{'='*60}")
    print(f"  RESULT: Can eval signal shape CONTENT representation?")
    print(f"{'='*60}")
    print(f"  {'Metric':22s} {'BGE(ASR)':>10s} {'Learned':>10s} {'Delta':>10s}")
    print(f"  {'Silhouette':22s} {base_sil:>10.4f} {learn_sil:>10.4f} {learn_sil-base_sil:>+10.4f}")
    print(f"  {'Probe accuracy':22s} {base_probe:>10.4f} {learn_probe:>10.4f} {learn_probe-base_probe:>+10.4f}")
    print()
    print(f"  KEY: input is ONLY ASR (video narration text), NO danmaku at test time.")
    print(f"  If learned > BGE: the projection learned to encode 'what evaluation this")
    print(f"  content triggers' — purely from content, without seeing audience reaction.")
    print()
    if learn_sil > base_sil and learn_probe > base_probe:
        print("  ✓ Evaluation signal successfully shaped CONTENT representation!")
        print("    Content embedding now encodes 'what reaction this triggers'")
        print("    without ever seeing evaluation keywords at test time.")
    elif learn_probe > base_probe:
        print("  ~ Partial success: probe improves but structure doesn't.")
    else:
        print("  ✗ Evaluation signal did not improve content representation.")
        print("    Possible causes: not enough data, or eval signal doesn't carry")
        print("    content-predictive information beyond what BGE already encodes.")

    out = {'baseline':{'silhouette':float(base_sil),'probe':float(base_probe)},
           'learned':{'silhouette':float(learn_sil),'probe':float(learn_probe)},
           'n_bins':len(bins),'n_train':len(train_emb),'n_test':len(test_emb),
           'valid_types':list(valid_types)}
    json.dump(out, open(os.path.join(_here,'../results/narrative/edrl_correct.json'),'w'), indent=2)
    print(f"\n  Saved: results/narrative/edrl_correct.json")


if __name__ == "__main__":
    main()
