"""Negative Control Experiment for EDRL
================================================================
The key scientific test: does dimension recurrence COLLAPSE toward null
when we destroy the danmaku-content relationship?

Three negative controls, each run through the IDENTICAL pipeline:
1. TOKEN-SHUFFLE: shuffle words within each comment (destroys semantics, keeps vocab)
2. CROSS-VIDEO SWAP: assign video A's comments to video B (destroys content link)
3. SYNTHETIC-GENERIC: replace comments with generic filler ("good", "nice", "ok"...)

If real recurrence (7 dims at 68-95%) >> control recurrence, the signal is real.
If controls also show high recurrence, we're just measuring LLM naming priors.

Usage:
  export DEEPSEEK_API_KEY=xxx
  export HF_HUB_OFFLINE=1
  python negative_control.py
"""
import os, sys, json, random, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, "D:/windows/code/project/research/memory/memory-engine/memory_engine")

from step1b_filter_and_recluster import load_danmaku, rule_filter, llm_classify_evaluative

def filter_evaluative(texts, client, max_n=150):
    """Apply the SAME evaluativeness filter as the main pipeline"""
    import random
    if len(texts) > max_n:
        texts = random.sample(texts, max_n)
    cls = llm_classify_evaluative(texts, client, batch_size=30)
    evaluative = [texts[i] for i, c in enumerate(cls) if i < len(texts) and c.get('eval', False)]
    return evaluative

def cluster_and_count_dims(texts, client, n_clusters=8):
    """Run the core pipeline: embed -> cluster -> name -> return dimension set"""
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    if len(texts) < 20:
        return []

    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    k = min(n_clusters, len(texts)//5)
    if k < 2:
        return []
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(emb)

    # Name each cluster
    dims = []
    for c in range(k):
        members = [texts[i] for i in range(len(texts)) if labels[i]==c]
        if len(members) < 3:
            continue
        sample = random.sample(members, min(8, len(members)))
        prompt = ("以下是一组视频弹幕聚类,用2-4个字概括评价维度:\n" +
                  "\n".join(f"- {s}" for s in sample) +
                  "\n只输出维度名称(如'剧情评价'/'视觉效果'),不要解释。")
        try:
            r = client.chat([{"role":"user","content":prompt}], temperature=0, max_tokens=20)
            dims.append(r['content'].strip().strip("'\"。"))
        except:
            pass
        time.sleep(0.3)
    return dims


def make_token_shuffle(danmaku):
    """Shuffle characters within each comment"""
    out = []
    for d in danmaku:
        chars = list(d['text'])
        random.shuffle(chars)
        out.append({'t': d['t'], 'text': ''.join(chars)})
    return out


def make_generic(danmaku):
    """Replace with generic filler comments"""
    fillers = ['好看','不错','厉害','哈哈','牛','可以','绝了','喜欢','一般','无聊',
               '有意思','太强了','支持','期待','感动','好家伙','离谱','笑死','泪目','666']
    return [{'t': d['t'], 'text': random.choice(fillers)} for d in danmaku]


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set"); sys.exit(1)
    from deepseek_client import DeepSeekClient
    client = DeepSeekClient()

    random.seed(42)

    # Use a subset of videos for the control (expensive: each needs LLM naming)
    data_dir = '../data/narrative'
    import glob
    videos = sorted(glob.glob(os.path.join(_here, data_dir, 'BV*.json')))
    videos = [v for v in videos if 'asr' not in v][:30]  # 30 videos for control

    print("="*60)
    print("  Negative Control Experiment (10 videos)")
    print("="*60)

    # Canonical dimension vocabulary (from main results)
    CANON = ['剧情','整体','细节','画面','情感','演技','节奏','音乐','叙事']
    def normalize_dim(d):
        for c in CANON:
            if c in d: return c
        return 'other'

    conditions = {
        'real': lambda dm: filter_evaluative([x['text'] for x in rule_filter(dm)], client),
        'token_shuffle': lambda dm: filter_evaluative([x['text'] for x in make_token_shuffle(rule_filter(dm))], client),
        'generic': lambda dm: filter_evaluative([x['text'] for x in make_generic(rule_filter(dm))], client),
    }

    # cross-video swap: handle separately
    all_danmaku = {}
    for vf in videos:
        dm, title, dur = load_danmaku(vf)
        all_danmaku[os.path.basename(vf)] = rule_filter(dm)

    results = {cond: {} for cond in conditions}
    results['cross_swap'] = {}

    from collections import Counter
    for cond, transform in conditions.items():
        print(f"\n  Condition: {cond}")
        dim_appear = Counter()
        n_vids = 0
        for vf in videos:
            dm, title, dur = load_danmaku(vf)
            texts = transform(dm)[:150]  # cap for cost
            if len(texts) < 20: continue
            dims = cluster_and_count_dims(texts, client)
            canon_dims = set(normalize_dim(d) for d in dims)
            for cd in canon_dims:
                if cd != 'other':
                    dim_appear[cd] += 1
            n_vids += 1
            print(f"    {os.path.basename(vf)[:14]}: {len(dims)} clusters -> {canon_dims}")
        results[cond] = {'n_videos': n_vids, 'dim_recurrence': {d: c/max(n_vids,1) for d,c in dim_appear.items()}}

    # Cross-video swap: shuffle which video's comments go where
    print(f"\n  Condition: cross_swap")
    keys = list(all_danmaku.keys())
    shuffled_keys = keys.copy()
    random.shuffle(shuffled_keys)
    dim_appear = Counter(); n_vids = 0
    for orig, swap in zip(keys, shuffled_keys):
        if orig == swap: continue
        texts = [x['text'] for x in all_danmaku[swap]][:150]
        if len(texts) < 20: continue
        dims = cluster_and_count_dims(texts, client)
        canon = set(normalize_dim(d) for d in dims)
        for cd in canon:
            if cd != 'other': dim_appear[cd] += 1
        n_vids += 1
    results['cross_swap'] = {'n_videos': n_vids, 'dim_recurrence': {d: c/max(n_vids,1) for d,c in dim_appear.items()}}

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS: Mean dimension recurrence by condition")
    print(f"{'='*60}")
    for cond in ['real','token_shuffle','generic','cross_swap']:
        rec = results[cond].get('dim_recurrence', {})
        # Mean recurrence of canonical dims
        canon_recs = [rec.get(c,0) for c in CANON]
        mean_rec = np.mean(canon_recs)
        top_dims = sorted(rec.items(), key=lambda x:-x[1])[:5]
        print(f"\n  {cond:15s}: mean canonical recurrence = {mean_rec:.0%}")
        print(f"    top: {[(d,f'{r:.0%}') for d,r in top_dims]}")

    with open(os.path.join(_here,'../results/narrative/negative_control.json'),'w',encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: results/narrative/negative_control.json")


if __name__ == "__main__":
    main()
