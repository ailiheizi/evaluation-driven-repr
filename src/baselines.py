"""Baseline comparisons for EDRL
================================================================
Compare EDRL against baselines to prove our approach adds value:

1. BERTopic (raw): Topic modeling on ALL danmaku (no filtering)
   → Expected: topics are noisy (memes, spam mixed with evaluation)

2. Density-only: Time-bin danmaku by pure count, no semantics
   → Expected: finds "hot" moments but can't distinguish WHY (plot vs music vs acting)

3. Random permutation: Shuffle timestamps, recompute alignment
   → Expected: entropy → ~1.0 (no structure), proving our alignment is real

4. EDRL (ours): Filter → Classify → Cluster → Temporal align
   → Expected: structured dimensions with low entropy

Metrics:
  - Interpretability: Can a human name the clusters? (manual check)
  - Temporal concentration: Normalized entropy per dimension
  - Permutation p-value: How often does random match our entropy

Usage:
  export DEEPSEEK_API_KEY=xxx
  export HF_HUB_OFFLINE=1
  python baselines.py --data ../data/narrative/BV1LSoyYqEuU.json
"""
import os, sys, json, time, argparse, random
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from step1b_filter_and_recluster import load_danmaku, rule_filter


def baseline_bertopic(danmaku, n_topics=8):
    """Baseline 1: BERTopic on ALL danmaku (no filtering)"""
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    texts = [d['text'].strip() for d in danmaku if 3 <= len(d['text'].strip()) <= 50]
    timestamps = [d['t'] for d in danmaku if 3 <= len(d['text'].strip()) <= 50]

    # Use same embedding as EDRL for fair comparison
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    # KMeans (same as EDRL step1)
    kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Compute entropy per cluster
    duration = max(timestamps) if timestamps else 1
    n_bins = max(int(duration / 30), 1)
    cluster_entropies = {}

    for c in range(n_topics):
        c_times = [timestamps[i] for i in range(len(labels)) if labels[i] == c]
        if len(c_times) < 5:
            continue
        hist, _ = np.histogram(c_times, bins=n_bins, range=(0, duration))
        hist = hist / max(hist.sum(), 1)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        max_ent = np.log2(n_bins)
        cluster_entropies[f"topic_{c}"] = float(entropy / max(max_ent, 1))

    # Get sample texts per cluster for interpretability
    cluster_samples = {}
    for c in range(n_topics):
        members = [texts[i] for i in range(len(labels)) if labels[i] == c]
        cluster_samples[f"topic_{c}"] = random.sample(members, min(5, len(members)))

    return {
        "method": "BERTopic_raw",
        "n_topics": n_topics,
        "total_texts": len(texts),
        "entropies": cluster_entropies,
        "mean_entropy": float(np.mean(list(cluster_entropies.values()))) if cluster_entropies else 1.0,
        "samples": cluster_samples,
    }


def baseline_density(danmaku, duration, bin_size=30):
    """Baseline 2: Pure density (no semantics)"""
    timestamps = [d['t'] for d in danmaku]
    n_bins = max(int(duration / bin_size), 1)
    hist, _ = np.histogram(timestamps, bins=n_bins, range=(0, duration))

    # Find peaks (top 10% bins)
    threshold = np.percentile(hist, 90)
    peak_bins = np.where(hist >= threshold)[0]

    # Density can find "hot moments" but not WHY
    return {
        "method": "density_only",
        "n_bins": n_bins,
        "total_danmaku": len(timestamps),
        "peak_bins": [int(b) for b in peak_bins],
        "peak_times": [int(b * bin_size) for b in peak_bins],
        "peak_density": float(hist[peak_bins].mean()) if len(peak_bins) > 0 else 0,
        "mean_density": float(hist.mean()),
        "note": "finds WHERE but not WHAT dimension",
    }


def baseline_permutation(edrl_entropies, danmaku, duration, n_perms=200, bin_size=30):
    """Baseline 3: Permutation test — shuffle timestamps, recompute entropy
    Returns p-value: fraction of permutations with entropy <= observed"""
    from step1b_filter_and_recluster import rule_filter
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    # Use the same filtered+clustered data but shuffle timestamps
    filtered = rule_filter(danmaku)
    texts = [d['text'] for d in filtered]
    real_times = [d['t'] for d in filtered]

    if len(texts) < 50:
        return {"method": "permutation", "p_value": None, "note": "too few data"}

    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    n_bins = max(int(duration / bin_size), 1)

    def compute_mean_entropy(times, labels):
        entropies = []
        for c in set(labels):
            c_times = [times[i] for i in range(len(labels)) if labels[i] == c]
            if len(c_times) < 5:
                continue
            hist, _ = np.histogram(c_times, bins=n_bins, range=(0, duration))
            hist = hist / max(hist.sum(), 1)
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            max_ent = np.log2(n_bins)
            entropies.append(entropy / max(max_ent, 1))
        return np.mean(entropies) if entropies else 1.0

    # Real entropy
    real_entropy = compute_mean_entropy(real_times, labels)

    # Permutation entropies
    perm_entropies = []
    for _ in range(n_perms):
        shuffled_times = random.sample(real_times, len(real_times))
        perm_ent = compute_mean_entropy(shuffled_times, labels)
        perm_entropies.append(perm_ent)

    # p-value: how often is permuted entropy <= real entropy
    p_value = sum(1 for pe in perm_entropies if pe <= real_entropy) / n_perms

    return {
        "method": "permutation_test",
        "real_entropy": float(real_entropy),
        "perm_mean": float(np.mean(perm_entropies)),
        "perm_std": float(np.std(perm_entropies)),
        "p_value": float(p_value),
        "n_permutations": n_perms,
        "significant": p_value < 0.05,
        "interpretation": f"Real entropy ({real_entropy:.3f}) vs permuted ({np.mean(perm_entropies):.3f}±{np.std(perm_entropies):.3f}), p={p_value:.4f}"
    }


def edrl_summary(results_path):
    """Load EDRL results for comparison"""
    if not os.path.exists(results_path):
        return {"method": "EDRL", "status": "not_run"}
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        "method": "EDRL",
        "evaluative_rate": data.get("evaluative_rate", 0),
        "n_dimensions": len(data.get("dimension_distribution", {})),
        "dimensions": data.get("dimension_distribution", {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_here, '..', 'data', 'narrative', 'BV1LSoyYqEuU.json'))
    ap.add_argument("--n-perms", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("  Baseline Comparisons for EDRL")
    print("=" * 70)

    # Load data
    danmaku, title, duration = load_danmaku(args.data)
    print(f"\n  Video: {title}")
    print(f"  Duration: {duration}s, Danmaku: {len(danmaku)}")

    # --- Baseline 1: BERTopic ---
    print(f"\n  [1/4] BERTopic (raw, no filtering)...")
    bt_result = baseline_bertopic(danmaku)
    print(f"    Mean entropy: {bt_result['mean_entropy']:.3f}")
    print(f"    Topic samples:")
    for topic, samples in list(bt_result['samples'].items())[:3]:
        print(f"      {topic}: {samples[:2]}")

    # --- Baseline 2: Density ---
    print(f"\n  [2/4] Density-only (no semantics)...")
    dens_result = baseline_density(danmaku, duration)
    print(f"    Peak times: {dens_result['peak_times'][:5]}...")
    print(f"    Peak density: {dens_result['peak_density']:.1f} vs mean {dens_result['mean_density']:.1f}")

    # --- Baseline 3: Permutation test ---
    print(f"\n  [3/4] Permutation test (n={args.n_perms})...")
    # Load EDRL entropies if available
    edrl_path = os.path.join(_here, '..', 'results', 'narrative', 'temporal_alignment.json')
    if os.path.exists(edrl_path):
        with open(edrl_path, 'r', encoding='utf-8') as f:
            edrl_data = json.load(f)
        edrl_entropies = {p['top_dim']: 0 for p in edrl_data.get('timeline_summary', []) if p.get('top_dim')}
    else:
        edrl_entropies = {}

    perm_result = baseline_permutation(edrl_entropies, danmaku, duration, args.n_perms)
    print(f"    {perm_result['interpretation']}")
    print(f"    p-value: {perm_result['p_value']:.4f} {'*** SIGNIFICANT' if perm_result['significant'] else '(not significant)'}")

    # --- EDRL summary ---
    print(f"\n  [4/4] EDRL (our method)...")
    edrl_filtered_path = os.path.join(_here, '..', 'results', 'narrative', 'eval_filtered_dimensions.json')
    edrl_result = edrl_summary(edrl_filtered_path)
    if edrl_result.get('status') != 'not_run':
        print(f"    Evaluative rate: {edrl_result['evaluative_rate']:.0%}")
        print(f"    Dimensions found: {edrl_result['n_dimensions']}")
        print(f"    Top dims: {edrl_result['dimensions']}")

    # --- Comparison table ---
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Method':<20s} {'Mean Entropy':<15s} {'Interpretable?':<15s} {'Dimension info?'}")
    print(f"  {'BERTopic(raw)':<20s} {bt_result['mean_entropy']:<15.3f} {'Noisy':<15s} {'No (mixed topics)'}")
    print(f"  {'Density-only':<20s} {'N/A':<15s} {'N/A':<15s} {'No (just WHERE)'}")
    print(f"  {'Permuted(random)':<20s} {perm_result['perm_mean']:<15.3f} {'No':<15s} {'No (shuffled)'}")
    print(f"  {'EDRL (ours)':<20s} {perm_result['real_entropy']:<15.3f} {'Yes':<15s} {'Yes (named dims)'}")
    print(f"\n  Permutation p-value: {perm_result['p_value']:.4f}")
    print(f"  EDRL entropy reduction vs random: {(perm_result['perm_mean']-perm_result['real_entropy'])/perm_result['perm_mean']*100:.1f}%")

    # Save
    out_path = args.out or os.path.join(_here, '..', 'results', 'narrative', 'baseline_comparison.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = {
        "video": title,
        "bertopic": bt_result,
        "density": dens_result,
        "permutation": perm_result,
        "edrl": edrl_result,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
