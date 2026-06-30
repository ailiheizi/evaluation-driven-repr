"""Fixed permutation test + silhouette score for k selection
================================================================
Fixes:
1. Permutation test: shuffle DIMENSION LABELS (not timestamps)
   - Old: shuffle timestamps → proves temporal non-uniformity (trivial)
   - New: shuffle dimension labels → proves semantic dimensions cause concentration

2. Silhouette score: justify k selection for KMeans

Usage:
  export HF_HUB_OFFLINE=1
  python fixed_validation.py --data ../data/narrative/BV1LSoyYqEuU.json
"""
import os, sys, json, random, argparse
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from step1b_filter_and_recluster import load_danmaku, rule_filter


def load_reference(ref_path):
    """Load evaluative samples as reference"""
    with open(ref_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_dimension_entropy(classified_items, duration, bin_size=30):
    """Compute per-dimension normalized entropy"""
    n_bins = max(int(duration / bin_size), 1)
    dims = set(d['dimension'] for d in classified_items)

    dim_entropies = {}
    for dim in dims:
        times = [d['t'] for d in classified_items if d['dimension'] == dim]
        if len(times) < 5:
            continue
        hist, _ = np.histogram(times, bins=n_bins, range=(0, duration))
        hist = hist / max(hist.sum(), 1e-10)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        max_ent = np.log2(n_bins)
        dim_entropies[dim] = float(entropy / max(max_ent, 1))

    return dim_entropies


def fixed_permutation_test(classified_items, duration, n_perms=500, bin_size=30):
    """FIXED permutation test: shuffle DIMENSION LABELS, keep timestamps fixed.

    This tests: "Does the assignment of dimensions to specific times matter?"
    If real entropy < permuted entropy → dimensions are genuinely concentrated
    at specific times (not just because comments are bursty in general).
    """
    # Real entropies
    real_entropies = compute_dimension_entropy(classified_items, duration, bin_size)
    real_mean = np.mean(list(real_entropies.values())) if real_entropies else 1.0

    # Permutation: keep timestamps, shuffle dimension labels
    dims_list = [d['dimension'] for d in classified_items]

    perm_means = []
    for _ in range(n_perms):
        # Shuffle which dimension each item belongs to (keep time fixed)
        shuffled_dims = dims_list.copy()
        random.shuffle(shuffled_dims)

        # Create shuffled items
        shuffled_items = []
        for i, d in enumerate(classified_items):
            shuffled_items.append({'t': d['t'], 'dimension': shuffled_dims[i]})

        perm_entropies = compute_dimension_entropy(shuffled_items, duration, bin_size)
        if perm_entropies:
            perm_means.append(np.mean(list(perm_entropies.values())))

    # p-value: how often is permuted mean entropy <= real mean entropy
    p_value = sum(1 for pm in perm_means if pm <= real_mean) / max(len(perm_means), 1)

    return {
        "test": "dimension_label_shuffle",
        "description": "Shuffle dimension labels keeping timestamps fixed. Tests whether semantic dimension assignment causes temporal concentration.",
        "real_mean_entropy": float(real_mean),
        "perm_mean_entropy": float(np.mean(perm_means)) if perm_means else None,
        "perm_std": float(np.std(perm_means)) if perm_means else None,
        "p_value": float(p_value),
        "n_permutations": n_perms,
        "significant": p_value < 0.05,
        "per_dimension": real_entropies,
    }


def silhouette_analysis(embeddings, k_range=range(3, 15)):
    """Find optimal k using silhouette score"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    results = []
    for k in k_range:
        if k >= len(embeddings):
            break
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, metric='cosine', sample_size=min(5000, len(embeddings)))
        inertia = kmeans.inertia_
        results.append({"k": k, "silhouette": float(score), "inertia": float(inertia)})

    # Find best k
    best = max(results, key=lambda x: x['silhouette'])
    return results, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_here, '..', 'data', 'narrative', 'BV1LSoyYqEuU.json'))
    ap.add_argument("--reference", default=os.path.join(_here, '..', 'results', 'narrative', 'eval_filtered_dimensions.json'))
    ap.add_argument("--n-perms", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=" * 60)
    print("  Fixed Validation: Permutation Test + Silhouette Score")
    print("=" * 60)

    # Load data
    danmaku, title, duration = load_danmaku(args.data)
    print(f"\n  Video: {title} ({duration}s)")

    # Load reference evaluative samples
    with open(args.reference, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
    evaluative = ref_data.get('evaluative_samples', [])
    print(f"  Evaluative samples: {len(evaluative)}")

    # --- FIX 1: Silhouette Score for k selection ---
    print(f"\n  [1/2] Silhouette Analysis (k=3..14)...")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    eval_texts = [e['text'] for e in evaluative]
    embeddings = model.encode(eval_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    sil_results, best_k = silhouette_analysis(embeddings)
    print(f"  Results:")
    print(f"  {'k':>4} {'Silhouette':>12} {'Inertia':>10}")
    for r in sil_results:
        marker = " <-- BEST" if r['k'] == best_k['k'] else ""
        print(f"  {r['k']:>4} {r['silhouette']:>12.4f} {r['inertia']:>10.1f}{marker}")
    print(f"\n  Optimal k = {best_k['k']} (silhouette = {best_k['silhouette']:.4f})")

    # --- FIX 2: Fixed Permutation Test ---
    print(f"\n  [2/2] Fixed Permutation Test (n={args.n_perms})...")
    print(f"  Method: Shuffle DIMENSION LABELS, keep timestamps fixed")
    print(f"  This proves: dimension assignment causes concentration,")
    print(f"  not just bursty comment patterns.")

    # Use classified items from reference
    # Each has 't' and 'dimension'
    classified = [{'t': e['t'], 'dimension': e.get('dimension', 'unknown')} for e in evaluative]

    perm_result = fixed_permutation_test(classified, duration, args.n_perms)

    print(f"\n  Results:")
    print(f"    Real mean entropy:     {perm_result['real_mean_entropy']:.4f}")
    print(f"    Permuted mean entropy: {perm_result['perm_mean_entropy']:.4f} +/- {perm_result['perm_std']:.4f}")
    print(f"    p-value:               {perm_result['p_value']:.4f}")
    print(f"    Significant (p<0.05):  {'YES ***' if perm_result['significant'] else 'No'}")
    print(f"\n  Per-dimension entropy:")
    for dim, ent in sorted(perm_result['per_dimension'].items(), key=lambda x: x[1]):
        conc = "CONCENTRATED" if ent < 0.6 else "dispersed"
        print(f"    {dim:15s}: H={ent:.3f} ({conc})")

    # Also run old-style test for comparison
    print(f"\n  [Comparison] Old test (shuffle timestamps):")
    old_times = [d['t'] for d in classified]
    old_perm_entropies = []
    for _ in range(200):
        shuffled = old_times.copy()
        random.shuffle(shuffled)
        shuffled_items = [{'t': shuffled[i], 'dimension': classified[i]['dimension']} for i in range(len(classified))]
        ents = compute_dimension_entropy(shuffled_items, duration)
        if ents:
            old_perm_entropies.append(np.mean(list(ents.values())))
    old_real = np.mean(list(perm_result['per_dimension'].values()))
    old_p = sum(1 for pe in old_perm_entropies if pe <= old_real) / max(len(old_perm_entropies), 1)
    print(f"    Old p-value (timestamp shuffle): {old_p:.4f}")
    print(f"    New p-value (label shuffle):     {perm_result['p_value']:.4f}")
    print(f"    {'Both significant - temporal structure is real AND dimension-specific' if perm_result['significant'] and old_p < 0.05 else 'Check results carefully'}")

    # Save
    out_path = args.out or os.path.join(_here, '..', 'results', 'narrative', 'fixed_validation.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = {
        "silhouette_analysis": sil_results,
        "optimal_k": best_k,
        "permutation_test_fixed": perm_result,
        "old_permutation_p": old_p,
        "conclusion": "Both tests significant" if perm_result['significant'] else "Label shuffle not significant - needs investigation"
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
