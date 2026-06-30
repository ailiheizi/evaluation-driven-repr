"""Batch run full EDRL pipeline on all videos
============================================================
For each video: step1b (filter + classify) → step2 (temporal alignment)
Collects all results into a unified cross-video analysis.

Usage:
  export DEEPSEEK_API_KEY=xxx
  export HF_HUB_OFFLINE=1
  python batch_pipeline.py [--max-classify 150] [--data-dir ../data/narrative]
"""
import os, sys, json, time, argparse, glob
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from step1b_filter_and_recluster import load_danmaku, rule_filter, llm_classify_evaluative
from step2_temporal_alignment import (classify_by_embedding_similarity, temporal_analysis,
                                       find_dimension_peaks)


def run_pipeline_single(video_path, results_dir, client, max_classify=150):
    """Run full pipeline on a single video, return summary dict"""
    basename = os.path.splitext(os.path.basename(video_path))[0]

    # Load
    danmaku, title, duration = load_danmaku(video_path)
    if len(danmaku) < 100:
        return {"bvid": basename, "title": title, "status": "too_few_raw", "raw_count": len(danmaku)}

    # Step 1b: Filter + Classify
    filtered = rule_filter(danmaku)
    if len(filtered) < 50:
        return {"bvid": basename, "title": title, "status": "too_few_filtered",
                "raw_count": len(danmaku), "filtered_count": len(filtered)}

    # LLM classify (sample)
    import random
    random.seed(42)
    sample_size = min(max_classify, len(filtered))
    sample_indices = sorted(random.sample(range(len(filtered)), sample_size))
    sample_texts = [filtered[i]['text'] for i in sample_indices]

    classifications = llm_classify_evaluative(sample_texts, client, batch_size=30)

    # Separate evaluative
    evaluative = []
    for idx, cls in zip(sample_indices, classifications):
        if cls.get('eval', False):
            evaluative.append({
                **filtered[idx],
                'dimension': cls.get('dim', 'unknown'),
                'polarity': cls.get('polarity', '0'),
            })

    eval_rate = len(evaluative) / max(len(classifications), 1)

    if len(evaluative) < 10:
        return {"bvid": basename, "title": title, "status": "too_few_evaluative",
                "raw_count": len(danmaku), "filtered_count": len(filtered),
                "evaluative_count": len(evaluative), "eval_rate": eval_rate}

    # Dimension distribution
    from collections import Counter
    dim_counts = Counter(e['dimension'] for e in evaluative)
    pol_counts = Counter(e['polarity'] for e in evaluative)

    # Save step1b result
    step1b_path = os.path.join(results_dir, f'eval_filtered_{basename}.json')
    with open(step1b_path, 'w', encoding='utf-8') as f:
        json.dump({
            "video": title, "bvid": basename,
            "total_raw": len(danmaku), "after_rule_filter": len(filtered),
            "sample_classified": len(classifications),
            "evaluative_count": len(evaluative), "evaluative_rate": eval_rate,
            "dimension_distribution": dict(dim_counts),
            "polarity_distribution": dict(pol_counts),
            "evaluative_samples": evaluative[:50],
        }, f, ensure_ascii=False, indent=2)

    # Step 2: Temporal alignment (use evaluative samples as reference)
    if len(evaluative) >= 15:
        try:
            texts = [d['text'] for d in filtered]
            pred_dims, pred_pols, confidences = classify_by_embedding_similarity(texts, evaluative)

            # Filter by confidence
            classified = []
            for i, (d, dim, pol, conf) in enumerate(zip(filtered, pred_dims, pred_pols, confidences)):
                if conf >= 0.6:
                    classified.append({'t': d['t'], 'text': d['text'],
                                      'dimension': dim, 'polarity': pol, 'confidence': float(conf)})

            if len(classified) >= 10:
                import numpy as np
                bins = temporal_analysis(classified, duration, bin_size=30)
                peaks, dim_timeseries = find_dimension_peaks(bins)

                # Compute entropy for each dimension
                dim_entropy = {}
                n_bins = len(bins)
                for dim, series in dim_timeseries.items():
                    arr = np.array(series)
                    arr = arr / max(arr.sum(), 1e-10)
                    entropy = -np.sum(arr[arr > 0] * np.log2(arr[arr > 0]))
                    max_entropy = np.log2(max(len(arr), 2))
                    dim_entropy[dim] = float(entropy / max(max_entropy, 1))

                # Save step2 result
                step2_path = os.path.join(results_dir, f'temporal_{basename}.json')
                with open(step2_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "video": title, "bvid": basename, "duration": duration,
                        "total_classified": len(classified),
                        "peaks": peaks, "dim_entropy": dim_entropy,
                    }, f, ensure_ascii=False, indent=2)

                return {
                    "bvid": basename, "title": title, "status": "ok",
                    "raw_count": len(danmaku), "filtered_count": len(filtered),
                    "evaluative_count": len(evaluative), "eval_rate": eval_rate,
                    "dimensions": dict(dim_counts), "polarity": dict(pol_counts),
                    "dim_entropy": dim_entropy, "peaks": peaks,
                    "duration": duration,
                }
        except Exception as e:
            print(f"    Step2 error: {e}")

    return {
        "bvid": basename, "title": title, "status": "step1b_only",
        "raw_count": len(danmaku), "filtered_count": len(filtered),
        "evaluative_count": len(evaluative), "eval_rate": eval_rate,
        "dimensions": dict(dim_counts), "polarity": dict(pol_counts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(_here, '..', 'data', 'narrative'))
    ap.add_argument("--results-dir", default=os.path.join(_here, '..', 'results', 'narrative'))
    ap.add_argument("--max-classify", type=int, default=150)
    args = ap.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set"); sys.exit(1)

    os.makedirs(args.results_dir, exist_ok=True)

    # Import DeepSeek client
    sys.path.insert(0, "D:/windows/code/project/research/memory/memory-engine/memory_engine")
    from deepseek_client import DeepSeekClient
    client = DeepSeekClient()

    # Find all video JSONs
    video_files = sorted(glob.glob(os.path.join(args.data_dir, 'BV*.json')))
    # Exclude ASR files
    video_files = [f for f in video_files if '_asr' not in f]

    print("=" * 70)
    print(f"  EDRL Batch Pipeline: {len(video_files)} videos")
    print("=" * 70)

    all_results = []
    for i, vf in enumerate(video_files):
        bvid = os.path.splitext(os.path.basename(vf))[0]
        print(f"\n  [{i+1}/{len(video_files)}] {bvid}...", flush=True)

        result = run_pipeline_single(vf, args.results_dir, client, args.max_classify)
        all_results.append(result)

        status = result['status']
        if status == 'ok':
            n_dims = len(result.get('dimensions', {}))
            top_dim = max(result.get('dimensions',{'?':0}), key=result['dimensions'].get) if result.get('dimensions') else '?'
            print(f"    OK: eval_rate={result['eval_rate']:.0%}, dims={n_dims}, top={top_dim}")
        else:
            print(f"    {status}")

    # Cross-video summary
    ok_results = [r for r in all_results if r['status'] == 'ok']

    print(f"\n{'=' * 70}")
    print(f"  Cross-Video Summary ({len(ok_results)}/{len(all_results)} successful)")
    print(f"{'=' * 70}")

    if ok_results:
        # Evaluative rate
        eval_rates = [r['eval_rate'] for r in ok_results]
        print(f"\n  Evaluative rate: mean={sum(eval_rates)/len(eval_rates):.0%}, "
              f"range=[{min(eval_rates):.0%}, {max(eval_rates):.0%}]")

        # Dimension frequency (how often each dimension appears across videos)
        from collections import Counter
        dim_freq = Counter()
        for r in ok_results:
            for dim in r.get('dimensions', {}):
                dim_freq[dim] += 1
        print(f"\n  Dimension stability (appears in N/{len(ok_results)} videos):")
        for dim, freq in dim_freq.most_common():
            print(f"    {dim:12s}: {freq}/{len(ok_results)} ({freq/len(ok_results):.0%})")

        # Entropy comparison
        print(f"\n  Temporal concentration by dimension (mean normalized entropy):")
        dim_entropies = {}
        for r in ok_results:
            for dim, ent in r.get('dim_entropy', {}).items():
                if dim not in dim_entropies:
                    dim_entropies[dim] = []
                dim_entropies[dim].append(ent)
        for dim, ents in sorted(dim_entropies.items(), key=lambda x: sum(x[1])/len(x[1])):
            mean_ent = sum(ents) / len(ents)
            conc = "★ CONCENTRATED" if mean_ent < 0.6 else "dispersed"
            print(f"    {dim:12s}: H={mean_ent:.2f} (n={len(ents)}) {conc}")

    # Save unified results
    unified_path = os.path.join(args.results_dir, '_cross_video_results.json')
    with open(unified_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_videos": len(all_results),
            "successful": len(ok_results),
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {unified_path}")


if __name__ == "__main__":
    main()
