"""Batch fetch danmaku for EDRL multi-genre validation
============================================================
Target: 20+ videos across different genres to validate cross-genre stability

Genres:
- Film analysis/review (电影解说) - 5 videos
- Suspense/mystery drama (悬疑剧) - 4 videos
- Gaming (游戏实况) - 3 videos
- Science/education (科普) - 3 videos
- Music/MV (音乐) - 3 videos
- Vlog/lifestyle (生活) - 2 videos

Usage:
  python batch_fetch_danmaku.py [--output-dir ../../data/narrative]
"""
import os, sys, json, time, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'paper', 'repo', 'src', 'narrative'))
# Also try direct import from same dir
_here = os.path.dirname(os.path.abspath(__file__))

# Inline fetch functions (no external deps needed)
import re, urllib.request, zlib, gzip

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
      "Accept-Encoding": "gzip, deflate"}

def get(url):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
            enc = r.headers.get("Content-Encoding", "")
        if enc == "gzip":
            return gzip.decompress(raw)
        if enc == "deflate":
            try: return zlib.decompress(raw)
            except zlib.error: return zlib.decompress(raw, -zlib.MAX_WBITS)
        return raw
    except Exception as e:
        print(f"    ERROR fetching {url}: {e}")
        return None

def get_cid_title(bvid):
    data = get(f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}")
    if data is None:
        return None, None, 0
    j = json.loads(data.decode("utf-8"))
    if j["code"] != 0:
        print(f"    API error for {bvid}: {j.get('message','?')}")
        return None, None, 0
    d = j["data"]
    return d["cid"], d["title"], d.get("duration", 0)

def fetch_danmaku(cid):
    xml_data = get(f"https://comment.bilibili.com/{cid}.xml")
    if xml_data is None:
        return []
    xml = xml_data.decode("utf-8")
    dms = []
    for m in re.finditer(r'<d p="([^"]+)">([^<]*)</d>', xml):
        attrs = m.group(1).split(",")
        t = float(attrs[0])
        text = m.group(2)
        dms.append({"t": t, "text": text})
    dms.sort(key=lambda x: x["t"])
    return dms


# Video list by genre
# Selected for: high danmaku count, varied content, popular videos
VIDEOS = [
    # Film analysis (电影解说) - already have BV1LSoyYqEuU
    {"bvid": "BV1bx411c7Qt", "genre": "film_analysis", "note": "movie explanation popular"},
    {"bvid": "BV1aS4y1P7bv", "genre": "film_analysis", "note": "movie review"},
    {"bvid": "BV1GJ411x7h7", "genre": "film_analysis", "note": "thriller analysis"},
    {"bvid": "BV1Ks411b7XK", "genre": "film_analysis", "note": "classic film"},

    # Suspense drama - already have BV1XHmwYpE8s
    {"bvid": "BV1xW411n7GS", "genre": "suspense_drama", "note": "mystery drama clip"},
    {"bvid": "BV1rt411x7Ue", "genre": "suspense_drama", "note": "detective drama"},
    {"bvid": "BV1KW411M7yQ", "genre": "suspense_drama", "note": "crime drama"},

    # Gaming
    {"bvid": "BV1GJ411x7h7", "genre": "gaming", "note": "game highlights"},
    {"bvid": "BV1Wx411f7VY", "genre": "gaming", "note": "esports commentary"},
    {"bvid": "BV1ms411A7Ey", "genre": "gaming", "note": "game walkthrough"},

    # Science/education
    {"bvid": "BV1kx411S7ZZ", "genre": "science", "note": "popular science"},
    {"bvid": "BV1Wx411L7Ny", "genre": "science", "note": "tech explanation"},
    {"bvid": "BV1GJ411B7vA", "genre": "science", "note": "documentary style"},

    # Music/MV
    {"bvid": "BV1Ws411o7JR", "genre": "music", "note": "music video"},
    {"bvid": "BV1xs411Q7Nz", "genre": "music", "note": "cover song"},
    {"bvid": "BV1Gx411w7RJ", "genre": "music", "note": "original composition"},

    # Vlog/lifestyle
    {"bvid": "BV1ms411M7Kp", "genre": "vlog", "note": "daily vlog"},
    {"bvid": "BV1Ks411Q7jy", "genre": "vlog", "note": "travel vlog"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=os.path.join(_here, '..', 'data', 'narrative'))
    ap.add_argument("--min-danmaku", type=int, default=500, help="Skip videos with fewer danmaku")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Batch Danmaku Fetcher for EDRL")
    print("=" * 60)

    results = []
    for i, v in enumerate(VIDEOS):
        bvid = v['bvid']
        genre = v['genre']
        print(f"\n  [{i+1}/{len(VIDEOS)}] {bvid} ({genre})...")

        # Check if already fetched
        out_path = os.path.join(args.output_dir, f"{bvid}.json")
        if os.path.exists(out_path):
            print(f"    Already exists, skip")
            with open(out_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            results.append({"bvid": bvid, "genre": genre, "title": d.get("title","?"),
                           "danmaku": len(d.get("danmaku",[])), "status": "cached"})
            continue

        cid, title, dur = get_cid_title(bvid)
        if cid is None:
            print(f"    FAILED to get video info")
            results.append({"bvid": bvid, "genre": genre, "status": "failed"})
            time.sleep(2)
            continue

        print(f"    Title: {title}")
        print(f"    Duration: {dur}s, CID: {cid}")

        dms = fetch_danmaku(cid)
        print(f"    Danmaku: {len(dms)}")

        if len(dms) < args.min_danmaku:
            print(f"    Too few danmaku ({len(dms)} < {args.min_danmaku}), skip")
            results.append({"bvid": bvid, "genre": genre, "title": title,
                           "danmaku": len(dms), "status": "too_few"})
            time.sleep(2)
            continue

        # Save
        data = {"bvid": bvid, "cid": cid, "title": title, "duration": dur,
                "genre": genre, "danmaku": dms}
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"    Saved: {out_path}")

        results.append({"bvid": bvid, "genre": genre, "title": title,
                       "danmaku": len(dms), "duration": dur, "status": "ok"})
        time.sleep(3)  # Rate limiting

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    ok = [r for r in results if r['status'] in ('ok', 'cached')]
    failed = [r for r in results if r['status'] == 'failed']
    few = [r for r in results if r['status'] == 'too_few']
    print(f"  Success: {len(ok)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Too few: {len(few)}")
    print(f"\n  By genre:")
    from collections import Counter
    genre_counts = Counter(r['genre'] for r in ok)
    for g, c in genre_counts.most_common():
        print(f"    {g:20s}: {c}")

    # Save manifest
    manifest_path = os.path.join(args.output_dir, '_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
