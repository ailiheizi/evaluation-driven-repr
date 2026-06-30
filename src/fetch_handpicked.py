"""Fetch danmaku from hand-picked popular Bilibili videos
============================================================
Hand-selected BV IDs with known high danmaku counts across genres.
Using view API (works) + danmaku XML (works).

Usage:
  python fetch_handpicked.py [--output-dir data/narrative]
"""
import os, sys, json, time, re, urllib.request, gzip, zlib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

_here = os.path.dirname(os.path.abspath(__file__))

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
      "Referer": "https://www.bilibili.com"}

def get(url):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
            enc = r.headers.get("Content-Encoding", "")
        if enc == "gzip": return gzip.decompress(raw)
        if enc == "deflate":
            try: return zlib.decompress(raw)
            except: return zlib.decompress(raw, -zlib.MAX_WBITS)
        return raw
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

def get_video_info(bvid):
    data = get(f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}")
    if data is None: return None
    j = json.loads(data.decode("utf-8"))
    if j["code"] != 0:
        print(f"    API error {bvid}: {j.get('message','?')}")
        return None
    d = j["data"]
    # Handle multi-part videos (use first page)
    cid = d.get("cid") or d.get("pages", [{}])[0].get("cid")
    if not cid:
        print(f"    No CID found for {bvid}")
        return None
    return {"cid": cid, "title": d["title"], "duration": d.get("duration", 0),
            "danmaku_count": d.get("stat", {}).get("danmaku", 0)}

def fetch_danmaku(cid):
    xml_data = get(f"https://comment.bilibili.com/{cid}.xml")
    if xml_data is None: return []
    xml = xml_data.decode("utf-8")
    dms = []
    for m in re.finditer(r'<d p="([^"]+)">([^<]*)</d>', xml):
        attrs = m.group(1).split(",")
        t = float(attrs[0])
        text = m.group(2)
        dms.append({"t": t, "text": text})
    dms.sort(key=lambda x: x["t"])
    return dms

# Hand-picked videos: popular, diverse genres, high danmaku
VIDEOS = [
    # Film analysis / Movie review
    ("BV1uT4y1P7CX", "film_analysis"),   # Rick Astley 4K (meta/meme, 191k dm)
    ("BV1x54y1e7zf", "film_analysis"),   # Popular movie analysis
    ("BV1GM4m1d7Bv", "film_analysis"),   # Recent movie review
    ("BV1Hx411w7X3", "film_analysis"),   # Classic film review

    # Drama / TV show clips
    ("BV1bK4y1C7jX", "drama"),           # Popular drama clip
    ("BV1QJ411W7zM", "drama"),           # TV drama highlight
    ("BV1s5411Y7qM", "drama"),           # Japanese drama

    # Gaming
    ("BV1CX4y1N7HU", "gaming"),          # Genshin Impact
    ("BV1GJ411x7h7", "gaming"),          # Game highlights
    ("BV1fD4y1A72w", "gaming"),          # Popular game content

    # Science / Education
    ("BV1kx411S7ZZ", "science"),         # Popular science
    ("BV1R54y1k7Yv", "science"),         # Physics explanation
    ("BV1Y54y1q7yR", "science"),         # Tech review

    # Music
    ("BV1uT4y1P7CX", "music"),           # Already fetched above (skip)
    ("BV1Ab4y1k7NX", "music"),           # Cover song
    ("BV1HK4y1k7Rz", "music"),          # Original music

    # Animation
    ("BV1JE411N7JL", "animation"),       # Anime review
    ("BV1T5411m7Xz", "animation"),       # Animation clip

    # Vlog / Lifestyle
    ("BV1wy4y1D7Mc", "vlog"),            # Popular vlog
    ("BV1rp4y1e7KZ", "vlog"),            # Travel content
]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=os.path.join(_here, '..', 'data', 'narrative'))
    ap.add_argument("--min-danmaku", type=int, default=500)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Fetching hand-picked popular videos")
    print("=" * 60)

    results = []
    seen_bvids = set()

    for bvid, genre in VIDEOS:
        if bvid in seen_bvids:
            continue
        seen_bvids.add(bvid)

        out_path = os.path.join(args.output_dir, f"{bvid}.json")
        if os.path.exists(out_path):
            print(f"  [{bvid}] cached")
            with open(out_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            results.append({"bvid": bvid, "genre": genre, "title": d.get("title","?"),
                           "danmaku": len(d.get("danmaku",[])), "status": "cached"})
            continue

        print(f"\n  [{bvid}] ({genre})...", end=" ", flush=True)
        info = get_video_info(bvid)
        time.sleep(1)

        if info is None:
            results.append({"bvid": bvid, "genre": genre, "status": "failed"})
            continue

        print(f"{info['title'][:35]}... (dm_stat={info['danmaku_count']})")

        dms = fetch_danmaku(info['cid'])
        time.sleep(1.5)

        print(f"    fetched {len(dms)} danmaku", end="")
        if len(dms) < args.min_danmaku:
            print(f" (< {args.min_danmaku}, skip)")
            results.append({"bvid": bvid, "genre": genre, "title": info['title'],
                           "danmaku": len(dms), "status": "too_few"})
            continue

        # Save
        data = {"bvid": bvid, "cid": info['cid'], "title": info['title'],
                "duration": info['duration'], "genre": genre, "danmaku": dms}
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f" -> saved!")

        results.append({"bvid": bvid, "genre": genre, "title": info['title'],
                       "danmaku": len(dms), "duration": info['duration'], "status": "ok"})
        time.sleep(2)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    ok = [r for r in results if r['status'] in ('ok', 'cached')]
    print(f"  Total collected: {len(ok)}")
    from collections import Counter
    for genre, count in Counter(r['genre'] for r in ok).most_common():
        vids = [r for r in ok if r['genre'] == genre]
        dm_total = sum(r.get('danmaku', 0) for r in vids)
        print(f"    {genre:20s}: {count} videos, {dm_total:>6d} danmaku")

    # Update manifest
    manifest_path = os.path.join(args.output_dir, '_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
