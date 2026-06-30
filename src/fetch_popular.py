"""Fetch popular Bilibili videos with high danmaku count
============================================================
Strategy: Use Bilibili search/ranking API to find recent popular videos
with high danmaku counts across different genres.

Usage:
  python fetch_popular.py [--output-dir data/narrative] [--min-danmaku 1000]
"""
import os, sys, json, time, re, urllib.request, gzip, zlib, argparse

_here = os.path.dirname(os.path.abspath(__file__))

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
      "Accept-Encoding": "gzip, deflate",
      "Referer": "https://www.bilibili.com"}

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
            except: return zlib.decompress(raw, -zlib.MAX_WBITS)
        return raw
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

def search_videos(keyword, page=1):
    """Search Bilibili for videos by keyword, sorted by danmaku count"""
    import urllib.parse
    encoded = urllib.parse.quote(keyword)
    url = (f"https://api.bilibili.com/x/web-interface/search/type?"
           f"keyword={encoded}&search_type=video&order=dm&page={page}")
    data = get(url)
    if data is None:
        return []
    j = json.loads(data.decode("utf-8"))
    if j["code"] != 0:
        print(f"    Search error: {j.get('message','?')}")
        return []
    results = j.get("data", {}).get("result", [])
    videos = []
    for r in results:
        videos.append({
            "bvid": r.get("bvid", ""),
            "title": re.sub(r'<[^>]+>', '', r.get("title", "")),  # strip HTML tags
            "duration": r.get("duration", ""),
            "danmaku": r.get("video_review", 0),  # danmaku count from search
            "play": r.get("play", 0),
        })
    return videos

def get_video_info(bvid):
    data = get(f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}")
    if data is None:
        return None
    j = json.loads(data.decode("utf-8"))
    if j["code"] != 0:
        return None
    d = j["data"]
    return {"cid": d["cid"], "title": d["title"], "duration": d.get("duration", 0),
            "danmaku_count": d.get("stat", {}).get("danmaku", 0)}

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


# Genre keywords for search
GENRE_KEYWORDS = [
    ("film_analysis", "电影解说"),
    ("film_analysis", "影视解说 悬疑"),
    ("suspense_drama", "悬疑剧 推理"),
    ("suspense_drama", "日剧 推理"),
    ("gaming", "游戏实况 高能"),
    ("gaming", "原神 剧情"),
    ("science", "科普 硬核"),
    ("science", "物理 数学 讲解"),
    ("music", "翻唱 热门"),
    ("music", "原创音乐"),
    ("vlog", "vlog 旅行"),
    ("animation", "动漫解说"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=os.path.join(_here, '..', 'data', 'narrative'))
    ap.add_argument("--min-danmaku", type=int, default=1000)
    ap.add_argument("--max-per-genre", type=int, default=4)
    ap.add_argument("--target-total", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Fetch Popular Videos by Genre (search API)")
    print("=" * 60)

    collected = []
    genre_counts = {}

    for genre, keyword in GENRE_KEYWORDS:
        if genre_counts.get(genre, 0) >= args.max_per_genre:
            continue
        if len(collected) >= args.target_total:
            break

        print(f"\n  Searching: '{keyword}' ({genre})...")
        videos = search_videos(keyword)
        time.sleep(2)

        if not videos:
            print(f"    No results")
            continue

        for v in videos[:5]:  # Top 5 from search
            if genre_counts.get(genre, 0) >= args.max_per_genre:
                break
            if len(collected) >= args.target_total:
                break

            bvid = v['bvid']
            if not bvid:
                continue

            # Check if already fetched
            out_path = os.path.join(args.output_dir, f"{bvid}.json")
            if os.path.exists(out_path):
                print(f"    {bvid}: already cached")
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                collected.append({"bvid": bvid, "genre": genre, "status": "cached"})
                continue

            # Get detailed info
            info = get_video_info(bvid)
            time.sleep(1)
            if info is None:
                continue

            print(f"    {bvid}: {info['title'][:40]}... (dm={info['danmaku_count']})")

            # Fetch danmaku
            dms = fetch_danmaku(info['cid'])
            time.sleep(1)

            if len(dms) < args.min_danmaku:
                print(f"      Only {len(dms)} danmaku, skip")
                continue

            # Save
            data = {"bvid": bvid, "cid": info['cid'], "title": info['title'],
                    "duration": info['duration'], "genre": genre, "danmaku": dms}
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            print(f"      Saved: {len(dms)} danmaku")

            genre_counts[genre] = genre_counts.get(genre, 0) + 1
            collected.append({"bvid": bvid, "genre": genre, "title": info['title'],
                            "danmaku": len(dms), "duration": info['duration'], "status": "ok"})
            time.sleep(2)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Results: {len([c for c in collected if c['status'] in ('ok','cached')])} videos collected")
    print(f"{'=' * 60}")
    from collections import Counter
    for genre, count in Counter(c['genre'] for c in collected if c['status'] in ('ok','cached')).most_common():
        print(f"    {genre:20s}: {count}")

    # Save manifest
    manifest_path = os.path.join(args.output_dir, '_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(collected, f, ensure_ascii=False, indent=2)
    print(f"\n  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
