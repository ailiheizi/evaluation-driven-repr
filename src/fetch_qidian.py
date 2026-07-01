"""Fetch Qidian novel chapters with per-paragraph reviews
================================================================
Source: vipreader.qidian.com API
Book: 诡秘之主 (bookId: 1010868264)

For each chapter: get paragraph text + paragraph-level reader reviews.
This gives us X(content) ≠ Y(evaluation) pairs naturally.

Usage:
  python fetch_qidian.py [--chapters 50] [--book 1010868264]
"""
import os, sys, json, time, urllib.request, urllib.parse, re
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))

UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      'Referer': 'https://vipreader.qidian.com'}


def get(url, proxy=None):
    req = urllib.request.Request(url, headers=UA)
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({'https': proxy, 'http': proxy}))
    opener = urllib.request.build_opener(*handlers) if handlers else urllib.request
    try:
        with opener.open(req, timeout=20) as r:
            return r.read().decode('utf-8')
    except Exception as e:
        print(f'    GET error: {e}')
        return None


def get_chapter_list(book_id, proxy=None):
    """Get chapter list for a book"""
    url = f'https://read.qidian.com/ajax/book/category?bookId={book_id}'
    data = get(url, proxy)
    if not data:
        return []
    try:
        j = json.loads(data)
        chapters = []
        for vol in j.get('data', {}).get('vs', []):
            for ch in vol.get('cs', []):
                chapters.append({'id': ch['id'], 'name': ch['cN'], 'wordCount': ch.get('cnt', 0)})
        return chapters
    except:
        return []


def get_chapter_content(chapter_id, proxy=None):
    """Get chapter paragraph text (free chapters only)"""
    url = f'https://read.qidian.com/ajax/chapter/chapterInfo?bookId=1010868264&chapterId={chapter_id}'
    data = get(url, proxy)
    if not data:
        return []
    try:
        j = json.loads(data)
        content = j.get('data', {}).get('chapterInfo', {}).get('content', '')
        # Split into paragraphs (by <p> tags or newlines)
        paras = re.findall(r'<p>(.*?)</p>', content)
        if not paras:
            paras = [p.strip() for p in content.split('\n') if p.strip()]
        # Clean HTML
        paras = [re.sub(r'<[^>]+>', '', p).strip() for p in paras if len(re.sub(r'<[^>]+>', '', p).strip()) > 5]
        return paras
    except:
        return []


def get_paragraph_reviews(chapter_id, seg_id, proxy=None):
    """Get reviews for a specific paragraph"""
    url = (f'https://vipreader.qidian.com/ajax/chapterReview/reviewList?'
           f'bookId=1010868264&chapterId={chapter_id}&segmentId={seg_id}&type=1&page=1&pageSize=20')
    data = get(url, proxy)
    if not data:
        return []
    try:
        j = json.loads(data)
        reviews = []
        for r in j.get('data', {}).get('reviews', []):
            text = r.get('content', '').strip()
            if text:
                reviews.append(text)
        return reviews
    except:
        return []


def get_review_counts(chapter_id, proxy=None):
    """Get review count per paragraph for a chapter"""
    url = (f'https://vipreader.qidian.com/ajax/chapterReview/reviewNum?'
           f'bookId=1010868264&chapterId={chapter_id}')
    data = get(url, proxy)
    if not data:
        return {}
    try:
        j = json.loads(data)
        # Returns {segId: count}
        counts = {}
        for item in j.get('data', {}).get('reviewNums', []):
            counts[item['segmentId']] = item['num']
        return counts
    except:
        return {}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--book', default='1010868264', help='Book ID (default: 诡秘之主)')
    ap.add_argument('--chapters', type=int, default=30, help='How many chapters to fetch')
    ap.add_argument('--proxy', default='http://127.0.0.1:7890')
    ap.add_argument('--out', default=os.path.join(_here, '..', 'data', 'qidian'))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    proxy = args.proxy

    # Get chapter list
    print(f'Fetching chapter list for book {args.book}...')
    chapters = get_chapter_list(args.book, proxy)
    print(f'  Found {len(chapters)} chapters')
    if not chapters:
        print('  ERROR: could not fetch chapter list. Try without proxy or check API.')
        return

    # Process first N chapters
    chapters = chapters[:args.chapters]
    results = []

    for i, ch in enumerate(chapters):
        ch_id = ch['id']
        print(f'\n[{i+1}/{len(chapters)}] {ch["name"]} (id={ch_id})...')

        # Get review counts per paragraph
        counts = get_review_counts(ch_id, proxy)
        time.sleep(0.5)

        if not counts:
            print(f'  No review counts available, skip')
            continue

        # Get paragraph content
        paras = get_chapter_content(ch_id, proxy)
        time.sleep(0.5)

        if not paras:
            print(f'  Could not get content (VIP chapter?), skip')
            continue

        print(f'  {len(paras)} paragraphs, {len(counts)} with review counts')

        # Get reviews for paragraphs with most reviews (top 5 per chapter to save API calls)
        top_segs = sorted(counts.items(), key=lambda x: -x[1])[:5]
        ch_data = {
            'chapter_id': ch_id,
            'chapter_name': ch['name'],
            'paragraphs': paras,
            'review_counts': counts,
            'segments': []
        }

        for seg_id, count in top_segs:
            if count < 3:
                continue
            reviews = get_paragraph_reviews(ch_id, seg_id, proxy)
            time.sleep(0.5)

            # Match paragraph text (segId is 1-indexed)
            text = paras[seg_id - 1] if seg_id <= len(paras) else ''
            ch_data['segments'].append({
                'segId': seg_id,
                'text': text,
                'reviewCount': count,
                'reviews': reviews,
            })

        if ch_data['segments']:
            results.append(ch_data)
            print(f'  Got {len(ch_data["segments"])} segments with reviews')

        time.sleep(1)

    # Save
    out_path = os.path.join(args.out, 'novel_paired.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'book': '诡秘之主', 'bookId': args.book,
                   'chapters': len(results), 'data': results}, f, ensure_ascii=False, indent=1)
    print(f'\n=== Done: {len(results)} chapters with paired data ===')
    print(f'Saved: {out_path}')

    # Stats
    total_segs = sum(len(ch['segments']) for ch in results)
    total_reviews = sum(len(s['reviews']) for ch in results for s in ch['segments'])
    print(f'Total segments with reviews: {total_segs}')
    print(f'Total reviews collected: {total_reviews}')


if __name__ == '__main__':
    main()
