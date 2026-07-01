"""Batch download audio + ASR transcription for EDRL
================================================================
Downloads B站 video audio via yt-dlp, transcribes with faster-whisper.
Selects content-rich videos (film/science/animation, skip pure music).

Usage:
  python batch_asr.py [--n 20] [--skip-download]
"""
import os, sys, json, glob, time, subprocess
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(_here, '..', 'audio')
DATA_DIR = os.path.join(_here, '..', 'data', 'narrative')
COOKIES = os.path.join(_here, '..', 'bili_cookies.txt')

# Skip music-heavy videos (ASR won't get much content)
SKIP_GENRES = {'music'}

def get_videos_needing_asr(max_n=20):
    """Find videos with danmaku but no ASR yet"""
    videos = []
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.startswith('BV') or not f.endswith('.json') or '_asr' in f:
            continue
        bvid = f.replace('.json', '')
        asr_path = os.path.join(DATA_DIR, f'{bvid}_asr.json')
        if os.path.exists(asr_path):
            continue
        # Check genre
        with open(os.path.join(DATA_DIR, f), 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        genre = data.get('genre', '')
        if genre in SKIP_GENRES:
            continue
        dm_count = len(data.get('danmaku', []))
        if dm_count < 500:
            continue
        videos.append({'bvid': bvid, 'genre': genre, 'title': data.get('title','')[:30], 'dm': dm_count})
    return videos[:max_n]


def download_audio(bvid):
    """Download audio for a video using yt-dlp"""
    out_path = os.path.join(AUDIO_DIR, f'{bvid}.m4a')
    if os.path.exists(out_path):
        return out_path
    cmd = [
        'yt-dlp', '--proxy', 'http://127.0.0.1:7890',
        '--cookies', COOKIES,
        '-f', 'bestaudio', '-x', '--audio-format', 'm4a',
        '-o', out_path,
        f'https://www.bilibili.com/video/{bvid}',
        '--no-check-certificates', '-q'
    ]
    try:
        subprocess.run(cmd, timeout=120, capture_output=True)
        if os.path.exists(out_path):
            return out_path
    except Exception as e:
        print(f'    download error: {e}')
    return None


def transcribe(audio_path, bvid):
    """Transcribe audio with faster-whisper"""
    from faster_whisper import WhisperModel
    # Load model (cached after first call)
    if not hasattr(transcribe, '_model'):
        print('  Loading whisper-small (first time)...')
        transcribe._model = WhisperModel('small', device='cpu', compute_type='int8')

    segments, info = transcribe._model.transcribe(
        audio_path, language='zh', beam_size=3,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500)
    )
    results = []
    for seg in segments:
        results.append({'start': round(seg.start, 2), 'end': round(seg.end, 2), 'text': seg.text.strip()})

    # Save
    out_path = os.path.join(DATA_DIR, f'{bvid}_asr.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'segments': results}, f, ensure_ascii=False, indent=1)
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--skip-download', action='store_true')
    args = ap.parse_args()

    os.makedirs(AUDIO_DIR, exist_ok=True)

    videos = get_videos_needing_asr(args.n)
    print(f'Videos needing ASR: {len(videos)}')
    for v in videos[:5]:
        print(f'  {v["bvid"]} ({v["genre"]}): {v["title"]}... ({v["dm"]} dm)')

    ok = 0
    for i, v in enumerate(videos):
        bvid = v['bvid']
        print(f'\n[{i+1}/{len(videos)}] {bvid} ({v["genre"]}): {v["title"]}...')

        # Download
        if not args.skip_download:
            audio = download_audio(bvid)
            if not audio:
                print(f'  SKIP (download failed)')
                continue
            print(f'  Downloaded: {os.path.getsize(audio)/1e6:.1f}MB')
            time.sleep(2)
        else:
            audio = os.path.join(AUDIO_DIR, f'{bvid}.m4a')
            if not os.path.exists(audio):
                print(f'  SKIP (no audio file)')
                continue

        # Transcribe
        print(f'  Transcribing...')
        try:
            segs = transcribe(audio, bvid)
            print(f'  ASR done: {len(segs)} segments')
            ok += 1
        except Exception as e:
            print(f'  ASR error: {e}')

    print(f'\n=== Done: {ok}/{len(videos)} videos transcribed ===')


if __name__ == '__main__':
    main()
