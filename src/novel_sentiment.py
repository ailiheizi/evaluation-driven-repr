"""Label paragraph review-sentiment, test if content predicts it
================================================================
For each paragraph: aggregate its reviews -> dominant reaction TYPE via LLM.
Then test: can paragraph CONTENT (X) predict review reaction type (Y)?

Reaction types (what readers DO in response):
- joke: 玩梗/吐槽/搞笑 (readers make jokes)
- awe: 震撼/赞叹文笔/牛 (impressed by writing/plot)
- emotion: 感动/泪目/心疼 (emotionally moved)
- analysis: 分析剧情/推测/考据 (analyze plot/theorize)
- confusion: 疑问/看不懂 (confused/questioning)

Usage:
  export DEEPSEEK_API_KEY=xxx; export HF_HUB_OFFLINE=1
  python novel_sentiment.py
"""
import os, sys, json, time, random
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "D:/windows/code/project/research/memory/memory-engine/memory_engine")

def main():
    from deepseek_client import DeepSeekClient
    client = DeepSeekClient()

    data = json.load(open(os.path.join(_here,'..','data','qidian','shengxu_reviews.json'),encoding='utf-8'))
    # Flatten segments
    segs = []
    for ci,ch in enumerate(data['chapters']):
        for s in ch['segments']:
            segs.append({'text':s['text'],'reviews':s['reviews'],'ch':ci})
    print(f'Segments: {len(segs)}')

    TYPES = ['joke','awe','emotion','analysis','confusion']
    # Label each segment's dominant reaction type via LLM (batch)
    labeled = []
    for i in range(0, len(segs), 15):
        batch = segs[i:i+15]
        lines = []
        for j,s in enumerate(batch):
            revs = ' | '.join(s['reviews'][:8])
            lines.append(f'{j+1}. 评论: {revs}')
        prompt = ('以下是小说段落的读者评论。判断每组评论的主要反应类型:\n'
                  'joke=玩梗/吐槽/搞笑, awe=震撼/赞文笔/觉得牛, emotion=感动/泪目/心疼, '
                  'analysis=分析剧情/推测/考据, confusion=疑问/看不懂\n\n'
                  + '\n'.join(lines) +
                  '\n\n每行输出JSON: {"id":序号,"type":"类型"}. 只输出JSON行。')
        try:
            r = client.chat([{"role":"user","content":prompt}], temperature=0, max_tokens=800)
            for line in r['content'].strip().split('\n'):
                line=line.strip()
                if line.startswith('{'):
                    try:
                        o=json.loads(line); idx=o['id']-1
                        if 0<=idx<len(batch) and o.get('type') in TYPES:
                            labeled.append({**batch[idx],'y':o['type']})
                    except: pass
        except Exception as e: print(f'  batch {i} err: {e}')
        if i%45==0: print(f'  labeled {len(labeled)}/{len(segs)}', flush=True)
        time.sleep(0.4)

    from collections import Counter
    dist = Counter(s['y'] for s in labeled)
    print(f'\nLabeled: {len(labeled)}')
    print(f'Type distribution: {dict(dist)}')

    # Now test: can content predict reaction type?
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

    tid={t:i for i,t in enumerate(TYPES)}
    # Keep types with >=15 examples
    valid={t for t,c in dist.items() if c>=15}
    data2=[s for s in labeled if s['y'] in valid]
    print(f'After filter (>=15/type): {len(data2)}, types={valid}')

    # Split by chapter
    chs=sorted(set(s['ch'] for s in data2)); random.seed(42); random.shuffle(chs)
    ntest=max(3,len(chs)//4); testch=set(chs[:ntest])
    train=[s for s in data2 if s['ch'] not in testch]; test=[s for s in data2 if s['ch'] in testch]

    tr_emb=model.encode([s['text'] for s in train],normalize_embeddings=True,show_progress_bar=False)
    te_emb=model.encode([s['text'] for s in test],normalize_embeddings=True,show_progress_bar=False)
    tr_y=np.array([tid[s['y']] for s in train]); te_y=np.array([tid[s['y']] for s in test])

    clf=LogisticRegression(max_iter=1000).fit(tr_emb,tr_y)
    acc=clf.score(te_emb,te_y)
    # majority baseline
    maj=Counter(te_y).most_common(1)[0][1]/len(te_y)
    print(f'\n=== Can content predict review reaction type? ===')
    print(f'  Train {len(train)}, Test {len(test)}')
    print(f'  Accuracy: {acc:.3f}  |  Majority baseline: {maj:.3f}  |  Chance: {1/len(valid):.3f}')
    print(f'  {"ABOVE majority: content has signal!" if acc>maj+0.05 else "~ majority: weak/no signal"}')

    json.dump({'labeled':len(labeled),'dist':dict(dist),'acc':float(acc),'majority':float(maj),
               'n_types':len(valid)}, open(os.path.join(_here,'..','results','narrative','novel_sentiment.json'),'w'),indent=2)
    print('\n  Saved: results/narrative/novel_sentiment.json')

if __name__=='__main__':
    main()
