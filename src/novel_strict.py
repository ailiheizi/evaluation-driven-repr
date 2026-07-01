"""Strict filtering: remove joke/spam reviews, keep only content-evaluative ones
================================================================
Hypothesis: joke reviews (63%) are noise. If we filter to ONLY reviews that
genuinely evaluate content (awe/emotion/analysis of the actual text), does
content then predict reaction?

Two-stage:
1. Review-level filter: LLM classifies each review as evaluative vs joke/spam
2. Keep segments where >=3 evaluative reviews remain
3. Re-test: content -> aggregated evaluative reaction

Usage:
  export DEEPSEEK_API_KEY=xxx; export HF_HUB_OFFLINE=1
  python novel_strict.py
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

    segs=[]
    for ci,ch in enumerate(data['chapters']):
        for s in ch['segments']:
            segs.append({'text':s['text'],'reviews':s['reviews'],'ch':ci})

    # === Stage 1: filter reviews — keep only content-evaluative ===
    # Collect all reviews, classify in batches
    all_reviews=[]
    for si,s in enumerate(segs):
        for r in s['reviews']:
            all_reviews.append({'seg':si,'text':r})
    print(f'Total reviews: {len(all_reviews)}')

    # Classify: is this review evaluating the CONTENT (plot/writing/character/emotion)
    # vs joke/meme/spam/off-topic?
    evaluative_flags={}
    for i in range(0,len(all_reviews),25):
        batch=all_reviews[i:i+25]
        lines=[f'{j+1}. {r["text"][:50]}' for j,r in enumerate(batch)]
        prompt=('判断每条小说段落评论是否在【认真评价内容】(评价剧情/文笔/人物/情节/被内容打动),'
                '还是【玩梗灌水】(玩梗/接龙/吐槽/搞笑/无关/单字).\n\n'
                +'\n'.join(lines)+
                '\n\n每行输出: {"id":N,"eval":true/false}. eval=true只给认真评价内容的. 只输出JSON行.')
        try:
            r=client.chat([{"role":"user","content":prompt}],temperature=0,max_tokens=700)
            for line in r['content'].strip().split('\n'):
                line=line.strip()
                if line.startswith('{'):
                    try:
                        o=json.loads(line); idx=i+o['id']-1
                        if 0<=idx<len(all_reviews): evaluative_flags[idx]=o.get('eval',False)
                    except: pass
        except Exception as e: print(f'  err {i}: {e}')
        if i%100==0: print(f'  classified {i}/{len(all_reviews)}',flush=True)
        time.sleep(0.3)

    n_eval=sum(1 for v in evaluative_flags.values() if v)
    print(f'\nEvaluative reviews: {n_eval}/{len(evaluative_flags)} = {n_eval/max(len(evaluative_flags),1):.0%}')

    # Rebuild segments with only evaluative reviews
    for si,s in enumerate(segs):
        s['eval_reviews']=[all_reviews[i]['text'] for i in range(len(all_reviews))
                           if all_reviews[i]['seg']==si and evaluative_flags.get(i,False)]
    strict_segs=[s for s in segs if len(s['eval_reviews'])>=3]
    print(f'Segments with >=3 evaluative reviews: {len(strict_segs)}')

    if len(strict_segs)<40:
        print('Too few segments after strict filter. Signal test not reliable.')
        print('CONCLUSION: evaluative reviews too sparse — most reviews are jokes/spam.')
        json.dump({'total_reviews':len(all_reviews),'evaluative':n_eval,
                   'eval_rate':n_eval/max(len(evaluative_flags),1),
                   'strict_segments':len(strict_segs),'conclusion':'too_sparse'},
                  open(os.path.join(_here,'..','results','narrative','novel_strict.json'),'w'),indent=2)
        return

    # === Stage 2: label reaction type on STRICT (evaluative-only) reviews ===
    TYPES=['awe','emotion','analysis','critique']
    labeled=[]
    for i in range(0,len(strict_segs),15):
        batch=strict_segs[i:i+15]
        lines=[f'{j+1}. 评论: {" | ".join(s["eval_reviews"][:6])}' for j,s in enumerate(batch)]
        prompt=('这些是【已筛选的认真评价】。判断每组主要反应:\n'
                'awe=震撼/赞文笔/觉得牛, emotion=感动/泪目/心疼, '
                'analysis=分析剧情/推测/考据, critique=批评/挑刺/觉得不合理\n\n'
                +'\n'.join(lines)+'\n\n每行JSON: {"id":N,"type":"..."}. 只输出JSON.')
        try:
            r=client.chat([{"role":"user","content":prompt}],temperature=0,max_tokens=600)
            for line in r['content'].strip().split('\n'):
                line=line.strip()
                if line.startswith('{'):
                    try:
                        o=json.loads(line); idx=o['id']-1
                        if 0<=idx<len(batch) and o.get('type') in TYPES:
                            labeled.append({**batch[idx],'y':o['type']})
                    except: pass
        except: pass
        time.sleep(0.3)

    from collections import Counter
    dist=Counter(s['y'] for s in labeled)
    print(f'\nStrict labeled: {len(labeled)}, dist: {dict(dist)}')

    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    model=SentenceTransformer('BAAI/bge-small-zh-v1.5')
    tid={t:i for i,t in enumerate(TYPES)}
    valid={t for t,c in dist.items() if c>=10}
    d2=[s for s in labeled if s['y'] in valid]
    if len(d2)<30 or len(valid)<2:
        print(f'Still too sparse ({len(d2)}, {valid}). CONCLUSION: even strict-filtered signal insufficient.')
        json.dump({'total_reviews':len(all_reviews),'evaluative':n_eval,'strict_labeled':len(labeled),
                   'dist':dict(dist),'conclusion':'sparse_after_strict'},
                  open(os.path.join(_here,'..','results','narrative','novel_strict.json'),'w'),indent=2)
        return

    chs=sorted(set(s['ch'] for s in d2)); random.seed(42); random.shuffle(chs)
    nt=max(3,len(chs)//4); tc=set(chs[:nt])
    tr=[s for s in d2 if s['ch'] not in tc]; te=[s for s in d2 if s['ch'] in tc]
    tr_e=model.encode([s['text'] for s in tr],normalize_embeddings=True,show_progress_bar=False)
    te_e=model.encode([s['text'] for s in te],normalize_embeddings=True,show_progress_bar=False)
    tr_y=np.array([tid[s['y']] for s in tr]); te_y=np.array([tid[s['y']] for s in te])
    acc=LogisticRegression(max_iter=1000).fit(tr_e,tr_y).score(te_e,te_y)
    maj=Counter(te_y).most_common(1)[0][1]/len(te_y)
    print(f'\n=== STRICT: content -> evaluative reaction type ===')
    print(f'  Train {len(tr)}, Test {len(te)}, types={valid}')
    print(f'  Accuracy: {acc:.3f} | Majority: {maj:.3f} | Chance: {1/len(valid):.3f}')
    print(f'  {"*** SIGNAL FOUND" if acc>maj+0.05 else "~ still no signal"}')

    json.dump({'total_reviews':len(all_reviews),'evaluative':n_eval,'eval_rate':n_eval/max(len(evaluative_flags),1),
               'strict_segments':len(strict_segs),'strict_labeled':len(labeled),'dist':dict(dist),
               'acc':float(acc),'majority':float(maj),'signal':bool(acc>maj+0.05)},
              open(os.path.join(_here,'..','results','narrative','novel_strict.json'),'w'),indent=2)
    print('\n  Saved: results/narrative/novel_strict.json')

if __name__=='__main__':
    main()
