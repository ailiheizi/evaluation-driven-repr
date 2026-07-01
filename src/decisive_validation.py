"""Decisive validation: is content-review retrieval real or artifact?
================================================================
Kill tests (from adversarial audit):
1. FIX non-determinism: fixed candidate pool, deterministic eval
2. LOCO-CV: leave-one-chapter-out, report mean±std across folds
3. Entity masking: replace shared character/place names with placeholders
   — if signal collapses, it's "text retrieves co-referent text" (circular)
4. BM25 lexical baseline: if BM25 also does well, it's vocabulary overlap

KILL conditions:
(a) learned-vs-BGE gap < cross-fold std -> split luck, not signal
(b) gap collapses under entity masking -> circular (co-reference retrieval)
Result is real ONLY if gap > std AND survives masking.

Usage:
  export HF_HUB_OFFLINE=1
  python decisive_validation.py
"""
import os, sys, json, re
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None
_here = os.path.dirname(os.path.abspath(__file__))

def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    data = json.load(open(os.path.join(_here,'..','data','qidian','shengxu_reviews.json'),encoding='utf-8'))

    segs=[]
    for ci,ch in enumerate(data['chapters']):
        for s in ch['segments']:
            if len(s['reviews'])>=3 and len(s['text'])>=15:
                segs.append({'text':s['text'],'reviews':s['reviews'][:15],'ch':ci})
    print(f'Segments: {len(segs)}')
    chapters = sorted(set(s['ch'] for s in segs))

    # === Entity extraction: find frequent proper nouns (names/places) ===
    # Simple heuristic: 2-4 char sequences appearing in both content and reviews frequently
    from collections import Counter
    # Common character/place names in 圣墟
    ENTITIES = ['楚风','西王','昆仑','夏千语','姜洛神','铜碑','青铜','石昊','荒','诸神','初','女帝','无始','妖','帝','仙']
    def mask_entities(text):
        for e in ENTITIES:
            text = text.replace(e, '□')
        return text

    def embed(texts):
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # Precompute embeddings for both masked and unmasked
    def build_embs(mask=False):
        contents = [mask_entities(s['text']) if mask else s['text'] for s in segs]
        c_emb = embed(contents)
        r_emb = []
        for s in segs:
            revs = [mask_entities(r) if mask else r for r in s['reviews']]
            r_emb.append(embed(revs).mean(0))
        return c_emb, np.array(r_emb)

    def recall_deterministic(c_emb, r_emb, idx, ks=(1,5,10)):
        """Deterministic recall: for test items idx, retrieve among THE SAME test candidate pool"""
        C = c_emb[idx]; R = r_emb[idx]
        # normalize
        C = C/np.linalg.norm(C,axis=1,keepdims=True); R=R/np.linalg.norm(R,axis=1,keepdims=True)
        sim = C@R.T
        n=len(idx); out={}
        for k in ks:
            hits=0
            for i in range(n):
                order=np.argsort(-sim[i], kind='stable')  # stable tie-break -> deterministic
                if i in order[:k]: hits+=1
            out[k]=hits/n
        return out

    # === BM25 baseline ===
    def bm25_recall(mask, idx, ks=(1,5,10)):
        import math
        contents=[mask_entities(segs[i]['text']) if mask else segs[i]['text'] for i in idx]
        reviews=[' '.join(mask_entities(r) if mask else r for r in segs[i]['reviews']) for i in idx]
        # char-level tokens
        def toks(t): return list(t)
        docs=[toks(r) for r in reviews]
        df=Counter();
        for d in docs:
            for w in set(d): df[w]+=1
        N=len(docs); avgdl=np.mean([len(d) for d in docs]) if docs else 1
        def score(q_toks, d, d_toks):
            k1,b=1.5,0.75; s=0; dl=len(d_toks); tf=Counter(d_toks)
            for w in q_toks:
                if w not in df: continue
                idf=math.log((N-df[w]+0.5)/(df[w]+0.5)+1)
                s+=idf*tf[w]*(k1+1)/(tf[w]+k1*(1-b+b*dl/avgdl))
            return s
        out={}
        for k in ks:
            hits=0
            for i in range(len(idx)):
                q=toks(contents[i])
                scores=[score(q,reviews[j],docs[j]) for j in range(len(idx))]
                order=np.argsort(-np.array(scores),kind='stable')
                if i in order[:k]: hits+=1
            out[k]=hits/len(idx)
        return out

    # === LOCO-CV for raw BGE (deterministic, no training) ===
    print('\n=== LOCO-CV: Raw BGE (deterministic) ===')
    for mask_mode in [False, True]:
        c_emb, r_emb = build_embs(mask=mask_mode)
        r1s, r5s = [], []
        bm_r1s = []
        for test_ch in chapters:
            idx=[i for i,s in enumerate(segs) if s['ch']==test_ch]
            if len(idx)<3: continue
            rec=recall_deterministic(c_emb,r_emb,idx)
            r1s.append(rec[1]); r5s.append(rec[5])
        tag = 'MASKED' if mask_mode else 'ORIGINAL'
        print(f'  [{tag}] Raw BGE R@1: {np.mean(r1s):.3f}±{np.std(r1s):.3f}  R@5: {np.mean(r5s):.3f}±{np.std(r5s):.3f}  (n_folds={len(r1s)})')

    # === BM25 LOCO ===
    print('\n=== LOCO-CV: BM25 lexical baseline ===')
    for mask_mode in [False, True]:
        r1s=[]
        for test_ch in chapters:
            idx=[i for i,s in enumerate(segs) if s['ch']==test_ch]
            if len(idx)<3: continue
            rec=bm25_recall(mask_mode, idx)
            r1s.append(rec[1])
        tag='MASKED' if mask_mode else 'ORIGINAL'
        print(f'  [{tag}] BM25 R@1: {np.mean(r1s):.3f}±{np.std(r1s):.3f}')

    # === Random baseline (expected) ===
    fold_sizes=[len([i for i,s in enumerate(segs) if s['ch']==c]) for c in chapters]
    fold_sizes=[n for n in fold_sizes if n>=3]
    rand_r1=np.mean([1/n for n in fold_sizes])
    print(f'\n  Random R@1 (avg over folds): {rand_r1:.3f}')

    print('\n=== INTERPRETATION ===')
    print('  If Raw BGE ORIGINAL >> random but MASKED collapses to ~random:')
    print('    -> signal is co-reference (entity overlap), NOT content->reaction. CIRCULAR.')
    print('  If BM25 ORIGINAL also high:')
    print('    -> pure vocabulary overlap, confirms circularity.')
    print('  If MASKED still >> random:')
    print('    -> there IS content signal beyond entity names. Real (weak) effect.')

    json.dump({'note':'see stdout for LOCO-CV mean±std, masked vs original, BM25'},
              open(os.path.join(_here,'..','results','narrative','decisive.json'),'w'))

if __name__=='__main__':
    main()
