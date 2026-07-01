"""Anchor test: are reviews of the SAME paragraph more similar than reviews of DIFFERENT paragraphs?
================================================================
This is the PREREQUISITE for the whole EDRL idea. If reviews of the same
paragraph cluster together (in embedding space), then there IS a
content-driven signal to learn. If not, reviews are content-independent
(pure meme/random) and no training can help.

Test: for each paragraph, compute avg pairwise similarity of its reviews
(same-para) vs random reviews from other paragraphs (diff-para).

If same-para > diff-para significantly -> signal exists, training is worth it.

Usage:
  export HF_HUB_OFFLINE=1
  python anchor_test.py
"""
import os, sys, json, random
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None
_here = os.path.dirname(os.path.abspath(__file__))

def main():
    from sentence_transformers import SentenceTransformer
    from scipy import stats
    random.seed(42)
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

    data = json.load(open(os.path.join(_here,'..','data','qidian','shengxu_reviews.json'),encoding='utf-8'))
    segs=[]
    for ch in data['chapters']:
        for s in ch['segments']:
            if len(s['reviews'])>=4:
                segs.append(s)
    print(f'Segments with >=4 reviews: {len(segs)}')

    # Embed all reviews per segment
    same_sims=[]; diff_sims=[]
    seg_review_embs=[]
    for s in segs:
        embs = model.encode(s['reviews'][:15], normalize_embeddings=True, show_progress_bar=False)
        seg_review_embs.append(embs)

    # Same-para: pairwise sim within segment
    for embs in seg_review_embs:
        n=len(embs)
        for i in range(n):
            for j in range(i+1,n):
                same_sims.append(float(embs[i]@embs[j]))

    # Diff-para: random pairs across segments
    all_embs=[(si,e) for si,embs in enumerate(seg_review_embs) for e in embs]
    for _ in range(len(same_sims)):
        (s1,e1),(s2,e2)=random.sample(all_embs,2)
        if s1!=s2: diff_sims.append(float(e1@e2))

    same_m,diff_m=np.mean(same_sims),np.mean(diff_sims)
    t,p=stats.ttest_ind(same_sims,diff_sims)
    d=(same_m-diff_m)/np.sqrt((np.std(same_sims)**2+np.std(diff_sims)**2)/2)

    print(f'\n=== ANCHOR TEST: same-para vs diff-para review similarity ===')
    print(f'  Same-paragraph reviews:  {same_m:.4f} (n={len(same_sims)} pairs)')
    print(f'  Diff-paragraph reviews:  {diff_m:.4f} (n={len(diff_sims)} pairs)')
    print(f'  Delta: {same_m-diff_m:+.4f}, Cohen d: {d:.3f}')
    print(f'  t={t:.2f}, p={p:.2e}')
    print()
    if same_m>diff_m and p<0.001:
        print(f'  *** SIGNAL EXISTS: same-para reviews ARE more similar.')
        print(f'      Reviews cluster by paragraph -> content drives reaction.')
        print(f'      => Training a content->review model IS worthwhile.')
    else:
        print(f'  ✗ NO clustering: reviews of same paragraph not more similar.')
        print(f'      Reviews are content-independent (meme/random driven).')
        print(f'      => No training can extract content->review signal.')

    json.dump({'same':float(same_m),'diff':float(diff_m),'delta':float(same_m-diff_m),
               'cohen_d':float(d),'p':float(p),'n_segs':len(segs),
               'signal':bool(same_m>diff_m and p<0.001)},
              open(os.path.join(_here,'..','results','narrative','anchor_test.json'),'w'),indent=2)
    print('\n  Saved: results/narrative/anchor_test.json')

if __name__=='__main__':
    main()
