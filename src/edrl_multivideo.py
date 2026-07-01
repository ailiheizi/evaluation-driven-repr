"""EDRL Multi-Video: Can evaluation signal shape CONTENT representation?
========================================================================
Now with 11 videos (each has ASR content + danmaku evaluation).
X = ASR content segment embedding (video narration, NO danmaku keywords)
Y = same-timebin danmaku evaluation type
Train projection on X supervised by Y; test if X-space becomes structured
by evaluation type, on HELD-OUT videos.

Usage:
  export HF_HUB_OFFLINE=1
  python edrl_multivideo.py
"""
import os, sys, glob, json, random
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from step1b_filter_and_recluster import load_danmaku, rule_filter

EVAL_LEXICON = {
    'excitement': ['高能','爽','燃','牛','厉害','强','帅','炸','热血','震撼'],
    'foreshadow': ['伏笔','细节','原来','这里','注意','发现','呼应','铺垫','暗示'],
    'emotion':    ['感动','泪','哭','破防','emo','心疼','难过','温暖','治愈'],
    'humor':      ['哈哈','笑','搞笑','梗','乐','逗','沙雕','好笑'],
    'negative':   ['无聊','尬','烂','差','拉胯','难看','无语','离谱','强行'],
    'plot':       ['剧情','逻辑','设定','反转','结局','故事','情节','悬念'],
}

def label_by_keyword(text):
    scores = {e: sum(1 for kw in kws if kw in text) for e,kws in EVAL_LEXICON.items()}
    scores = {e:c for e,c in scores.items() if c>0}
    return max(scores, key=scores.get) if scores else None


def build_bins(bvid, data_dir, bin_size=30):
    """Build (asr_text, eval_type) bins for one video"""
    asr_path = os.path.join(data_dir, f'{bvid}_asr.json')
    dm_path = os.path.join(data_dir, f'{bvid}.json')
    if not os.path.exists(asr_path) or not os.path.exists(dm_path):
        return []
    with open(asr_path,'r',encoding='utf-8') as f:
        asr = json.load(f)['segments']
    danmaku, title, dur = load_danmaku(dm_path)
    filtered = rule_filter(danmaku)
    if not dur: dur = max((s['end'] for s in asr), default=0)

    from collections import Counter
    n_bins = int(np.ceil(dur/bin_size))
    bins = []
    for b in range(n_bins):
        start, end = b*bin_size, (b+1)*bin_size
        asr_text = ' '.join(s['text'] for s in asr if s['start']<end and s['end']>start)
        labels = [label_by_keyword(d['text']) for d in filtered if start<=d['t']<end]
        labels = [l for l in labels if l]
        if not asr_text.strip() or len(labels)<3:
            continue
        bins.append({'asr':asr_text, 'eval':Counter(labels).most_common(1)[0][0], 'bvid':bvid})
    return bins


def main():
    import torch, torch.nn as nn
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LogisticRegression
    from collections import Counter

    random.seed(42); torch.manual_seed(42); np.random.seed(42)
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    data_dir = os.path.join(_here,'..','data','narrative')

    # Find all videos with ASR
    asr_files = glob.glob(os.path.join(data_dir,'BV*_asr.json'))
    bvids = [os.path.basename(f).replace('_asr.json','') for f in asr_files]
    print(f'Videos with ASR: {len(bvids)}')

    all_bins = []
    for bv in bvids:
        b = build_bins(bv, data_dir)
        all_bins.extend(b)
        print(f'  {bv}: {len(b)} bins')

    types = sorted(EVAL_LEXICON.keys())
    tid = {t:i for i,t in enumerate(types)}
    print(f'\nTotal bins: {len(all_bins)}')
    print(f'Eval distribution: {dict(Counter(b["eval"] for b in all_bins))}')

    # Split by VIDEO (anti-circularity)
    vids = sorted(set(b['bvid'] for b in all_bins))
    random.shuffle(vids)
    n_test = max(2, len(vids)//4)
    test_vids = set(vids[:n_test])
    train = [b for b in all_bins if b['bvid'] not in test_vids]
    test = [b for b in all_bins if b['bvid'] in test_vids]
    print(f'Train: {len(train)} bins ({len(vids)-n_test} videos), Test: {len(test)} bins ({n_test} videos)')

    if len(train)<30 or len(test)<10:
        print('Not enough data'); return

    tr_emb = model.encode([b['asr'] for b in train], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    te_emb = model.encode([b['asr'] for b in test], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    tr_y = np.array([tid[b['eval']] for b in train])
    te_y = np.array([tid[b['eval']] for b in test])

    # Baseline
    print('\n=== BASELINE (raw BGE of ASR content) ===')
    base_sil = silhouette_score(te_emb, te_y, metric='cosine') if len(set(te_y))>1 else -1
    base_probe = LogisticRegression(max_iter=1000).fit(tr_emb,tr_y).score(te_emb,te_y)
    print(f'  Silhouette: {base_sil:.4f}  Probe: {base_probe:.4f}')

    # Train projection
    print('\n=== TRAINING projection (eval signal shapes content) ===')
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    Xtr = torch.tensor(tr_emb,dtype=torch.float32).to(dev)
    Ytr = torch.tensor(tr_y,dtype=torch.long).to(dev)
    proj = nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,128)).to(dev)
    opt = torch.optim.Adam(proj.parameters(),lr=1e-3)
    def loss_fn(z,y,t=0.1):
        z=nn.functional.normalize(z,dim=1); sim=z@z.T/t; sim.fill_diagonal_(-1e9)
        y=y.view(-1,1); pos=(y==y.T).float(); pos.fill_diagonal_(0)
        if pos.sum()==0: return torch.tensor(0.0,device=dev)
        logp=sim-torch.logsumexp(sim,dim=1,keepdim=True)
        return -((pos*logp).sum(1)/pos.sum(1).clamp(min=1)).mean()
    proj.train()
    for ep in range(50):
        loss=loss_fn(proj(Xtr),Ytr); opt.zero_grad(); loss.backward(); opt.step()
        if ep%20==0: print(f'  ep{ep}: loss={loss.item():.4f}')

    # Eval on held-out videos
    print('\n=== LEARNED (held-out videos) ===')
    proj.eval()
    with torch.no_grad():
        te_proj = proj(torch.tensor(te_emb,dtype=torch.float32).to(dev)).cpu().numpy()
        tr_proj = proj(Xtr).cpu().numpy()
    learn_sil = silhouette_score(te_proj, te_y, metric='cosine') if len(set(te_y))>1 else -1
    learn_probe = LogisticRegression(max_iter=1000).fit(tr_proj,tr_y).score(te_proj,te_y)
    print(f'  Silhouette: {learn_sil:.4f}  Probe: {learn_probe:.4f}')

    print(f'\n{"="*55}')
    print(f'  {"Metric":15s} {"BGE":>10s} {"Learned":>10s} {"Delta":>10s}')
    print(f'  {"Silhouette":15s} {base_sil:>10.4f} {learn_sil:>10.4f} {learn_sil-base_sil:>+10.4f}')
    print(f'  {"Probe acc":15s} {base_probe:>10.4f} {learn_probe:>10.4f} {learn_probe-base_probe:>+10.4f}')
    print()
    print('  KEY: input is ONLY ASR content (no danmaku at test). If learned>BGE,')
    print('  evaluation signal shaped content representation on UNSEEN videos.')
    if learn_probe>base_probe:
        print('  ✓ Content representation improved by evaluation signal.')
    else:
        print('  ✗ No improvement — signal insufficient or content not predictive.')

    json.dump({'baseline':{'sil':float(base_sil),'probe':float(base_probe)},
               'learned':{'sil':float(learn_sil),'probe':float(learn_probe)},
               'n_videos':len(vids),'n_train':len(train),'n_test':len(test)},
              open(os.path.join(data_dir,'..','..','results','narrative','edrl_multivideo.json') if os.path.exists(os.path.join(data_dir,'..','..','results')) else os.path.join(_here,'..','results','narrative','edrl_multivideo.json'),'w'), indent=2)
    print('\n  Saved: results/narrative/edrl_multivideo.json')


if __name__=='__main__':
    main()
