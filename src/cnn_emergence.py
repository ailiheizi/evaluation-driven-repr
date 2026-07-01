"""Emergence on UNSTRUCTURED input: can a blank CNN learn structured
visual features from ONLY aesthetic evaluation signal?
================================================================
This is the RIGHT test of Liu's hypothesis. Text/CLIP embeddings already
have semantic structure (pretraining did the work). Raw pixels do NOT.
A blank (randomly-initialized) CNN starts with no visual concepts.

Controlled synthetic experiment (proof of concept before real AVA data):
- Generate images with KNOWN latent attributes: symmetry, color_harmony,
  balance (the "咸甜酸苦" — objective properties, no aesthetic label)
- Aesthetic "score" = known function of these attributes (simulates human rating)
- Train blank CNN on ONLY the score (the "好吃/难吃")
- Probe: do CNN features encode symmetry/harmony/balance? (emergence)
- Compare vs untrained (random-init) CNN

If trained-CNN features predict the latent attributes MUCH better than
random-init CNN -> evaluation signal induced structured representation
from unstructured input. Hypothesis CONFIRMED (in principle).

Usage:
  python cnn_emergence.py
"""
import sys, os, random
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

def main():
    import torch, torch.nn as nn
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    random.seed(42); torch.manual_seed(42); np.random.seed(42)

    IMG=32  # small images, CPU-friendly
    N=2000

    # === Generate synthetic images with KNOWN latent attributes ===
    def gen_image():
        # attribute 1: symmetry (0-1) — left-right mirror strength
        sym = random.random()
        # attribute 2: color_harmony (0-1) — how close the 3 channels' hues align
        harm = random.random()
        # attribute 3: balance (0-1) — center of mass near center
        bal = random.random()

        img = np.random.rand(3, IMG, IMG).astype(np.float32)*0.3
        # inject symmetry: blend with mirror
        mirror = img[:, :, ::-1]
        img = img*(1-sym) + mirror*sym
        # inject color harmony: pull channels together
        mean_ch = img.mean(0, keepdims=True)
        img = img*(1-harm) + mean_ch*harm
        # inject balance: add a blob at position controlled by bal
        cx = int(IMG/2 + (1-bal)*IMG/3*random.choice([-1,1]))
        cx = max(2,min(IMG-3,cx))
        img[:, IMG//2-2:IMG//2+2, cx-2:cx+2] += 0.5
        img = np.clip(img,0,1)
        return img, np.array([sym,harm,bal],dtype=np.float32)

    imgs=np.zeros((N,3,IMG,IMG),dtype=np.float32); attrs=np.zeros((N,3),dtype=np.float32)
    for i in range(N):
        im,a=gen_image(); imgs[i]=im; attrs[i]=a

    # aesthetic score = known nonlinear function of attributes (the human rating)
    score = (0.5*attrs[:,0] + 0.3*attrs[:,1] + 0.2*attrs[:,2]
             + 0.1*attrs[:,0]*attrs[:,1] + np.random.randn(N)*0.05)
    score_bin = (score > np.median(score)).astype(np.int64)  # like/dislike

    X=torch.tensor(imgs); Yb=torch.tensor(score_bin)
    ntr=int(N*0.8)
    Xtr,Xte=X[:ntr],X[ntr:]; Ytr=Yb[:ntr]
    attr_tr,attr_te=attrs[:ntr],attrs[ntr:]

    # === Blank CNN ===
    class CNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.f=nn.Sequential(
                nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(4))
            s.head=nn.Linear(32*16,2)
        def feat(s,x): return s.f(x).flatten(1)
        def forward(s,x): return s.head(s.feat(x))

    def probe(net, tag):
        """Can CNN features predict the KNOWN latent attributes?"""
        net.eval()
        with torch.no_grad():
            ftr=net.feat(Xtr).numpy(); fte=net.feat(Xte).numpy()
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        r2s=[]
        for k,name in enumerate(['symmetry','harmony','balance']):
            m=Ridge().fit(ftr,attr_tr[:,k])
            r2=r2_score(attr_te[:,k], m.predict(fte))
            r2s.append(r2)
            print(f'    {name:10s}: R²={r2:.3f}')
        print(f'    [{tag}] mean R²={np.mean(r2s):.3f}')
        return np.mean(r2s)

    # === Untrained (random init) baseline ===
    net=CNN()
    print('=== BEFORE training (random-init blank CNN) ===')
    print('  Can random CNN features read latent attributes?')
    base=probe(net,'random-init')

    # === Train on ONLY aesthetic score ===
    print('\n=== Training CNN on ONLY aesthetic score (like/dislike) ===')
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    lossf=nn.CrossEntropyLoss()
    net.train()
    bs=64
    for ep in range(15):
        perm=torch.randperm(ntr)
        tot=0
        for i in range(0,ntr,bs):
            idx=perm[i:i+bs]
            out=net(Xtr[idx]); loss=lossf(out,Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
        if ep%5==0: print(f'  ep{ep}: loss={tot/(ntr//bs):.4f}')

    print('\n=== AFTER training on score: do attributes EMERGE in features? ===')
    learned=probe(net,'score-trained')

    print(f'\n{"="*55}')
    print(f'  Mean attribute R²: random-init={base:.3f} -> score-trained={learned:.3f} (Δ={learned-base:+.3f})')
    if learned>base+0.05:
        print('  *** EMERGENCE CONFIRMED: training on aesthetic score ALONE')
        print('      made latent attributes (symmetry/harmony/balance) readable.')
        print('      Evaluation signal DID induce structured representation')
        print('      from unstructured (pixel) input.')
    else:
        print('  ✗ No emergence even on synthetic data — check design.')

if __name__=='__main__':
    main()
