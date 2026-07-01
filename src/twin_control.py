"""DECISIVE control: used-vs-unused twin attributes
================================================================
Kills the tautology. Inject 6 attributes, same rendering, same variance:
  - 3 USED (in the aesthetic score): sym_u, harm_u, bal_u
  - 3 UNUSED twins (rendered identically, NOT in score): sym_x, harm_x, bal_x

Train CNN on score. Probe all 6.
- If USED attributes emerge MORE than UNUSED twins -> evaluation signal
  causally induced their structure (hypothesis CONFIRMED).
- If USED ~= UNUSED -> training just amplifies pixel-readable statistics
  regardless of the signal (hypothesis FALSIFIED — it's plain supervised
  learning, not evaluation-driven emergence).

Also: nonlinear score (so features can't trivially = linear attr combo).

Usage:
  python twin_control.py
"""
import sys, random
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

def main():
    import torch, torch.nn as nn
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    IMG=32; N=3000

    def render(a):
        # a = [sym_u,harm_u,bal_u, sym_x,harm_x,bal_x]
        img=np.random.rand(3,IMG,IMG).astype(np.float32)*0.3
        # symmetry pair: two independent mirror-blends on top/bottom halves
        for k,half in [(0,slice(0,IMG//2)),(3,slice(IMG//2,IMG))]:
            sub=img[:,half,:]; img[:,half,:]=sub*(1-a[k])+sub[:,:,::-1]*a[k]
        # harmony pair: channel-pull on left/right halves
        for k,col in [(1,slice(0,IMG//2)),(4,slice(IMG//2,IMG))]:
            m=img[:,:,col].mean(0,keepdims=True); img[:,:,col]=img[:,:,col]*(1-a[k])+m*a[k]
        # balance pair: two blobs at controlled x
        for k,row in [(2,slice(2,IMG//2)),(5,slice(IMG//2,IMG-2))]:
            cx=int(IMG/2+(1-a[k])*IMG/3*random.choice([-1,1])); cx=max(3,min(IMG-4,cx))
            img[:,row,cx-2:cx+2]+=0.5
        return np.clip(img,0,1)

    attrs=np.random.rand(N,6).astype(np.float32)
    imgs=np.stack([render(attrs[i]) for i in range(N)])

    # NONLINEAR score using ONLY the 3 USED attributes (idx 0,1,2)
    su,hu,bu=attrs[:,0],attrs[:,1],attrs[:,2]
    score=np.tanh(2*su-1)+np.sin(3*hu)+bu**2 + np.random.randn(N)*0.05
    yb=(score>np.median(score)).astype(np.int64)

    X=torch.tensor(imgs); Y=torch.tensor(yb)
    ntr=int(N*0.8); Xtr,Xte=X[:ntr],X[ntr:]; Ytr=Y[:ntr]
    atr,ate=attrs[:ntr],attrs[ntr:]

    class CNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.f=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(4))
            s.head=nn.Linear(32*16,2)
        def feat(s,x): return s.f(x).flatten(1)
        def forward(s,x): return s.head(s.feat(x))

    names=['sym_USED','harm_USED','bal_USED','sym_unused','harm_unused','bal_unused']
    def probe(net):
        net.eval()
        with torch.no_grad(): ftr=net.feat(Xtr).numpy(); fte=net.feat(Xte).numpy()
        return [r2_score(ate[:,k], Ridge().fit(ftr,atr[:,k]).predict(fte)) for k in range(6)]

    net=CNN()
    base=probe(net)
    print('=== Random-init CNN (baseline readability) ===')
    for n,r in zip(names,base): print(f'  {n:14s}: {r:.3f}')

    opt=torch.optim.Adam(net.parameters(),lr=1e-3); lf=nn.CrossEntropyLoss()
    net.train(); bs=64
    for ep in range(20):
        p=torch.randperm(ntr)
        for i in range(0,ntr,bs):
            idx=p[i:i+bs]; loss=lf(net(Xtr[idx]),Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    learned=probe(net)
    print('\n=== After score-training (nonlinear score, only USED attrs) ===')
    print(f'  {"attribute":14s} {"base":>7s} {"trained":>8s} {"Δ":>7s}')
    for n,b,l in zip(names,base,learned):
        print(f'  {n:14s} {b:>7.3f} {l:>8.3f} {l-b:>+7.3f}')

    used_d=np.mean([learned[k]-base[k] for k in [0,1,2]])
    unused_d=np.mean([learned[k]-base[k] for k in [3,4,5]])
    print(f'\n  USED mean Δ:   {used_d:+.3f}')
    print(f'  UNUSED mean Δ: {unused_d:+.3f}')
    print(f'  Difference (used - unused): {used_d-unused_d:+.3f}')
    print()
    if used_d > unused_d + 0.05:
        print('  *** CONFIRMED: USED attributes emerged MORE than unused twins.')
        print('      Evaluation signal CAUSALLY induced their structure.')
        print('      => Hypothesis holds: eval signal drives emergence.')
    else:
        print('  ✗ FALSIFIED: used ~= unused. Training amplifies pixel-readable')
        print('      statistics regardless of the eval signal. Not emergence —')
        print('      just ordinary supervised learning of cheap pixel features.')

if __name__=='__main__':
    main()
