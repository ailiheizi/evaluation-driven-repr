# When Evaluation Signals Don't Induce Structure

[![DOI](https://zenodo.org/badge/1285039651.svg)](https://doi.org/10.5281/zenodo.21070906)

**A Cautionary Study with Controls for Illusory Emergence**

A rigorous negative result: the appealing intuition that human evaluations (comments, ratings) can *induce* structured content representations does **not** hold—and naive experiments easily produce false positives. We contribute the controls that distinguish genuine emergence from artifacts.

## The Hypothesis (EDRL)

> Training a representation with only natural evaluation signal ("good/bad") should make interpretable content dimensions *emerge*—like salt/sugar levels giving rise to "salty/sweet," or danmaku giving rise to "plot/pacing/emotion."

**Result: it doesn't.** Across 8 experiments in text and images, every apparent success dissolved under a targeted control.

## Key Findings

| Experiment | Apparent result | After control |
|-----------|-----------------|---------------|
| Danmaku clustering + LLM naming | 9 "dimensions" | = shuffled/generic controls (naming prior) |
| Content→reaction-type classify | — | chance (0.667 ≤ majority 0.700) |
| Anchor test | same-para reviews cluster (d=0.60) | topical, not reaction |
| Content↔review retrieval | R@1=0.51 (17× random) | = BM25, survives entity masking → **lexical overlap** |
| Probe raw BGE for dims | — | **0.74–0.83 already** (pretraining did it) |
| Blank CNN + aesthetic score | attributes emerge +0.26 R² | **twin control: used Δ0.175 = unused Δ0.179** → not the signal |

## The Decisive Control: Used-vs-Unused Twins

Inject 6 latent attributes, identical rendering: 3 **used** in the score, 3 **unused twins** absent from it. If evaluation signal causally induces structure, used should emerge more.

```
USED attributes    mean Δ R² = +0.175
UNUSED twins       mean Δ R² = +0.179   ← equal!
Difference:                    -0.005
```

**Used and unused emerge equally** → the effect is ordinary supervised amplification of pixel-readable statistics, **not** evaluation-driven emergence.

## Why It Fails

1. **Text:** pretrained embeddings already encode content dimensions (0.74–0.83 probe accuracy at init). Evaluation signal is redundant.
2. **Images:** supervised training amplifies cheap, label-correlated input statistics. The twin control shows the *signal itself* isn't the driver.

The EDRL intuition conflates "supervised learning finds label-correlated features" (trivial) with "evaluation induces structure that wasn't there" (false).

## Recommended Controls (the contribution)

For any emergence claim from a supervisory signal:
- **Initialization baseline** — probe untrained representation; emergence must exceed what's already readable
- **Lexical/BM25 baseline + entity masking** — is apparent alignment just vocabulary overlap?
- **Used-vs-unused twin control** — inject label-irrelevant twins; genuine emergence must beat the twin baseline
- **Anchor test** — necessary (does signal cluster data?) but not sufficient

## Repository

```
├── paper/paper.md                # Full paper
├── src/
│   ├── twin_control.py            # ★ decisive control (used vs unused)
│   ├── emergence_test.py          # text: dims already in BGE
│   ├── decisive_validation.py     # retrieval: BM25 + entity masking
│   ├── anchor_test.py             # same vs diff paragraph similarity
│   ├── cnn_emergence.py           # synthetic CNN (apparent emergence)
│   └── negative_control.py        # danmaku shuffle controls
└── data/qidian/                   # paragraph-review paired data
```

## Reproduce

```bash
pip install torch numpy scikit-learn sentence-transformers
# The decisive control (fully synthetic, ~2 min CPU):
python src/twin_control.py
```

## Citation

```bibtex
@article{liu2026emergence,
  title={When Evaluation Signals Don't Induce Structure: A Cautionary Study with Controls for Illusory Emergence},
  author={Zhi Liu (ailiheizi)},
  year={2026},
  doi={10.5281/zenodo.21070906},
  url={https://github.com/ailiheizi/evaluation-driven-repr}
}
```

## License

MIT
