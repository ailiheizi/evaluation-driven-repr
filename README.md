# Evaluation-Driven Representation Learning (EDRL)

[![DOI](https://zenodo.org/badge/1285039651.svg)](https://doi.org/10.5281/zenodo.21070906)

**Content Predicts Collective Reader Reactions**

We show that the reactions a piece of content provokes are a **learnable function of the content itself**—but only under a retrieval formulation that bypasses noisy categorical labels. Studied on web-novel paragraphs paired with reader reviews.

## Key Findings

| Finding | Evidence |
|---------|----------|
| Naive categorical prediction **fails** | Accuracy 0.667 ≤ majority 0.700 (jokes dominate 63%) |
| But the signal **exists** (anchor test) | Same-paragraph reviews cluster: Cohen's d=0.60, p<10⁻²²² |
| Retrieval formulation **succeeds** | Content→review R@1=0.35 (17× random, 2.4× raw BGE) |
| Small-data technique matters | Residual projection + early stopping prevents overfit |

## The Story: Fail, Then Succeed

Three formulations, three outcomes:

```
Categorical classification:  FAILS   (chance) — noise-dominated, lossy labels
Raw embedding retrieval:     PARTIAL (R@1=0.14) — signal present, not isolated
Contrastive alignment:       SUCCEEDS (R@1=0.35) — extracts content→reaction map
```

The key insight: **collective evaluation is noisy (63% memes/jokes), but the content-correlated component survives in aggregate**. Categorical classification lets the majority (jokes) win; retrieval uses the full similarity structure where the content signal still ranks the true pairing highest.

## Results

### Anchor Test (does signal exist?)

| Comparison | Mean similarity |
|-----------|-----------------|
| Same-paragraph reviews | 0.417 |
| Different-paragraph reviews | 0.354 |

Δ=+0.062, Cohen's d=0.60, p<10⁻²²² — **reviews cluster by paragraph, content drives reaction.**

### Content→Review Retrieval (held-out chapters)

| Method | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| Random | 0.020 | 0.102 | 0.204 |
| Raw BGE | 0.143 | 0.612 | 0.653 |
| **Learned (ours)** | **0.347** | **0.633** | **0.694** |

## Data

- **Source:** 起点中文网 (Qidian), novel 《圣墟》, paragraph-level reader reviews
- **Scale:** 30 chapters, 279 paragraphs (≥2 reviews), 1,763 reviews
- **Key property:** content (paragraph text) and evaluation (reviews) are distinct objects

## Repository Structure

```
├── paper/paper.md              # Full paper (positive result)
├── src/
│   ├── anchor_test.py           # Same vs diff-paragraph review similarity
│   ├── content_review_align_v2.py  # Two-tower contrastive alignment (main result)
│   ├── novel_sentiment.py       # Naive categorical baseline (fails)
│   └── fetch_qidian.py          # Data collection
└── data/qidian/                 # Paragraph-review paired data
```

## Quick Start

```bash
pip install sentence-transformers scikit-learn numpy torch scipy
export HF_HUB_OFFLINE=1

# 1. Anchor test: does content-driven signal exist?
python src/anchor_test.py

# 2. Main result: contrastive content-review alignment
python src/content_review_align_v2.py
```

## Note on Prior Version

An earlier version of this project explored automatic dimension discovery from video danmaku via clustering. Rigorous negative controls revealed that approach was dominated by LLM naming priors rather than true content signal. This version pivots to a validated positive result: content-review retrieval alignment. The negative findings are documented in the commit history and informed the current formulation.

## Citation

```bibtex
@article{edrl2026,
  title={Evaluation-Driven Representation Learning: Content Predicts Collective Reader Reactions},
  author={Zhi Liu (ailiheizi)},
  year={2026},
  doi={10.5281/zenodo.21070906},
  url={https://github.com/ailiheizi/evaluation-driven-repr}
}
```

## License

MIT
