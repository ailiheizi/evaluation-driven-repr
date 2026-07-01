# Evaluation-Driven Representation Learning: Content Predicts Collective Reader Reactions

**Zhi Liu**

Independent Researcher, China

`ailiheizi@gmail.com`

## Abstract

Do the reactions a piece of content provokes reflect learnable properties of the content itself, or are they dominated by audience culture and noise? We study this using web-novel paragraphs paired with reader paragraph-level reviews (起点中文网 danmaku-style comments). We first show a naive approach fails: classifying paragraphs into predefined reaction categories performs at chance, because ~63% of reviews are jokes/memes that swamp content signal. However, an anchor test reveals the signal *is* present: reviews of the same paragraph are significantly more similar to each other than to reviews of other paragraphs (Cohen's d=0.60, p<10⁻²²²). We then show the correct formulation—**contrastive content↔review alignment**—successfully extracts this signal: a trained two-tower model retrieves a paragraph's true review-set on held-out chapters at Recall@1=0.35, 17× above random and 2.4× above the raw-embedding baseline. Our contribution is methodological: **collective evaluation is a learnable function of content, but only under a retrieval formulation that bypasses noisy categorical labels.**

## 1. Introduction

When readers comment on a passage of fiction, are their reactions determined by the content, or by factors external to it (memes, community culture, individual mood)? This question bears on a broader hypothesis we call **Evaluation-Driven Representation Learning (EDRL)**: that naturally-occurring human evaluations carry enough information to shape meaningful content representations, without explicit feature engineering.

We test EDRL on a clean data source—web-novel paragraphs with paragraph-level reader reviews—where content (X) and evaluation (Y) are naturally distinct. Our investigation yields a nuanced answer with a clear methodological lesson.

**Contributions:**
1. We show that the naive EDRL formulation (predict predefined reaction categories from content) **fails at chance level**, because meme/joke reviews dominate (63%) and categorical labels compress away the signal.
2. We establish via an anchor test that content-driven signal nonetheless exists: same-paragraph reviews cluster significantly (d=0.60, p<10⁻²²²).
3. We show the **retrieval formulation succeeds**: contrastive content↔review alignment retrieves true review-sets at R@1=0.35 (17× random) on held-out chapters, beating the raw-embedding baseline.
4. We identify the key technique for small-data regimes: residual projection with early stopping, preventing the overfitting that destroys naive contrastive training.

## 2. Related Work

**Aspect-based sentiment analysis** [1] extracts evaluative aspects but requires predefined categories or labels. **Danmaku/comment analysis** [2] treats reactions as scalar sentiment, missing the content-reaction mapping. **RLHF** [3] uses curated preferences; we use natural, unsolicited reactions. **Cross-modal retrieval** (CLIP-style) [4] aligns paired modalities; we adapt this to align content with its collective evaluation. Unlike prior work, we (a) test whether reactions are a *function* of content, and (b) show a retrieval formulation succeeds where classification fails.

## 3. Data

**Source:** 起点中文网 (Qidian), novel 《圣墟》(Shengxu), paragraph-level reader reviews scraped via the public review API. Each paragraph has its text and the reviews readers attached to it.

**Scale:** 30 chapters, 279 paragraphs with ≥2 reviews (242 with ≥3), 1,763 total reviews.

**Key property:** Content (paragraph text) and evaluation (reviews) are distinct objects—unlike video danmaku where the comment text *is* the signal, here we can ask whether content predicts reaction.

## 4. Experiments

### 4.1 Naive Formulation Fails

We label each paragraph's dominant reaction type (via LLM aggregation over its reviews) into categories {joke, awe, emotion, analysis, confusion} and train a classifier from paragraph content.

**Result:** Accuracy 0.667 vs. majority baseline 0.700—**below majority**. The distribution is dominated by `joke` (63%), with `awe`/`emotion` nearly absent (<2% each). Predefined categories compress the signal and meme-reviews swamp it.

**Finding 1:** Predicting predefined reaction categories from content fails at chance. This is the formulation most prior EDRL-style attempts would use—and it does not work.

### 4.2 Anchor Test: The Signal Exists

Before concluding no signal exists, we test the prerequisite directly: are reviews of the *same* paragraph more similar than reviews of *different* paragraphs?

| Comparison | Mean similarity | n pairs |
|-----------|-----------------|---------|
| Same-paragraph reviews | 0.417 | 5,947 |
| Different-paragraph reviews | 0.354 | 5,906 |

**Finding 2:** Same-paragraph reviews are significantly more similar (Δ=+0.062, Cohen's d=0.60, t=32.6, p<10⁻²²²). Reviews cluster by paragraph—**content does drive reaction**. The naive failure (§4.1) was a formulation problem, not an absence of signal.

### 4.3 Retrieval Formulation Succeeds

We train a two-tower contrastive model: `f(content)` and `g(reviews)` projected to a shared space, with a CLIP-style loss matching each paragraph's content to its own review-set. Evaluation: on held-out chapters, rank all candidate review-sets by similarity to `f(content)`; measure Recall@k. We use residual projections (output = normalize(x + α·Δ(x))) with dropout, weight decay, and early stopping on a validation split—critical to prevent overfitting on small data.

| Method | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| Random | 0.020 | 0.102 | 0.204 |
| Raw BGE (no training) | 0.143 | 0.612 | 0.653 |
| **Learned (ours)** | **0.347** | **0.633** | **0.694** |

**Finding 3:** The trained model retrieves the correct review-set at R@1=0.35—17× above random and 2.4× above raw BGE. Training extracts content features that predict collective reaction. (Split by chapter prevents leakage: test chapters are entirely unseen.)

**Finding 4:** Naive contrastive training (plain MLP towers, no regularization) overfits catastrophically (R@1 drops to 0.10 below raw BGE). Residual projection + early stopping is essential in the small-data regime.

## 5. Discussion

**Collective evaluation is a learnable function of content—but the formulation matters.** Three formulations, three outcomes:
- Categorical classification: **fails** (chance) — noise-dominated, compressed labels.
- Raw embedding retrieval: **partial** (R@1=0.14) — signal present but not isolated.
- Contrastive alignment: **succeeds** (R@1=0.35) — extracts content→reaction mapping.

**Why the noise doesn't kill retrieval.** Even with 63% joke reviews, the aggregate review embedding retains a content-correlated component (anchor test). Classification forces a hard category decision dominated by the majority (jokes); retrieval uses the full similarity structure, where the content-correlated component still ranks the true pairing highest.

**Relation to a broader boundary.** This complements findings that *implicit* knowledge is hard to access [author's other work]: here, collective evaluation—seemingly noisy and subjective—turns out to carry an *explicit*, retrievable content signal, once the task avoids lossy categorical bottlenecks.

## 6. Limitations

1. **Single book/platform.** One novel (《圣墟》) on one platform (Qidian). Generalization to other genres, platforms, and languages is untested.
2. **Scale.** 242 paragraphs; retrieval R@k is measured over tens of held-out candidates, not thousands. Larger-scale retrieval would test robustness.
3. **Aggregate reviews.** We mean-pool review embeddings; richer aggregation (attention, set encoders) may capture more.
4. **Correlation, not mechanism.** We show content predicts reaction, not *which* content features drive it. Interpretability is future work.

## 7. Conclusion

We show that collective reader reactions to fiction are a learnable function of content—but only under a retrieval formulation. Naive categorical prediction fails at chance because meme-reviews dominate; yet an anchor test proves the signal exists (d=0.60), and contrastive content↔review alignment extracts it (R@1=0.35, 17× random). The methodological lesson generalizes: when evaluations are noisy, retrieval-based alignment recovers content-evaluation structure that categorical classification destroys.

## 8. Ethics Statement

All reviews are publicly posted reader comments collected via a public API, containing only text—no user identifiers or PII. Data is used in aggregate for non-commercial academic research. We release code and aggregate statistics; we do not redistribute raw user comments beyond anonymized examples.

## 9. Reproducibility Statement

Code and data-processing scripts are publicly released. Settings: BGE-small-zh embeddings (512-dim, normalized); two-tower residual projections (512→128→512, dropout 0.3, weight decay 1e-3, lr 5e-4); early stopping on validation Recall@5 (patience 6); chapter-level train/val/test split (seed 42).

## References

[1] Zhang, W., et al. (2022). A Survey on Aspect-Based Sentiment Analysis. *IEEE TKDE*.

[2] Zhao, J., et al. (2024). Sentiment Analysis of Video Danmakus. *Scientific Reports* 14.

[3] Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS*.

[4] Radford, A., et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP). *ICML*.

[5] Xiao, S., et al. (2023). C-Pack: Packed Resources for General Chinese Embeddings (BGE). *arXiv:2309.07597*.

[6] Reimers, N., & Gurevych, I. (2019). Sentence-BERT. *EMNLP*.

[7] He, R., et al. (2017). An Unsupervised Neural Attention Model for Aspect Extraction. *ACL*.

[8] Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educ. Psych. Meas.*
