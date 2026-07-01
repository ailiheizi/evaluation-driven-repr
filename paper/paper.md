# When Evaluation Signals Don't Induce Structure: A Cautionary Study with Controls for Illusory Emergence

**Zhi Liu**

Independent Researcher, China

`ailiheizi@gmail.com`

## Abstract

A recurring intuition in representation learning is that naturally-occurring human evaluations (comments, ratings, likes) carry enough information to induce structured representations of content—that training on "good/bad" should make interpretable dimensions emerge, analogous to how salt/sugar levels give rise to "salty/sweet." We test this hypothesis rigorously across two modalities (text and images) and eight experiments, and find it **does not hold**—but the interesting result is *why*, and how easily naive experiments produce false positives. We show that (1) on text, content dimensions are already present in pretrained embeddings, leaving evaluation signal nothing to induce; (2) apparent content↔evaluation alignment is largely lexical/topical overlap (matched by BM25, robust to entity masking); (3) on images, a blank CNN trained on aesthetic scores appears to make latent attributes emerge (+0.26 R²)—but a **twin-attribute control** reveals used and unused attributes emerge equally (Δ 0.175 vs 0.179), proving the effect is ordinary supervised amplification of pixel-readable statistics, not evaluation-driven emergence. Our contribution is a methodology: **anchor tests, entity-masking controls, and used-vs-unused twin controls** that distinguish genuine emergence from artifacts. We argue such controls should be standard when claiming representation emergence.

## 1. Introduction

Consider the hypothesis we call **Evaluation-Driven Representation Learning (EDRL)**: that human evaluations, being cheap and abundant, can supervise the discovery of structured content representations without explicit feature engineering. The intuition is seductive—train on wine ratings and "body/acidity/tannin" axes should emerge; train on danmaku and "plot/pacing/emotion" should emerge.

We set out to validate this and instead accumulated eight negative or artifact results across text and image modalities. Rather than a failure report, this paper documents **the controls needed to avoid false positives**, several of which we ourselves initially failed. Each naive experiment produced an encouraging signal that a targeted control then dissolved.

**Contributions:**
1. A systematic negative result: evaluation signal does not induce structured representations in either pretrained text embeddings or blank image CNNs, across 8 experiments.
2. Three diagnostic controls—anchor test, entity-masking, and used-vs-unused twin control—that separate genuine emergence from artifacts.
3. An explanatory account: apparent emergence comes from (a) pretraining (text) or (b) supervised amplification of cheap input statistics (pixels), not from the evaluation signal itself.

## 2. Hypothesis and Data

**Hypothesis (EDRL):** Training a representation with only naturally-occurring evaluation signal as supervision causes interpretable, content-relevant dimensions to emerge that were not accessible before.

**Data:**
- *Text:* Bilibili danmaku (time-synced video comments) and 起点中文网 (Qidian) novel paragraph-level reader reviews (novel 《圣墟》: 30 chapters, 279 paragraphs, 1,763 reviews). Content (paragraph/frame) and evaluation (comments) are distinct objects.
- *Images:* Controlled synthetic images with known latent attributes, used to test emergence where input is genuinely unstructured (raw pixels vs. pretrained text embeddings).

## 3. Text Experiments

### 3.1 Naive Dimension Discovery Fails

Clustering danmaku embeddings and naming clusters with an LLM yields "9 interpretable dimensions"—but a negative control (clustering shuffled/mismatched/generic comments through the identical pipeline) produces the same recurrence rates. **The dimensions reflect LLM naming priors, not content signal.**

### 3.2 Reaction-Type Classification Fails

Predicting a paragraph's dominant reaction type (from content) performs at chance: accuracy 0.667 ≤ majority 0.700. Reviews are 63% jokes/memes; categorical labels are dominated by this noise.

### 3.3 Anchor Test: Signal Exists, But What Kind?

Reviews of the same paragraph are significantly more similar than reviews of different paragraphs (Cohen's d=0.60, p<10⁻²²²). This *seems* to support EDRL—until we ask what drives the similarity.

### 3.4 Control: Content↔Review Alignment Is Lexical

A retrieval formulation (content retrieves its own review-set) achieves R@1=0.51 under deterministic leave-one-chapter-out CV (17× random). But two controls dissolve the interpretation:
- **Entity masking:** replacing shared character/place names with placeholders does *not* reduce retrieval (0.51→0.54). The signal is not deep content-reaction mapping.
- **BM25 baseline:** pure lexical matching achieves 0.50—**equal to the neural model**. The alignment is topical/vocabulary overlap: reviews mention what the paragraph is about.

**Finding:** content and its reviews share vocabulary (readers discuss what happens), but this is not "evaluation inducing structure"—it is topical co-occurrence, recoverable by BM25.

### 3.5 Control: Dimensions Already Exist in Pretraining

Probing raw BGE embeddings for content attributes (action/emotion/suspense/description) yields 0.74–0.83 accuracy—**before any evaluation-signal training**. Training on review-count adds nothing (mean ΔR²=−0.02). Pretrained text embeddings already encode these dimensions; evaluation signal has nothing left to induce.

## 4. Image Experiments

### 4.1 Apparent Emergence on Blank CNN

To remove the pretraining confound, we use raw pixels—genuinely unstructured input—and a randomly-initialized CNN. Synthetic images carry known latent attributes (symmetry, color harmony, balance); an aesthetic "score" is a function of them; the CNN trains on score alone.

**Apparent result:** after training, CNN features predict the latent attributes far better than at initialization (mean R² 0.355→0.615, Δ=+0.26). This looks like evaluation-driven emergence.

### 4.2 Twin Control Kills It

We inject six attributes with identical rendering and variance: three **used** (in the score) and three **unused twins** (rendered into pixels identically, absent from the score). If the evaluation signal causally induces structure, used attributes should emerge more than their twins.

| Attribute | base R² | trained R² | Δ |
|-----------|---------|-----------|-----|
| harmony (USED) | 0.233 | 0.762 | +0.529 |
| harmony (unused twin) | 0.235 | 0.774 | +0.539 |
| USED mean Δ | | | **+0.175** |
| UNUSED mean Δ | | | **+0.179** |

**Used and unused attributes emerge equally (difference −0.005).** The emergence is not caused by the evaluation signal—it is ordinary supervised learning amplifying pixel-readable statistics (channel correlation, spatial moments) that a random CNN already partly decodes. The highest-weighted attribute in the score (symmetry) never emerges at all (R²≈0), because it requires relational computation a small CNN doesn't surface—further evidence that readability, not the signal, determines emergence.

## 5. Why the Hypothesis Fails

Two mechanisms explain every apparent positive:

1. **Pretraining already did it (text).** Modern embeddings are highly semantic; content dimensions are present at initialization. Evaluation signal is redundant.
2. **Supervised learning amplifies cheap statistics (pixels).** Training on any label surfaces input features that are (a) easy to read and (b) correlated with the label. This is indistinguishable from "emergence" without a twin control—and the twin control shows the signal itself is not the driver.

The EDRL intuition conflates "supervised learning finds label-correlated features" (true, trivial) with "the evaluation signal induces content structure that wasn't there" (what we tested, false).

## 6. Recommended Controls

For any claim of representation emergence from a supervisory signal:
- **Anchor test:** does the signal cluster the data at all? (necessary, not sufficient)
- **Lexical/BM25 baseline + entity masking:** is apparent alignment just vocabulary overlap?
- **Used-vs-unused twin control:** inject label-irrelevant twins; genuine signal-driven emergence must exceed the twin baseline.
- **Initialization baseline:** probe the untrained representation; emergence must exceed what's already readable.

## 7. Limitations

Our text data is Chinese, single-platform; image experiments are synthetic (real aesthetic data like AVA is future work, though the twin control would apply identically). We test contrastive and classification formulations; we cannot rule out that some exotic objective induces genuine emergence. Our claim is scoped: the natural, direct formulations of EDRL that practitioners would try do not work, and the reasons are systematic.

## 8. Conclusion

The intuition that human evaluations induce structured content representations is appealing but, in the formulations we tested across text and images, false. Apparent successes are pretraining artifacts (text) or supervised amplification of readable input statistics (images)—not evaluation-driven emergence. We contribute a control methodology (anchor, masking, twin, init baselines) that distinguishes real emergence from these artifacts, and we recommend it become standard practice. Negative results like this one are, we argue, exactly where such controls prove their worth.

## 9. Ethics Statement

All data are publicly posted comments/reviews collected via public APIs, containing only text—no user identifiers or PII, used in aggregate for non-commercial research. Synthetic images involve no human data.

## 10. Reproducibility Statement

All code, data-processing scripts, and the twin-control experiment are publicly released. Key settings documented in the repository (BGE-small-zh embeddings; CNN 3-conv layers; Ridge probes; deterministic LOCO-CV; twin-attribute rendering).

## References

[1] Zhang, W., et al. (2022). A Survey on Aspect-Based Sentiment Analysis. *IEEE TKDE*.
[2] Radford, A., et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP). *ICML*.
[3] Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS*.
[4] Xiao, S., et al. (2023). C-Pack: Packed Resources for General Chinese Embeddings (BGE). *arXiv:2309.07597*.
[5] Robert, S., et al. (1994). Okapi BM25. *TREC*.
[6] Murphy, K., et al. (2019). On the pitfalls of probing and interpretability. (methodology)
[7] Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. *EMNLP*.
[8] Ross, A., et al. (2021). Evaluating the interpretability illusions. (control methodology)
