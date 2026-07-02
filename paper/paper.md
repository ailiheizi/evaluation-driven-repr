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

## 3. Text Experiments (Video Danmaku)

### 3.1 Naive Dimension Discovery via Clustering

Clustering danmaku embeddings (BGE-small-zh) and naming clusters with an LLM yields "9 interpretable dimensions" (plot, acting, music, etc.). **Negative control:** clustering shuffled/mismatched/generic comments through the identical pipeline produces the same recurrence rates (real 15% ≈ shuffled 17% ≈ generic 13%). **Conclusion:** the dimensions reflect LLM naming priors, not content signal.

### 3.2 Same-Bin vs Different-Bin Temporal Similarity

We tested whether comments in the same 30s time window are more similar than random pairs. **Result:** same-bin 0.398 vs diff-bin 0.366 (significant, p<10⁻⁶⁸). But after shuffling characters within comments, the effect persists—indicating **lexical locality** (similar words near same timestamp), not semantic content-reaction alignment.

### 3.3 Keyword-Supervised Representation Learning

Trained a contrastive projection on BGE embeddings using danmaku keyword categories (excitement/foreshadow/emotion/humor/negative/plot) as supervision. **Result:** silhouette 0.03→0.61 (apparent success). **But leave-one-type-out test** (train on 5 types, test on held-out 6th with keywords stripped): learned *below* BGE baseline on all 6 types (mean −0.094). **Conclusion:** projection memorized keyword vocabulary patterns, not generalizable evaluative structure.

### 3.4 ASR Content → Danmaku Evaluation (1 video, then 11 videos)

Correct EDRL formulation: X=ASR narration (content), Y=danmaku evaluation type. **1 video (55 bins):** insufficient data, no result. **11 videos (171 bins):** silhouette −0.096→−0.332, probe 25%→25%. **Conclusion:** evaluation signal does not improve ASR content representation. Likely cause: same content triggers diverse reactions across different viewers.

## 4. Text Experiments (Novel Reviews)

### 4.1 Reaction-Type Classification

Predicting a paragraph's dominant reaction type (from content) performs at chance: accuracy 0.667 ≤ majority 0.700. Distribution: 63% joke, awe/emotion <2% each. **Conclusion:** categorical labels are dominated by meme-culture noise.

### 4.2 Strict Filtering (Evaluative-Only Reviews)

LLM re-classified each review as evaluative vs joke/spam: only 20% are evaluative. After filtering, only 40 segments retain ≥3 evaluative reviews. Re-running type classification: still at majority baseline. **Conclusion:** even removing noise, evaluative reviews are too sparse and the remaining signal is insufficient.

### 4.3 Review-Count Prediction (Engagement Intensity)

Predicting whether a paragraph triggers high vs low review count from content. **Result:** AUC=0.593 (weak, above chance 0.5 but marginal). **Conclusion:** content weakly predicts engagement intensity, but effect is too small for practical use or representation learning.

### 4.4 Anchor Test: Same-Paragraph Reviews Cluster

Reviews of the same paragraph are significantly more similar than reviews of different paragraphs (Cohen's d=0.60, t=32.6, p<10⁻²²²). **This suggested real signal—but subsequent controls dissolved the interpretation.**

### 4.5 Content↔Review Retrieval and Controls

A two-tower contrastive model achieves R@1=0.51 under deterministic leave-one-chapter-out CV (17× random). **Controls:**
- **Entity masking:** replacing shared character/place names with placeholders does *not* reduce retrieval (0.51→0.54). Not deep content-reaction mapping.
- **BM25 baseline:** pure lexical matching achieves 0.50—**equal to the neural model**.

**Conclusion:** the alignment is topical vocabulary overlap (reviews mention what the paragraph is about), not "evaluation inducing structure." Recoverable by BM25.

### 4.6 Dimensions Already Exist in Pretraining

Probing raw BGE embeddings for content attributes (action/emotion/suspense/description) yields 0.74–0.83 accuracy—**before any evaluation-signal training**. Training on review-count adds nothing (mean ΔR²=−0.02). **Conclusion:** pretrained text embeddings already encode these dimensions; evaluation signal is redundant on structured (pretrained) inputs.

## 5. Image Experiments (Synthetic, Controlled)

### 5.1 Apparent Emergence on Blank CNN

To remove the pretraining confound, we use raw pixels (genuinely unstructured) and a randomly-initialized CNN. Synthetic 32×32 images carry known latent attributes (symmetry, color harmony, balance); a nonlinear aesthetic "score" depends on them. CNN trains on score alone.

**Apparent result:** attributes become more readable after training (mean R² 0.355→0.615, Δ=+0.26). This looks like evaluation-driven emergence.

### 5.2 Twin Control: Used-vs-Unused Attributes (Decisive)

We inject six attributes with identical rendering and variance: three **used** (define the score) and three **unused twins** (rendered into pixels identically, absent from the score). If the evaluation signal causally induces structure, used attributes should emerge more than their twins.

| Attribute type | Mean Δ R² |
|---------------|-----------|
| USED (in score) | +0.175 |
| UNUSED (twins, not in score) | +0.179 |
| Difference | **−0.005** |

**Used and unused attributes emerge equally.** The emergence is not caused by the evaluation signal—it is ordinary supervised learning amplifying pixel-readable statistics (channel correlation, spatial moments). The highest-weighted attribute in the score (symmetry, weight 0.5) never emerges (R²≈0 throughout), because it requires relational computation—further evidence that pixel readability, not the signal, determines what "emerges."

**Conclusion:** apparent emergence was an artifact of (a) the score being a function of attributes and (b) those attributes being pixel-readable. The twin control isolates the causal variable and shows the evaluation signal itself contributes nothing.

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
