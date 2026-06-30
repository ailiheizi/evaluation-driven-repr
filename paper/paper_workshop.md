# Evaluation-Driven Representation Discovery: When Audience Reactions Reveal Content Structure Without Labels

**Zhi Liu**

Independent Researcher (ailiheizi@github)

## Abstract

We demonstrate that natural audience evaluations (real-time video comments) contain sufficient semantic structure to automatically discover meaningful content dimensions without human-defined labels. By clustering semantic embeddings of evaluative comments and aligning them with temporal content, we show that: (1) 9 interpretable evaluation dimensions emerge automatically, stable across 16 videos and 3 genres (69–100% recurrence); (2) these dimensions temporally align with specific content segments (p<0.0001, permutation test); (3) dimension distribution patterns serve as content-type fingerprints. We propose Evaluation-Driven Representation Learning (EDRL) as a general paradigm for leveraging natural human evaluations to discover content structure.

**Keywords:** representation learning, unsupervised dimension discovery, audience evaluation, danmaku, temporal alignment

## 1. Introduction

Understanding what audiences notice, appreciate, or criticize in content is central to content analysis, recommendation, and creative AI. Traditional approaches require expert-defined taxonomies (labor-intensive) or large-scale annotation (expensive). We propose using naturally-occurring audience evaluations as the signal source.

Our key insight: audience reactions are not noise—they contain structured semantic information about *which dimensions of content* triggered the response. Unlike RLHF (scalar preference), natural evaluations provide **multi-dimensional semantic feedback** (praise, criticism, analysis across specific aspects).

**Contributions:**
1. Automatic discovery of 9 interpretable content dimensions from danmaku clustering, without predefined taxonomy
2. Temporal alignment validation (entropy 0.17–0.78, p<0.0001) across 16 videos
3. Cross-video stability (8 core dimensions at 69–100% recurrence) and genre fingerprinting
4. Baseline comparisons (vs BERTopic, density-only, random permutation) and inter-annotator validation

## 2. Related Work

**Aspect-Based Sentiment Analysis (ABSA)** extracts aspects from reviews but requires predefined categories or labeled data [1]. Recent unsupervised approaches [2] use clustering but do not validate temporal-content alignment.

**Danmaku Analysis** applies sentiment classification to time-synced comments [3], but treats reactions as scalar signals (positive/negative), missing multi-dimensional structure.

**Video Content Understanding** uses engagement signals for highlight detection [4], but only density-based—cannot distinguish *why* audiences react (plot vs. music vs. acting).

**Our contribution fills the gap:** unsupervised dimension discovery + temporal alignment validation + cross-video stability, from natural evaluations.

## 3. Method

![EDRL Pipeline](figures/fig1_pipeline.png)

**Data:** 19 Bilibili videos (3 genres: film analysis, gaming, science), ~30,000 time-synced comments (danmaku). 16 successfully processed.

**Pipeline:**
1. **Rule-based filtering:** Remove duplicates, spam, short (<4 chars), repetitive (>3 occurrences). Removes ~42% noise.
2. **Evaluativeness classification:** LLM (DeepSeek-V3) judges whether each comment evaluates content quality vs. social interaction/memes. ~40% pass as evaluative.
3. **Dimension discovery:** Encode evaluative comments with BGE-small-zh (512-dim), cluster via KMeans (k=6-10), LLM-name each cluster. No predefined taxonomy.
4. **Temporal alignment:** Bin timeline (30s windows), compute dimension distribution per bin, measure normalized entropy per dimension.

## 4. Results

### 4.1 Dimension Discovery

9 interpretable dimensions emerge, aligning with established film criticism categories:

| Dimension | Proportion | Cross-video recurrence |
|-----------|-----------|----------------------|
| Overall | 17% | 100% (16/16) |
| Plot | 37% | 94% (15/16) |
| Details | 15% | 88% (14/16) |
| Visuals | 8% | 88% (14/16) |
| Pacing | 2% | 81% (13/16) |
| Music | 2% | 75% (12/16) |
| Acting | 12% | 69% (11/16) |

### 4.2 Temporal Alignment

![Temporal heatmap showing concentrated dimensions](figures/fig2_heatmap.png)

Content-specific dimensions concentrate at specific moments; general dimensions disperse:

| Dimension | Mean Entropy | Concentrated? |
|-----------|-------------|---------------|
| Pacing | 0.18 | Yes (content-specific) |
| Acting | 0.28 | Yes |
| Music | 0.28 | Yes |
| Emotional resonance | 0.45 | Moderate |
| Overall | 0.61 | No (general) |
| Plot | 0.78 | No (discussed throughout) |

**Permutation test (n=200):** Real entropy 0.788 vs. shuffled 0.918±0.003, p<0.0001.

### 4.3 Cross-Video Stability and Genre Fingerprinting

![Dimension stability and entropy across videos](figures/fig3_stability.png)

8 core dimensions recur in 69–100% of videos. The same dimension shows opposite patterns by genre: plot is concentrated in film analysis (H=0.46, triggered at explanation points) but dispersed in suspense drama (H=0.90, debated throughout).

### 4.4 Baseline Comparisons

![Permutation test](figures/fig4_permutation.png)

| Method | Mean Entropy | Interpretable? | Dimension info? |
|--------|-------------|----------------|-----------------|
| Random (permuted) | 0.918 | No | No |
| BERTopic (raw) | 0.652 | Noisy | No |
| Density-only | N/A | N/A | WHERE only |
| **EDRL (ours)** | **0.788** | **Yes** | **Yes** |

BERTopic achieves lower entropy by capturing spam patterns (repeated memes at video start), not meaningful evaluation. EDRL filters noise to produce interpretable named dimensions.

### 4.5 Annotation Reliability

Second LLM annotator (stricter prompt): on the 21/50 items both confirm as evaluative, dimension agreement reaches 57% (vs. 11% chance for 9 categories), polarity agreement 86%.

## 5. Discussion

**The EDRL Paradigm.** Natural evaluations are not scalar rewards—they encode (1) what dimensions exist, (2) which are active now, (3) the judgment along each, (4) how dimensions interact. This is strictly richer than RLHF-style preferences and generated for free.

**Broader applicability.** EDRL extends beyond video: product reviews (quality dimensions), game feedback (experience dimensions), food reviews (flavor dimensions), code reviews (quality dimensions).

**Limitations.** Current study uses Chinese-language data from one platform. The "evaluative" boundary is fuzzy (40% broad vs. 17% strict). Downstream task validation remains future work.

## 6. Conclusion

We demonstrate that natural audience evaluations automatically reveal interpretable content dimensions through semantic clustering, validated by temporal alignment (p<0.0001), cross-video stability (16 videos, 69–100% recurrence), and baseline comparisons. We propose EDRL as a paradigm for leveraging the vast resource of naturally-occurring human evaluations.

## References

[1] Zhang et al. A Survey on Aspect-Based Sentiment Analysis. *EMNLP* 2022.

[2] Park & Kim. A Scalable Unsupervised Framework for Multi-Aspect Labeling. *arXiv* 2025.

[3] Zhao et al. Sentiment Analysis of Video Danmakus Based on MIBE-RoBERTa. *Scientific Reports* 2024.

[4] Xu et al. Highlight Detection from Video Comments. *ACM MM* 2021.

[5] Ouyang et al. Training Language Models to Follow Instructions with Human Feedback. *NeurIPS* 2022.
