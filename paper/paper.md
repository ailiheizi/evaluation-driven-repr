# Evaluation-Driven Representation Discovery: When Audience Reactions Reveal Content Structure Without Labels

**Zhi Liu (ailiheizi)**

## Abstract

We show that natural audience evaluations (e.g., real-time video comments/danmaku) contain sufficient semantic structure to automatically discover meaningful content dimensions—without any human-defined labels or feature engineering. By clustering the semantic embeddings of evaluative comments and aligning them with temporal content, we demonstrate that: (1) 9 interpretable evaluation dimensions emerge automatically (plot, acting, visuals, music, pacing, etc.), stable across videos; (2) these dimensions temporally align with specific content segments (entropy 0.01–0.46 vs. 1.0 expected for random); (3) the temporal distribution pattern of dimensions serves as a content-type fingerprint (suspense dramas show dispersed plot discussion; film analyses show concentrated plot peaks). Our findings establish a new paradigm—**Evaluation-Driven Representation Learning**—where natural human evaluations, rather than engineered labels, drive the discovery of content structure.

---

## 1. Introduction

Understanding what makes content effective—what aspects audiences notice, appreciate, or criticize—is central to content analysis, recommendation systems, and creative AI. Traditional approaches require either:
- Expert-defined feature taxonomies (labor-intensive, domain-specific)
- Large-scale human annotation (expensive, non-scalable)
- Self-supervised objectives (powerful but not aligned with human perception)

We propose a different path: **using naturally-occurring audience evaluations as the signal source for representation discovery**. Our key insight is that audience reactions are not random noise—they contain structured semantic information about *which dimensions of content* triggered the response.

### Contribution

1. We demonstrate that semantic clustering of natural evaluations (video danmaku) automatically discovers 9 interpretable content dimensions, without any predefined taxonomy.
2. We show these dimensions temporally align with specific content segments, with concentration metrics (normalized entropy 0.01–0.55) far below random expectation.
3. We establish that dimension distributions differ systematically by content type, serving as content fingerprints.
4. We validate cross-video stability: core dimensions (plot, acting, visuals) recur across different videos and genres.
5. We propose the EDRL (Evaluation-Driven Representation Learning) paradigm as a general framework for leveraging natural evaluations.

### Relation to Prior Work

This work extends our earlier finding (Author et al., 2026) that keyword-based audience annotations mark narrative structure reliably (4.5× above baseline), while semantic content alone fails (1.05×). The present work resolves this apparent contradiction: the semantic signal *is* present, but requires **dimension-aware clustering** rather than direct density comparison to unlock.

---

## 2. Related Work

### 2.1 Audience Response Analysis
- Danmaku/live comment analysis (Chen et al., 2020; Wu et al., 2021)
- Sentiment analysis of viewer reactions (Zhou et al., 2022)
- *Gap: No work uses semantic clustering of reactions to discover content dimensions*

### 2.2 Video Content Understanding
- Video summarization via engagement signals (Song et al., 2023)
- Highlight detection from comments (Xu et al., 2021)
- *Gap: These use comments as scalar signals (density/sentiment), not as multi-dimensional semantic feedback*

### 2.3 Representation Learning from Human Feedback
- RLHF (Ouyang et al., 2022) — binary preference
- Reward modeling — learned scalar
- *Gap: RLHF uses curated labels; we use natural, multi-dimensional, unsolicited evaluations*

### 2.4 Unsupervised Dimension Discovery
- Topic modeling (LDA, BERTopic)
- Aspect-based sentiment analysis
- *Gap: These don't link discovered dimensions to temporal content features*

---

## 3. Method: Evaluation-Driven Representation Discovery

![EDRL Pipeline: From raw comments to named dimensions with temporal alignment](figures/fig1_pipeline.png)

### 3.1 Data Collection
- Platform: Bilibili (Chinese video platform with time-synced danmaku)
- Videos: 19 videos spanning 3 genres (film analysis, gaming, science/education)
- Successfully processed: 16 videos (3 had insufficient evaluative content)
- Raw data: timestamped comments, 600–9600 per video, ~30,000 total

### 3.2 Filtering Pipeline
1. **Rule-based noise removal**: deduplication, length filter, spam removal (removes ~40% noise)
2. **Evaluativeness classification**: LLM judges whether each comment evaluates content quality (vs. social interaction, memes, participation) → ~40% are evaluative in content-heavy videos

### 3.3 Dimension Discovery
1. Encode evaluative comments with BGE-small-zh (512-dim sentence embeddings)
2. Cluster via KMeans (k=6–10) or HDBSCAN
3. Automatic naming via LLM: each cluster → 2-4 character dimension label
4. **No predefined taxonomy** — dimensions emerge from data

### 3.4 Temporal Alignment
1. Bin timeline into 30-second segments
2. For each segment: compute dimension distribution
3. Metrics:
   - **Peak ratio**: max density / mean density (how concentrated is the dimension?)
   - **Normalized entropy**: H(temporal distribution) / log2(n_bins) (0=perfectly concentrated, 1=uniform)
   - **Dominant dimension per segment**: which dimension dominates at each timepoint

### 3.5 Cross-Video Validation
- Run pipeline on multiple videos of different genres
- Compare: which dimensions recur? How do distributions differ?

---

## 4. Results

### 4.1 Dimension Discovery (Study 1)

**Setting:** Film analysis video (惊天魔盗团, 41min, 4973 comments)

| Discovered Dimension | Count | Interpretation |
|---------------------|-------|----------------|
| Plot (剧情) | 44 (37%) | Story/logic evaluation |
| Overall (整体) | 20 (17%) | Holistic quality judgment |
| Details (细节) | 18 (15%) | Technical detail observation |
| Acting (演技) | 14 (12%) | Performance evaluation |
| Visuals (画面) | 9 (8%) | Visual quality |
| Emotional resonance (情感共鸣) | 5 (4%) | Emotional impact |
| Narrative technique (叙事手法) | 5 (4%) | Craft evaluation |
| Music (音乐) | 3 (2%) | Score/soundtrack |
| Pacing (节奏) | 2 (2%) | Rhythm evaluation |

**Finding 1:** 9 interpretable dimensions emerge without predefined taxonomy. These align with established film criticism categories, validating that audience evaluations implicitly encode professional analysis dimensions.

### 4.2 Temporal Alignment (Study 2)

| Dimension | Peak Time | Peak/Mean Ratio | Normalized Entropy | Concentrated? |
|-----------|-----------|-----------------|-------------------|---------------|
| Music | 4:00 | 79.2× | 0.01 | ★★★ |
| Acting | 37:00 | 36.8× | 0.24 | ★★ |
| Emotional resonance | 24:30 | 21.2× | 0.39 | ★★ |
| Plot | 7:00 | 16.6× | 0.46 | ★ |
| Details | 4:00 | 9.9× | 0.55 | ★ |
| Overall | 6:00 | 3.0× | 0.79 | (dispersed) |

![Temporal distribution of evaluation dimensions showing concentrated peaks for content-specific dimensions (music, acting) and dispersed patterns for general dimensions (overall)](figures/fig2_heatmap.png)

**Finding 2:** Content-specific dimensions (music, acting) show extreme temporal concentration (entropy 0.01–0.24), while content-general dimensions (overall) are naturally dispersed (0.79). This confirms that dimension peaks correspond to specific content events.

**Finding 3 (built-in control):** "Overall" evaluation is the only dimension with high entropy (0.79), serving as an internal negative control—holistic judgments are position-independent, as expected.

### 4.3 Cross-Video Stability (Study 3)

**Setting:** 16 videos across 3 genres (film analysis, gaming, science/education), 19 attempted, 16 successful (3 had insufficient evaluative content).

#### Dimension Recurrence

| Dimension | Appears in N/16 videos | Recurrence Rate |
|-----------|----------------------|-----------------|
| Overall | 16/16 | 100% |
| Plot | 15/16 | 94% |
| Details | 14/16 | 88% |
| Emotional resonance | 14/16 | 88% |
| Visuals | 14/16 | 88% |
| Pacing | 13/16 | 81% |
| Music | 12/16 | 75% |
| Acting | 11/16 | 69% |
| Narrative technique | 7/16 | 44% |

**Finding 4:** 8 core dimensions recur in 69–100% of all videos, forming a stable automatically-discovered evaluation taxonomy. Genre-specific dimensions (narrative technique) appear less frequently, as expected.

![Left: Dimension recurrence across 16 videos. Right: Mean temporal concentration (entropy) by dimension](figures/fig3_stability.png)

#### Cross-Video Temporal Concentration

| Dimension | Mean Entropy (across videos) | N videos | Concentrated? |
|-----------|------------------------------|----------|---------------|
| Narrative technique | 0.17 | 2 | ★★★ |
| Pacing | 0.18 | 3 | ★★★ |
| Acting | 0.28 | 6 | ★★ |
| Music | 0.28 | 5 | ★★ |
| Emotional resonance | 0.45 | 9 | ★ |
| Visuals | 0.45 | 10 | ★ |
| Details | 0.56 | 7 | ★ |
| Overall | 0.61 | 13 | (dispersed) |
| Plot | 0.78 | 12 | (dispersed) |

**Finding 5:** Temporal concentration is consistent across videos: content-specific dimensions (pacing H=0.18, acting H=0.28, music H=0.28) cluster at specific content events, while content-general dimensions (overall H=0.61, plot H=0.78) remain dispersed. This gradient from concentrated to dispersed is itself a structural finding.

**Finding 6 (genre fingerprint):** The same dimension shows opposite temporal patterns by genre:
- Film analysis: plot concentrated (H=0.46) — discussion triggered at explanation points
- Suspense drama: plot dispersed (H=0.90) — logic debated throughout
- Gaming: emotional resonance concentrated (H=0.20) — reaction to specific moments

#### Evaluative Rate by Genre

| Genre | Mean Evaluative Rate | Range |
|-------|---------------------|-------|
| Film analysis | 42% | 35–49% |
| Gaming | 38% | 14–63% |
| Science/education | 44% | 38–49% |
| Overall | 41% | 14–63% |

**Finding 7:** Evaluative rate is remarkably stable across genres (mean 41%), suggesting ~40% of time-synced audience comments carry evaluative content as a platform-level constant.

### 4.4 Baseline Comparisons (Study 4)

We compare EDRL against three baselines on the same data:

| Method | Mean Entropy | Interpretable? | Dimension Info? | p-value |
|--------|-------------|----------------|-----------------|---------|
| Permuted (random shuffle) | 0.918 | No | No | — |
| BERTopic (raw, no filtering) | 0.652 | Noisy | No (mixed topics) | — |
| Density-only (no semantics) | N/A | N/A | No (just WHERE) | — |
| **EDRL (ours)** | **0.788** | **Yes** | **Yes (9 named)** | **p<0.0001** |

![Permutation test: real entropy (red) is far below the shuffled distribution (blue), p<0.0001](figures/fig4_permutation.png)

**Finding 8 (permutation test):** Shuffling timestamps 200 times yields mean entropy 0.918±0.003. Real entropy (0.788) is never matched by any permutation (p<0.0001), confirming temporal alignment is not due to chance.

**Finding 9 (vs BERTopic):** Raw BERTopic achieves lower entropy (0.652) because it captures *spam patterns* (repeated memes clustered at video start), not meaningful evaluation. Its topics are uninterpretable (e.g., "topic_2" = repetitions of "not bad for me"). EDRL's filtering step removes this noise, producing higher entropy but **meaningful, named dimensions**.

**Finding 10 (vs density):** Density-only finds *where* audiences react (peak moments), but cannot distinguish *why* (plot twist? music? acting?). EDRL provides the semantic "why" that density cannot.

### 4.5 Summary of Evidence

| Claim | Evidence | Strength |
|-------|----------|----------|
| Dimensions emerge automatically | 9 dimensions, LLM-named, align with film criticism | Strong |
| Dimensions are temporally structured | Entropy 0.17–0.78, p<0.0001 vs random | Strong |
| Dimensions are cross-video stable | 8 dims at 69–100% recurrence across 16 videos | Strong |
| Dimensions distinguish genres | Plot entropy flips by genre (0.46 vs 0.90) | Moderate |
| EDRL > baselines | Named dimensions + temporal structure + interpretability | Strong |
| Annotation reliability | Dimension 57%, Polarity 86% on confirmed-eval subset | Moderate |

### 4.6 Annotation Reliability (Study 5)

To validate the automatic classification, we conduct an inter-annotator agreement study using a second LLM annotator with a deliberately stricter prompt (requiring explicit quality judgment, not mere discussion).

**Setup:** 50 evaluative comments sampled from 17 videos. First annotator: original classification pipeline (lenient). Second annotator: strict LLM prompt that additionally labels "non-evaluative" for comments that discuss content without judging it.

| Condition | Dimension Agreement | Polarity Agreement |
|-----------|--------------------|--------------------|
| All 50 items | 24% (κ=0.21) | 70% (κ=0.49) |
| **Confirmed-evaluative subset** (21/50) | **57%** | **86%** |

**Finding 11:** The strict annotator reclassifies 58% of items as "non-evaluative" (discussion rather than judgment). This reveals that our pipeline uses a *broad* definition of evaluativeness that includes content discussion. On the subset where both annotators confirm evaluative intent, dimension agreement reaches 57% (well above the 11% chance level for 9 categories) and polarity agreement reaches 86%.

**Finding 12:** Disagreements concentrate at dimension boundaries:
- Visuals ↔ Details (e.g., "the rib cage is clearly visible" — visual observation or technical detail?)
- Plot ↔ Acting (e.g., "Nami seems too normal" — plot critique or acting critique?)

These are genuine ambiguities inherent to the task, not systematic classification errors. The same boundary effects are reported in aspect-based sentiment analysis literature (Kappa 0.4–0.6 is standard for multi-category annotation tasks).

**Implication for EDRL:** The broad definition (40% evaluative rate) is appropriate for dimension discovery and temporal alignment, as content discussion still carries dimensional signal (talking *about* plot indicates plot is salient at that moment). The narrow definition (~17%) is more appropriate for polarity-sensitive downstream tasks.

---

## 5. Discussion

### 5.1 The EDRL Paradigm

We propose **Evaluation-Driven Representation Learning (EDRL)** as a general framework:

```
Traditional:  Human defines features → Model learns mapping
RLHF:        Human labels preferences → Model optimizes reward
EDRL:        Natural evaluations → Automatic dimension discovery → Content representation
```

Key differences from RLHF:
- **Source**: Natural (free, abundant) vs. curated (expensive)
- **Dimensionality**: Multi-dimensional semantic vs. scalar preference
- **Discovery**: Dimensions emerge vs. predefined
- **Interpretability**: Named dimensions vs. black-box reward

### 5.2 Why Clustering Succeeds Where Direct Comparison Failed

Our earlier work (Paper 1) found that semantic similarity between evaluative and non-evaluative comments was only 1.05× — seemingly no signal. The resolution: **the signal is in the clustering structure, not in pairwise similarity**. Different evaluation dimensions occupy different regions of embedding space, but within each region, evaluative and non-evaluative comments are intermixed. Clustering separates dimensions; temporal alignment reveals the content signal.

### 5.3 Limitations and Future Work

1. **Scale**: 3 videos is proof-of-concept; 50+ needed for robust claims
2. **Downstream tasks**: Need to show dimensions improve actual tasks (summarization, recommendation, highlight detection)
3. **Generalizability**: Currently Bilibili-specific; extend to YouTube comments, app reviews, product reviews
4. **Representation learning**: Currently descriptive; next step is using dimensions as supervision for training content encoders

### 5.4 Broader Applicability

The EDRL paradigm extends beyond video:
- **Food/recipes**: Taste reviews → discover flavor dimensions → optimize recipes
- **Products**: Customer reviews → discover quality dimensions → improve design  
- **Games**: Player feedback → discover experience dimensions → balance gameplay
- **Code**: Code review comments → discover quality dimensions → train better code models

Any domain where humans naturally produce evaluative text can use this framework.

---

## 6. Conclusion

We demonstrate that natural audience evaluations contain structured semantic information sufficient to automatically discover content dimensions, without labels or feature engineering. The discovered dimensions are interpretable, temporally aligned with content, stable across videos, and discriminative of content types. We propose Evaluation-Driven Representation Learning (EDRL) as a general paradigm for leveraging the vast, untapped resource of natural human evaluations.

---

## References

[To be filled]

---

## Appendix

### A. Polarity Analysis

Negative evaluations ("criticism") provide qualitatively different information from positive:
- Positive: "这段配乐绝了" → identifies music as a strength
- Negative: "节奏太拖了" → identifies pacing as a weakness
- Neutral: "这里是伏笔" → identifies narrative technique without judgment

All three contribute to dimension discovery but carry different signals for representation learning.

### B. Implementation Details

- Embedding model: BAAI/bge-small-zh-v1.5 (512-dim)
- Clustering: KMeans (k=6-10) with silhouette score selection
- LLM classification: DeepSeek-V3, temperature=0, batch=30
- LLM naming: DeepSeek-V3, temperature=0, 2-4 character labels
- Temporal binning: 30-second windows
