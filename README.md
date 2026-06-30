# Evaluation-Driven Representation Discovery (EDRL)

**When Audience Reactions Reveal Content Structure Without Labels**

We demonstrate that natural audience evaluations (real-time video comments) contain sufficient semantic structure to automatically discover meaningful content dimensions—without any human-defined labels or feature engineering.

## Key Findings

| Finding | Evidence |
|---------|----------|
| 9 interpretable dimensions emerge automatically | Plot, Acting, Visuals, Music, Pacing, etc. |
| Dimensions temporally align with content | Entropy 0.01–0.46 (vs 1.0 random) |
| Cross-video stability | Core dimensions recur across genres |
| Content-type fingerprints | Same dimension shows opposite patterns by genre |
| Evaluative rate signals content type | 40-44% (content-heavy) vs 10% (entertainment) |

## Core Idea

```
Traditional:  Human defines features → Model learns mapping
RLHF:        Human labels preferences → Model optimizes reward
EDRL:        Natural evaluations → Automatic dimension discovery → Content representation
```

Natural evaluations are **not just reward signals** — they contain multi-dimensional semantic structure (praise, criticism, analysis) that reveals *which aspects* of content matter.

## Results

### Dimension Discovery (automatic, no predefined taxonomy)

| Dimension | Proportion | Temporal Concentration |
|-----------|-----------|----------------------|
| Plot | 37% | ★ Concentrated (H=0.46) |
| Acting | 12% | ★★ Highly concentrated (H=0.24) |
| Music | 2% | ★★★ Extreme (H=0.01) |
| Overall | 17% | Dispersed (H=0.79) — built-in control |

### Genre Fingerprinting

| | Film Analysis | Suspense Drama |
|---|---|---|
| Plot entropy | 0.46 (concentrated) | 0.90 (dispersed) |
| Interpretation | Discussed at key points | Debated throughout |

## Repository Structure

```
├── paper/
│   └── paper.md          # Full paper draft
├── src/
│   ├── step1_discover_dimensions.py     # Dimension discovery via clustering
│   ├── step1b_filter_and_recluster.py   # Noise filtering + evaluativeness classification
│   └── step2_temporal_alignment.py      # Temporal alignment analysis
├── data/narrative/
│   ├── BV1LSoyYqEuU.json   # Film analysis (Now You See Me)
│   ├── BV1XHmwYpE8s.json   # Suspense drama
│   └── BV1x441187u5.json   # Entertainment montage
└── results/narrative/
    ├── eval_dimensions_discovered.json
    ├── eval_filtered_dimensions.json
    ├── temporal_alignment.json
    └── temporal_alignment_BV1XHmwYpE8s.json
```

## Quick Start

```bash
# Install dependencies
pip install sentence-transformers scikit-learn numpy

# Step 1: Discover dimensions from danmaku
export DEEPSEEK_API_KEY=your_key
export HF_HUB_OFFLINE=1  # use cached model
python src/step1_discover_dimensions.py --data data/narrative/BV1LSoyYqEuU.json

# Step 1b: Filter noise + classify evaluativeness  
python src/step1b_filter_and_recluster.py --data data/narrative/BV1LSoyYqEuU.json

# Step 2: Temporal alignment
python src/step2_temporal_alignment.py
```

## Relation to Prior Work

This extends our earlier paper ["Boundaries of Learnability"](https://github.com/ailiheizi/boundaries-of-learnability):
- Paper 1: Keywords mark structure (4.5×), but semantic content alone fails (1.05×)
- **This paper**: The semantic signal *is* present — it requires dimension-aware clustering to unlock

## Beyond Video: The EDRL Paradigm

EDRL is not limited to video danmaku. **Any domain where humans naturally produce evaluative feedback** can use this framework:

### Demonstrated (this paper)
- **Video content** → Danmaku/comments reveal plot, acting, music, pacing dimensions

### Immediate extensions
- **Food/Recipes** → Taste reviews ("too salty"/"not enough umami") → discover flavor dimensions (saltiness, umami, bitterness) → optimize recipes by navigating flavor space
- **Product design** → Customer reviews → discover quality dimensions (durability, aesthetics, ergonomics) → guide iterative design
- **Game balance** → Player feedback → discover experience dimensions (difficulty, fairness, pacing) → tune gameplay

### Theoretical extensions
- **Image aesthetics** → "beautiful"/"bad composition" → discover visual dimensions → not predefined (rule of thirds, color harmony) but *emergent from human perception*
- **Code quality** → Code review comments → discover quality dimensions (readability, performance, correctness) → train code models with human-aligned objectives
- **Life decisions** → Personal reflections/ratings → build personalized world models where every experience updates a learned embedding space

### The deeper claim

Traditional ML asks: "Given labels, learn features."
RLHF asks: "Given preferences, learn a reward."
**EDRL asks: "Given natural evaluations, discover what dimensions matter — then use those dimensions to structure all downstream learning."**

The key insight is that human evaluations are **not scalar rewards** — they are **structured, multi-dimensional, semantically rich signals** that implicitly encode:
1. What dimensions exist (automatic taxonomy)
2. Which dimensions are active in this context (temporal/spatial alignment)
3. What the human judgment is along each dimension (polarity)
4. How dimensions interact and trade off (implicit preference structure)

This is strictly more information than a single reward signal, and it's **generated for free** by every review, comment, and reaction on the internet.

### Connection to world models

If we view EDRL as a continuous process:
```
Experience → Evaluate → Update representation → Better predictions → Better experiences
```

This is precisely the loop that builds a **world model aligned with human values** — not by optimizing a reward function, but by continuously discovering and refining the dimensions that humans care about.

## Citation

```bibtex
@article{edrl2026,
  title={Evaluation-Driven Representation Discovery: When Audience Reactions Reveal Content Structure Without Labels},
  author={Anonymous},
  year={2026},
  url={https://github.com/ailiheizi/evaluation-driven-repr}
}
```

## License

MIT
