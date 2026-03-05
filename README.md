# Prototype LLM Recommendation System for Building Technical Intuition

Can an LLM infer your latent style preferences from just 10 clicks — and then recommend items from retailers it has never seen?

This repo contains a complete, reproducible experiment testing LLM-based cold-start recommendation. A user browsed ~370 jackets on Anthropologie.com, clicked on 10, and an LLM (Claude Opus 4.6) synthesized a preference profile from those clicks alone. That profile was then tested blind against 103 items from Free People and Banana Republic.

The goal isn't a production system — it's building technical intuition for what LLMs can and can't do in recommendation, and where the interesting failure modes are.

## Key Findings

### 1. Preference extraction works — but there's a text bottleneck

LLMs can extract sophisticated preferences from minimal engagement data. From 10 positive and 35 negative samples, five independent agents unanimously extracted nuanced conditional preferences — not just "likes bombers" but "likes bombers ONLY with embellishment." Agent variability was remarkably low; a single run captures the core profile reliably.

However, the image-to-text-to-LLM pipeline loses information. When the model writes "warm earth tones," it collapses a continuous visual space into a discrete label. Downstream scoring then matches items that are technically earth-toned but visually wrong. A native multimodal approach comparing images directly without the text intermediary might unlock better performance.

### 2. Noisy negatives can backfire

The experiment's most counterintuitive result. Adding 35 skipped items as negative signal *sharpened* the preference profile but *hurt* recommendations:

| Metric | Positive-only | Contrastive |
|--------|--------------|-------------|
| Recall @ recommendation set | 50.0% | 22.2% |
| Precision @ recommendation set | 23.7% | 23.5% |
| Lift over baseline | 1.71x | 1.45x |
| F1 | 32.1% | 22.9% |
| Catastrophic misses (clicked items scored "No Match") | 2 | 5 |

Why? Browse skips are noisy — subtle factors (photo angles, cognitive effort) drive inaction. The LLM treated soft browsing skips as hard rejections, creating overly strict rules. Three suede rejections became a categorical material ban; zero blazer clicks became a silhouette exclusion. Neither held up against real behavior. Negative signals should be treated as hypotheses with confidence proportional to evidence — not absolute exclusions.

### 3. Better filter than ranker

Items scored as "No Match" (Tier 4) were clicked at half the baseline rate (9.3% vs. 17.5%) — the model reliably identifies what the user does *not* want. But within the "potentially interesting" zone, the model's confidence ranking was no better than random:

| Tier | Clicks / Total | Click Rate |
|------|---------------|------------|
| Tier 1 (Strong Match) | 1 / 4 | 25.0% |
| Tier 2 (Moderate Match) | 3 / 13 | 23.1% |
| Tier 3 (Weak Match) | 9 / 32 | 28.1% |
| Tier 4 (No Match) | 5 / 54 | 9.3% |

The contrastive profile's Tier 3 items actually outperformed its Tier 1+2 — an inverted calibration. The practical implication: use LLM scoring as a pre-filter to eliminate the bottom ~50% of a catalog, then use traditional signals (collaborative filtering, engagement prediction) for fine-grained ranking.

### 4. Adjacent contexts outperform distant ones

The same preference profile produced starkly different results depending on aesthetic distance from the training data:

| Retailer | Click Rate | Aesthetic Distance |
|----------|-----------|-------------------|
| Free People | 24.3% | Near (sister brand to Anthropologie under URBN Inc.) |
| Banana Republic | 3.0% | Far (classic, corporate, minimal) |

The profile captured preferences specific to Anthropologie's aesthetic world. When applied to a similar world (Free People), those preferences transferred. When applied to a distant world (Banana Republic), the profile had almost nothing to match against. Preference extraction is only as generalizable as the training data is diverse.

### 5. The sweet spot: cold-start and niche interests

LLM inference cost is non-trivial (~$130 total on Claude Opus 4.6 across 430 API calls). The highest-ROI application is where co-engagement signals are sparse — cold-start users/items and niche interests — where collaborative filtering lacks data and LLMs' incremental value is highest. Where co-engagement data is abundant, collaborative filtering already performs well.

## What Remains Open

1. How well do these directional insights hold on a larger, more diverse dataset?
2. Can more specific prompts improve LLM recommender accuracy and bypass the text bottleneck?
3. How does LLM-based recommendation scale upon longer and more diversified user history?
4. How many and how diverse are the minimum viable signals (user history) needed?
5. Can a native multimodal embedding without the text intermediary improve accuracy?
6. Does the negative-signal backfire hold with higher-quality negative signals at scale?
7. Would weighted multi-attribute scoring — rather than binary rule-matching — produce better ranking gradations?

## What's In This Repo

```
.
├── llm_recsys_notebook.ipynb     # Walkthrough notebook (Option A)
├── scripts/                       # CLI tools for reuse (Option B)
│   ├── synthesize_profile.py      #   Infer preference profile from clicks
│   ├── score_catalog.py           #   Score items against the profile
│   ├── generate_eval.py           #   Generate blind evaluation UI
│   ├── analyze_results.py         #   Compute precision, recall, lift
│   ├── _prompts.py                #   All prompt templates
│   ├── _utils.py                  #   Shared helpers
│   ├── config.yaml                #   Default configuration
│   └── README.md                  #   CLI usage guide
├── data/
│   ├── training/                  # 10 positive + 35 negative items with images
│   ├── test/                      # 103 test items from Free People + Banana Republic
│   └── results/                   # Saved profiles, scores, agent syntheses, ground truth
└── README.md                      # This file
```

### Option A: Jupyter Notebook

A narrative walkthrough of the full experiment. Every prompt, every API call, every analysis step — with visualizations.

- **`RUN_API_CALLS = False`** (default): Explore using the saved experiment data, no API key needed
- **`RUN_API_CALLS = True`**: Re-run the full experiment live (requires Anthropic API key)

### Option B: CLI Scripts

Modular scripts so you can replicate the experiment with your own data:

```bash
# 1. Synthesize a preference profile from your clicks
python scripts/synthesize_profile.py \
    --positive my_clicks.json \
    --negative my_skips.json \
    --output output/

# 2. Score a new catalog
python scripts/score_catalog.py \
    --catalog zara_jackets.json \
    --profile output/preference_brief.md \
    --output output/scores.json

# 3. Generate blind evaluation page
python scripts/generate_eval.py \
    --scores output/scores.json \
    --output output/eval.html

# 4. Analyze (after you label items in the HTML)
python scripts/analyze_results.py \
    --scores output/scores.json \
    --labels output/my_labels.json
```

## Quick Start

### Just explore the results (no API key needed)

```bash
git clone https://github.com/sunmx0809/prototype-llm-recsys.git
cd prototype-llm-recsys
pip install jupyterlab pandas matplotlib seaborn
jupyter lab llm_recsys_notebook.ipynb
```

### Re-run or adapt the experiment

```bash
pip install anthropic pandas matplotlib seaborn pyyaml
export ANTHROPIC_API_KEY="sk-ant-..."
```

Then either set `RUN_API_CALLS = True` in the notebook, or use the CLI scripts.

## Experiment Design

```
Phase 1: Training Data Collection
    User browses ~370 items, clicks 10, skips 35
         │
Phase 2: Preference Synthesis (5-agent ensemble)
    ├── 2A: Positive-only (10 clicked items)
    └── 2B: Contrastive (10 clicked + 35 skipped)
         │
Phase 3: Score 103 New Items (from 2 unseen retailers)
    Each item scored Tier 1-4 by both profiles
         │
Phase 4: Blind Evaluation
    User labels all 103 items (shuffled, no scores visible)
         │
Phase 5: Analysis
    Precision, recall, lift, tier calibration, error analysis
```

### The 4-Tier Scoring Schema

| Tier | Label | Meaning |
|------|-------|---------|
| 1 | Strong Match | High confidence the user would click to explore |
| 2 | Moderate Match | Aligns with several preferences but has notable gaps |
| 3 | Weak Match | One or two alignment points, but overall not a fit |
| 4 | No Match | Conflicts with core preferences or hits anti-preferences |

### The 5-Agent Ensemble

To reduce LLM output variance, each preference synthesis runs 5 independent agents with the same prompt. Their outputs are merged into a consensus profile with agreement counts (e.g., "5/5 agents agreed on bomber preference"). This is analogous to inter-rater reliability in human annotation. Agent variability was remarkably low — differences were cosmetic (naming conventions, edge-case observations), not substantive.

## Cost

The original experiment cost ~$130 on Claude Opus 4.6. A standalone replication is much cheaper:

| Model | 5 agents, 103 items | 3 agents, 50 items |
|-------|--------------------|--------------------|
| Sonnet | ~$15-20 | ~$5-8 |
| Haiku | ~$2-4 | ~$1-2 |
| Opus | ~$40-60 | ~$15-25 |

## Adapting to Other Domains

This approach works for any product category where visual browsing is the primary signal:
- **Furniture** — browse a catalog, click on pieces you'd explore
- **Shoes** — same click/skip pattern
- **Jewelry, home decor, art prints** — anywhere "I know it when I see it" dominates

Tips:
- 5-15 positive items is sufficient; more doesn't help much
- 3 agents is usually enough; 5 gives marginal improvement
- Use Sonnet for profile synthesis, Haiku for scoring to optimize cost
- Even 20-30 test items give meaningful signal

## License

This project is shared for educational and research purposes. The product images are sourced from publicly available retailer websites and are included solely for reproducing the experiment.
