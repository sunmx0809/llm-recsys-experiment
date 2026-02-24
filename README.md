# Prototype LLM Recommendation System for Building Technical Intuition

Can an LLM infer your latent style preferences from just 10 clicks — and then recommend items from retailers it has never seen?

This repo contains a complete, reproducible experiment testing LLM-based cold-start recommendation. A user browsed ~370 jackets on Anthropologie.com, clicked on 10, and an LLM (Claude) synthesized a preference profile from those clicks alone. That profile was then tested blind against 103 items from Free People and Banana Republic.

The goal isn't a production system — it's building technical intuition for what LLMs can and can't do in recommendation, and where the interesting failure modes are.

## Key Findings

| Finding | Detail |
|---------|--------|
| The LLM understood preferences well from 10 clicks | Correctly identified: craft-forward bombers, textural complexity, dark earth tones, cropped silhouettes |
| Negative signal improved understanding but hurt recommendations | Contrastive profile had sharper logic ("bombers ONLY with embellishment") but created hard rejection gates, causing more misses |
| Positive-only profile achieved better recall at similar precision | Simpler profile was more permissive, capturing more items the user actually liked |
| A "text bottleneck" limits quality | Image → text → LLM pipeline loses continuous visual info (texture, drape, color temperature) when compressing to text tokens |
| Tier 4 calibration worked well | Items scored as "No Match" had ~9% click rate vs ~17% base rate — the system reliably identifies non-matches |
| Cross-retailer transfer is hard | User click rate: Free People 24% vs Banana Republic 3% — brand/aesthetic fit matters beyond item attributes |

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
cd llm-recsys-experiment
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

To reduce LLM output variance, each preference synthesis runs N independent agents with the same prompt. Their outputs are merged into a consensus profile with agreement counts (e.g., "5/5 agents agreed on bomber preference"). This is analogous to inter-rater reliability in human annotation.

## Cost

The original experiment cost ~$130 on Claude Opus (the most expensive model) in a conversational setting. A standalone replication is much cheaper:

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
