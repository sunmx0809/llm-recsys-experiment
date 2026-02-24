# LLM Recommendation Experiment — CLI Scripts

Modular CLI scripts for replicating the LLM-as-recommendation-engine experiment with your own data.

## Installation

```bash
pip install anthropic pyyaml
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Scripts

| Script | Phase | What it does |
|--------|-------|-------------|
| `synthesize_profile.py` | 2 | Infers a user preference profile from training data via multi-agent LLM ensemble |
| `score_catalog.py` | 3 | Scores test catalog items against the preference profile on a 4-tier scale |
| `generate_eval.py` | 4 | Generates a shuffled HTML page for blind user evaluation |
| `analyze_results.py` | 5 | Computes precision, recall, lift, tier calibration from scores + user labels |

Shared modules (`_prompts.py`, `_utils.py`) contain prompt templates and helper functions.

## Full Workflow Example

### 1. Synthesize a preference profile

Positive-only (from clicked items):

```bash
python scripts/synthesize_profile.py \
    --positive training_data.json \
    --num-agents 5 \
    --output output/phase2a
```

Contrastive (clicked + skipped items):

```bash
python scripts/synthesize_profile.py \
    --positive training_data.json \
    --negative negative_samples.json \
    --num-agents 5 \
    --output output/phase2b
```

### 2. Score a test catalog

```bash
python scripts/score_catalog.py \
    --catalog fp_catalog.json \
    --profile output/phase2b/preference_brief.md \
    --images-dir phase3_images/fp \
    --output output/fp_scores.json
```

### 3. Generate blind evaluation HTML

```bash
python scripts/generate_eval.py \
    --scores output/fp_scores.json \
    --images-dir phase3_images/fp \
    --output output/blind_eval.html
```

Open `output/blind_eval.html` in a browser. Click "Would Explore" or "Skip" for each item. When done, copy the JSON blob at the bottom and save it to `output/labels.json`.

### 4. Analyze results

Single scoring:

```bash
python scripts/analyze_results.py \
    --scores output/fp_scores.json \
    --labels output/labels.json
```

Side-by-side comparison (e.g., positive-only vs. contrastive):

```bash
python scripts/analyze_results.py \
    --scores output/fp_scores_contrastive.json \
    --labels output/labels.json \
    --scores-alt output/fp_scores_posonly.json \
    --output output/report.txt
```

## Data Format Specs

### Training data (positive / negative items)

JSON file — either an array of items, or an object with an `"items"` key:

```json
{
  "source": "retailer-name.com",
  "category": "Jackets & Coats",
  "items": [
    {
      "id": 1,
      "name": "Item Name",
      "price_usd": 150,
      "hero_img_path": "images/item1.jpg",
      "hero_img_url": "https://example.com/item1.jpg",
      "visual_description": "optional text description",
      "key_details": "optional details"
    }
  ]
}
```

Required fields per item: `id`, `name`, `price_usd` (or `price`).

Optional image fields (checked in this order):
- `hero_img_path` — local file path
- `hero_img_url` — remote URL (downloaded automatically)
- `hero_img` — URL or local path (auto-detected)

Second image: `hero_img2_path`, `hero_img2_url`, or `hero_img2`.

### Test catalog

Same format as training data. Can also be a plain array:

```json
[
  {
    "id": "FP_001",
    "name": "Devon Striped Balloon Jacket",
    "price": "$168.00",
    "img_url": "https://example.com/fp001.jpg"
  }
]
```

### Scores (output of score_catalog.py)

```json
[
  {
    "id": "FP_001",
    "name": "Devon Striped Balloon Jacket",
    "tier": 3,
    "rationale": "Cropped bomber-adjacent silhouette aligns, but..."
  }
]
```

### Labels (output of blind evaluation)

```json
{
  "FP_001": true,
  "FP_002": false,
  "BR_001": true
}
```

`true` = user would explore (click), `false` = user would skip.

## Configuration

Edit `scripts/config.yaml` to change defaults:

```yaml
model: "claude-sonnet-4-5-20250929"
num_agents: 5
tier_schema:
  1: "Strong Match ..."
  2: "Moderate Match ..."
  3: "Weak Match ..."
  4: "No Match ..."
```

All config values can be overridden via CLI args.

## Cost Estimates

| Model | 5 agents, 100 items | 3 agents, 50 items |
|-------|--------------------|--------------------|
| Sonnet | ~$15-20 | ~$5-8 |
| Haiku | ~$2-4 | ~$1-2 |
| Opus | ~$40-60 | ~$15-25 |

Costs are dominated by the scoring phase (one API call per item per profile). Use Haiku for scoring and Sonnet/Opus for profile synthesis to optimize cost.
