#!/usr/bin/env python3
"""
Score a test catalog against a user preference profile.

Each item is scored on a 4-tier scale:
  Tier 1 — Strong Match
  Tier 2 — Moderate Match
  Tier 3 — Weak Match
  Tier 4 — No Match

Examples:
    python scripts/score_catalog.py \
        --catalog fp_catalog.json \
        --profile output/phase2b/preference_brief.md \
        --output scores.json

    python scripts/score_catalog.py \
        --catalog fp_catalog.json \
        --profile output/phase2b/preference_brief.md \
        --images-dir phase3_images/fp \
        --model claude-haiku-4-5-20251001 \
        --output scores.json
"""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import anthropic

from _prompts import SCORING_SYSTEM_PROMPT, build_scoring_user_content
from _utils import build_item_content_blocks, load_config, load_items, _log


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score catalog items against a user preference profile.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Output format:\n"
            "  JSON array of {id, name, tier, rationale} for each item.\n"
        ),
    )
    parser.add_argument(
        "--catalog", required=True, metavar="PATH",
        help="JSON file of items to score (array or object with 'items').",
    )
    parser.add_argument(
        "--profile", required=True, metavar="PATH",
        help="Path to the merged preference brief (.md file).",
    )
    parser.add_argument(
        "--model", default=None, metavar="MODEL",
        help="Anthropic model ID (default: from config.yaml).",
    )
    parser.add_argument(
        "--images-dir", default=None, metavar="DIR",
        help="Base directory containing item images (looked up by item ID).",
    )
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help="Path to config.yaml (default: scripts/config.yaml).",
    )
    parser.add_argument(
        "--output", required=True, metavar="PATH",
        help="Output path for the scores JSON file.",
    )
    return parser.parse_args()


def score_single_item(client, model, item, preference_brief, item_content_blocks):
    """Score a single catalog item against a preference profile.

    Returns:
        dict with keys: id, name, tier (int or None), rationale (str).
    """
    user_content = build_scoring_user_content(
        item, preference_brief, item_content_blocks,
    )

    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=SCORING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    text = response.content[0].text

    # Parse tier
    tier_match = re.search(r"Tier:\s*(\d)", text)
    tier = int(tier_match.group(1)) if tier_match else None

    # Parse rationale
    rationale_match = re.search(r"Rationale:\s*(.+)", text, re.DOTALL)
    rationale = rationale_match.group(1).strip() if rationale_match else text

    return {
        "id": item.get("id", "?"),
        "name": item["name"],
        "tier": tier,
        "rationale": rationale,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model = args.model or cfg["model"]

    # Load catalog and profile
    catalog = load_items(args.catalog)
    preference_brief = Path(args.profile).read_text()
    _log(f"Catalog: {len(catalog)} items from {args.catalog}")
    _log(f"Profile: {len(preference_brief)} chars from {args.profile}")
    _log(f"Model: {model}")

    images_dir = args.images_dir
    temp_dir = tempfile.mkdtemp(prefix="llm_recsys_imgs_")
    client = anthropic.Anthropic()

    # Score each item
    results = []
    for i, item in enumerate(catalog):
        blocks = build_item_content_blocks(
            item, images_dir=images_dir, temp_dir=temp_dir,
        )
        result = score_single_item(client, model, item, preference_brief, blocks)
        results.append(result)

        tier_label = f"Tier {result['tier']}" if result["tier"] else "???"
        _log(f"  [{i+1}/{len(catalog)}] {result['id']}: {tier_label} — {item['name'][:50]}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    _log(f"\nScores saved to: {output_path}")

    # Print tier distribution summary
    from collections import Counter
    tier_counts = Counter(r["tier"] for r in results)
    _log("\nTier distribution:")
    for t in [1, 2, 3, 4]:
        count = tier_counts.get(t, 0)
        pct = count / len(results) * 100 if results else 0
        _log(f"  Tier {t}: {count:>3} items ({pct:.0f}%)")
    if None in tier_counts:
        _log(f"  Parse errors: {tier_counts[None]}")


if __name__ == "__main__":
    main()
