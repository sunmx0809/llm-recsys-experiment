#!/usr/bin/env python3
"""
Synthesize a user preference profile from training data.

Runs a multi-agent ensemble: N independent LLM agents analyze the same
training data, then a merge step produces a consensus preference brief.

Phase 2A (positive-only): provide only --positive
Phase 2B (contrastive):   provide both --positive and --negative

Examples:
    python scripts/synthesize_profile.py \
        --positive training_data.json \
        --output output/phase2a

    python scripts/synthesize_profile.py \
        --positive training_data.json \
        --negative negative_samples.json \
        --num-agents 3 \
        --output output/phase2b
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Allow running from the scripts/ directory or the repo root
sys.path.insert(0, str(Path(__file__).parent))

import anthropic

from _prompts import (
    POSITIVE_ONLY_SYSTEM_PROMPT,
    CONTRASTIVE_SYSTEM_PROMPT,
    build_positive_only_user_content,
    build_contrastive_user_content,
    build_merge_system_prompt,
    build_merge_user_content,
)
from _utils import build_item_content_blocks, load_config, load_items, _log


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize a user preference profile via multi-agent LLM ensemble.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Input JSON format:\n"
            "  Array of items, or object with 'items' key.\n"
            "  Each item: {id, name, price or price_usd, hero_img_path?, hero_img_url?}\n"
        ),
    )
    parser.add_argument(
        "--positive", required=True, metavar="PATH",
        help="JSON file of clicked (positive) items.",
    )
    parser.add_argument(
        "--negative", metavar="PATH", default=None,
        help="JSON file of skipped (negative) items. If provided, runs contrastive synthesis.",
    )
    parser.add_argument(
        "--num-agents", type=int, default=None, metavar="N",
        help="Number of independent agents in the ensemble (default: from config.yaml).",
    )
    parser.add_argument(
        "--model", default=None, metavar="MODEL",
        help="Anthropic model ID (default: from config.yaml).",
    )
    parser.add_argument(
        "--images-dir", default=None, metavar="DIR",
        help="Base directory to look for item images by ID.",
    )
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help="Path to config.yaml (default: scripts/config.yaml).",
    )
    parser.add_argument(
        "--output", required=True, metavar="DIR",
        help="Output directory. Agent outputs and merged brief are saved here.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model = args.model or cfg["model"]
    num_agents = args.num_agents or cfg["num_agents"]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.images_dir

    # Create a temp dir for any downloaded images
    temp_dir = tempfile.mkdtemp(prefix="llm_recsys_imgs_")

    # Load items
    positive_items = load_items(args.positive)
    _log(f"Loaded {len(positive_items)} positive items from {args.positive}")

    negative_items = None
    if args.negative:
        negative_items = load_items(args.negative)
        _log(f"Loaded {len(negative_items)} negative items from {args.negative}")

    is_contrastive = negative_items is not None
    mode_label = "contrastive" if is_contrastive else "positive-only"
    _log(f"Mode: {mode_label}")
    _log(f"Model: {model}")
    _log(f"Agents: {num_agents}")

    # Helper to build content blocks with image resolution
    def item_blocks(item):
        return build_item_content_blocks(item, images_dir=images_dir, temp_dir=temp_dir)

    # Build the user content for the synthesis prompt
    if is_contrastive:
        system_prompt = CONTRASTIVE_SYSTEM_PROMPT
        user_content = build_contrastive_user_content(
            positive_items, negative_items, item_blocks,
        )
    else:
        system_prompt = POSITIVE_ONLY_SYSTEM_PROMPT
        user_content = build_positive_only_user_content(positive_items, item_blocks)

    # Initialize the Anthropic client
    client = anthropic.Anthropic()

    # ── Run N independent agents ─────────────────────────────────────
    syntheses = []
    for i in range(num_agents):
        agent_num = i + 1
        _log(f"Running agent {agent_num}/{num_agents} ...")
        response = client.messages.create(
            model=model,
            max_tokens=6000 if is_contrastive else 4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        synthesis = response.content[0].text
        syntheses.append(synthesis)

        # Save individual agent output
        agent_path = output_dir / f"agent{agent_num}_synthesis.md"
        agent_path.write_text(synthesis)
        _log(f"  Agent {agent_num} done — {len(synthesis)} chars → {agent_path}")

    # ── Merge into consensus profile ─────────────────────────────────
    _log(f"Merging {num_agents} agent outputs ...")
    merge_system = build_merge_system_prompt(num_agents)
    merge_user = build_merge_user_content(syntheses)

    response = client.messages.create(
        model=model,
        max_tokens=6000,
        system=merge_system,
        messages=[{"role": "user", "content": merge_user}],
    )
    merged_brief = response.content[0].text

    brief_path = output_dir / "preference_brief.md"
    brief_path.write_text(merged_brief)
    _log(f"Merged preference brief saved → {brief_path}")

    # Also save metadata
    meta = {
        "mode": mode_label,
        "model": model,
        "num_agents": num_agents,
        "positive_items": len(positive_items),
        "negative_items": len(negative_items) if negative_items else 0,
        "positive_source": args.positive,
        "negative_source": args.negative,
    }
    meta_path = output_dir / "synthesis_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Print the merged brief to stdout
    print(merged_brief)
    _log(f"\nDone. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
