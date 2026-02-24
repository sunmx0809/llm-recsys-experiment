"""
Exact prompt templates from the LLM recommendation experiment notebook.

These prompts are used across the CLI scripts for profile synthesis,
merging, and scoring. They are extracted verbatim from the original
notebook (llm_recsys_notebook.ipynb).
"""

# ── Phase 2A: Positive-Only Synthesis ────────────────────────────────

POSITIVE_ONLY_SYSTEM_PROMPT = """You are a style analyst specializing in inferring latent
fashion preferences from implicit behavioral signals. You will receive a set of items
that a user clicked on (chose to explore further) while browsing a large product
category page. Your task is to infer the user's underlying style preferences.

For each preference dimension you identify, classify your confidence:
- STRONG PREFERENCES: Consistent patterns across most items (high confidence)
- MEDIUM CONFIDENCE: Appears in some items, plausible pattern
- ANTI-PREFERENCES: What the user likely avoids (inferred from absence)

Be specific and actionable. Instead of "likes casual styles", say "prefers relaxed/
oversized bomber silhouettes in cropped lengths with textural complexity."

Analyze the IMAGES carefully — visual signals (texture, drape, color temperature,
styling context) are often more informative than text descriptions."""


def build_positive_only_user_content(items, build_item_content_blocks_fn):
    """Build the user message content blocks for positive-only synthesis.

    Args:
        items: list of positive (clicked) item dicts.
        build_item_content_blocks_fn: callable(item) -> list of content blocks.
    """
    content = []
    content.append({
        "type": "text",
        "text": (
            f"A user browsed a product category page and selected "
            f"these {len(items)} items (clicked to view detail page). "
            f"Infer the user's implicit style preferences from ONLY these positive signals.\n\n"
            f"For each item below, you'll see the name, price, and the hero images "
            f"visible on the category browse page (what the user saw before clicking).\n"
        ),
    })
    for item in items:
        content.extend(build_item_content_blocks_fn(item))
    content.append({
        "type": "text",
        "text": (
            "\n\nBased on these clicked items, produce a comprehensive User Preference Profile "
            "with: STRONG PREFERENCES, MEDIUM CONFIDENCE PREFERENCES, and ANTI-PREFERENCES "
            "(inferred from what's absent in the selections). "
            "Cover: silhouette, material/texture, color palette, closure type, "
            "embellishment/details, price behavior, and overall aesthetic."
        ),
    })
    return content


# ── Phase 2B: Contrastive Synthesis ──────────────────────────────────

CONTRASTIVE_SYSTEM_PROMPT = """You are a style analyst specializing in inferring latent
fashion preferences from implicit behavioral signals. You will receive two sets of items:

1. POSITIVE items: items the user clicked on (chose to explore further)
2. NEGATIVE items: items the user saw on the same page but did NOT click

Use both signals to build a nuanced preference profile. The negative items are especially
valuable for:
- Sharpening boundaries within liked categories (e.g., "likes leather BUT NOT in moto style")
- Adding conditional logic ("likes embroidery ONLY in structured silhouettes")
- Confirming anti-preferences with direct evidence

For each preference, classify confidence and note whether it comes from positive signal,
negative signal, or both.

Analyze the IMAGES carefully — visual signals are often more informative than text."""


def build_contrastive_user_content(pos_items, neg_items, build_item_content_blocks_fn):
    """Build user message for contrastive synthesis.

    Args:
        pos_items: list of positive (clicked) item dicts.
        neg_items: list of negative (skipped) item dicts.
        build_item_content_blocks_fn: callable(item) -> list of content blocks.
    """
    content = []
    content.append({
        "type": "text",
        "text": (
            f"A user browsed a product category page. "
            f"They clicked on {len(pos_items)} items (POSITIVE signal) and saw but "
            f"did NOT click on {len(neg_items)} other items (NEGATIVE signal).\n\n"
            f"── POSITIVE ITEMS (user clicked to view) ──\n"
        ),
    })
    for item in pos_items:
        content.extend(build_item_content_blocks_fn(item))

    content.append({"type": "text", "text": "\n── NEGATIVE ITEMS (user saw but skipped) ──\n"})
    for item in neg_items:
        content.extend(build_item_content_blocks_fn(item))

    content.append({
        "type": "text",
        "text": (
            "\n\nProduce a User Preference Profile using BOTH positive and negative signals. "
            "For each preference: state it, note the evidence (which positive items support it, "
            "which negative items confirm the boundary), and classify confidence.\n"
            "Include: STRONG PREFERENCES, CONDITIONAL PREFERENCES (likes X only when Y), "
            "and ANTI-PREFERENCES (with specific negative-item evidence)."
        ),
    })
    return content


# ── Merge Prompt ─────────────────────────────────────────────────────

MERGE_SYSTEM_PROMPT_TEMPLATE = """You are given {n} independent style analyses of the same user's
fashion preferences. Each was produced by a separate analyst seeing the same data.

Produce a single merged User Preference Profile that:
1. Notes consensus level for each preference (e.g., "5/5 agents agreed" vs "3/5")
2. Elevates high-consensus findings to STRONG PREFERENCES
3. Flags disagreements or low-consensus items as SPECULATION
4. Preserves specific, actionable language (not vague summaries)
5. Includes ANTI-PREFERENCES with consensus counts

Format the output as a structured markdown document with clear sections."""


def build_merge_system_prompt(n):
    """Return the merge system prompt with the agent count filled in."""
    return MERGE_SYSTEM_PROMPT_TEMPLATE.format(n=n)


def build_merge_user_content(syntheses):
    """Build the user message for merging N agent syntheses.

    Args:
        syntheses: list of strings, each an agent's synthesis output.
    """
    agent_texts = "\n\n---\n\n".join(
        [f"## Agent {i+1} Analysis\n\n{s}" for i, s in enumerate(syntheses)]
    )
    return agent_texts


# ── Phase 3: Scoring ─────────────────────────────────────────────────

SCORING_SYSTEM_PROMPT = """You are a recommendation engine. You have a detailed user
preference profile and must score new items on how well they match.

Score each item on this 4-tier scale:
- Tier 1 (Strong Match): High confidence the user would click to explore this item
- Tier 2 (Moderate Match): Aligns with several preferences but has notable gaps
- Tier 3 (Weak Match): One or two alignment points, but overall not a fit
- Tier 4 (No Match): Conflicts with core preferences or hits anti-preferences

For each item, provide:
1. The tier (1-4)
2. A 1-2 sentence rationale citing specific preference dimensions

Be calibrated: Tier 1 should be rare (~5-10% of items). Most items should be Tier 3-4.
Analyze the IMAGE carefully — visual match matters more than text description match."""


def build_scoring_user_content(item, preference_brief, item_content_blocks):
    """Build the user message for scoring a single item.

    Args:
        item: dict with at least 'name' and 'price'.
        preference_brief: the merged preference profile text.
        item_content_blocks: list of content blocks (text + images) for the item.
    """
    content = []
    content.append({
        "type": "text",
        "text": f"## User Preference Profile\n\n{preference_brief}",
    })
    content.append({
        "type": "text",
        "text": f"\n## Item to Score\n\nName: {item['name']}\nPrice: {item.get('price', item.get('price_usd', 'N/A'))}",
    })
    # Append image blocks (skip the text block from build_item_content_blocks
    # since we already have the name/price above)
    for block in item_content_blocks:
        if block["type"] == "image":
            content.append(block)
    content.append({
        "type": "text",
        "text": "\nScore this item. Respond in this exact format:\nTier: [1-4]\nRationale: [your reasoning]",
    })
    return content
