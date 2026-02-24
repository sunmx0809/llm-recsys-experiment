#!/usr/bin/env python3
"""
Generate a blind evaluation HTML page for user labeling.

Creates a shuffled HTML page where a user clicks "Would Explore" or "Skip"
for each item without seeing the LLM's tier scores. At the end, the page
displays a JSON blob the user can copy-paste as ground truth labels.

Also saves a shuffle key JSON for mapping display order back to item IDs.

Example:
    python scripts/generate_eval.py \
        --scores scores.json \
        --images-dir phase3_images/fp \
        --output blind_eval.html
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _utils import _log


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a blind evaluation HTML for user labeling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scores", required=True, metavar="PATH",
        help="JSON file from score_catalog.py (array of {id, name, tier, ...}).",
    )
    parser.add_argument(
        "--images-dir", default=None, metavar="DIR",
        help=(
            "Base directory containing item images. "
            "Images are referenced as <images-dir>/<item_id>.jpg in the HTML. "
            "If not provided, items are shown without images."
        ),
    )
    parser.add_argument(
        "--catalog", default=None, metavar="PATH",
        help=(
            "Optional: original catalog JSON with image URLs/paths. "
            "Used to find image references if --images-dir is not sufficient."
        ),
    )
    parser.add_argument(
        "--output", required=True, metavar="PATH",
        help="Output path for the HTML file.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for shuffle reproducibility.",
    )
    return parser.parse_args()


def find_image_src(item_id, images_dir, catalog_lookup):
    """Find the best image source for an item.

    Returns a string suitable for an HTML <img src="..."> attribute,
    or None if no image is found.
    """
    if images_dir:
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidate = Path(images_dir) / f"{item_id}{ext}"
            if candidate.exists():
                return str(candidate)
        # Try with _alt for a second image
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidate = Path(images_dir) / f"{item_id}_alt{ext}"
            if candidate.exists():
                return str(candidate)

    # Fall back to catalog data
    if catalog_lookup and item_id in catalog_lookup:
        cat_item = catalog_lookup[item_id]
        for key in ["hero_img_path", "img_path", "hero_img_url", "img_url", "hero_img"]:
            if key in cat_item and cat_item[key]:
                return cat_item[key]

    return None


def generate_html(scored_items, images_dir, catalog_lookup, seed=None):
    """Generate the blind evaluation HTML string and shuffle key.

    Returns:
        (html_string, shuffle_key_list)
    """
    # Shuffle
    indexed = list(enumerate(scored_items))
    if seed is not None:
        random.seed(seed)
    random.shuffle(indexed)

    total = len(scored_items)

    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Blind Evaluation</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 800px;
         margin: 40px auto; padding: 0 20px; background: #f5f5f5; color: #2A2320; }}
  h1 {{ text-align: center; }}
  .instructions {{ background: white; padding: 16px 20px; border-radius: 12px;
                   margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .item {{ background: white; padding: 20px; margin: 16px 0; border-radius: 12px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
  .item img {{ max-width: 300px; max-height: 400px; margin: 8px; border-radius: 8px; }}
  .item-name {{ font-size: 14px; color: #666; margin: 8px 0; }}
  .buttons {{ margin-top: 12px; }}
  .buttons button {{ padding: 10px 24px; margin: 0 8px; border: none; border-radius: 8px;
                     cursor: pointer; font-size: 14px; transition: all 0.15s; }}
  .buttons button:hover {{ transform: scale(1.05); }}
  .btn-click {{ background: #5C7D68; color: white; }}
  .btn-skip {{ background: #D8CFBF; color: #2A2320; }}
  .done {{ opacity: 0.35; pointer-events: none; }}
  .done .buttons button {{ cursor: default; }}
  #progress {{ text-align: center; font-size: 18px; font-weight: 600;
               position: sticky; top: 0; background: #f5f5f5; padding: 12px;
               z-index: 10; border-bottom: 1px solid #ddd; }}
  #results {{ margin-top: 40px; background: white; padding: 20px; border-radius: 12px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  #results pre {{ white-space: pre-wrap; font-family: monospace; font-size: 12px;
                  background: #f0f0f0; padding: 16px; border-radius: 8px;
                  max-height: 400px; overflow: auto; }}
  .copy-btn {{ background: #5C7D68; color: white; border: none; padding: 10px 20px;
               border-radius: 8px; cursor: pointer; font-size: 14px; margin-top: 12px; }}
</style>
</head><body>
<h1>Blind Item Evaluation</h1>
<div class="instructions">
  <p><strong>Instructions:</strong> For each item below, decide whether you would
  click to explore it further. Items are in random order. Do not overthink it —
  go with your gut reaction as if browsing a product page.</p>
  <p>When you finish all {total} items, a JSON result will appear at the bottom
  for you to copy.</p>
</div>
<p id="progress">0 / {total} evaluated</p>
""")

    shuffle_key = []

    for display_pos, (orig_idx, item) in enumerate(indexed):
        item_id = item.get("id", f"item_{orig_idx}")
        item_name = item.get("name", "Unknown Item")
        shuffle_key.append({"display_pos": display_pos, "id": item_id})

        # Find image(s)
        img_html = ""
        img_src = find_image_src(str(item_id), images_dir, catalog_lookup)
        if img_src:
            img_html = f'<div><img src="{img_src}" onerror="this.style.display=\'none\'"></div>'

            # Try a second image (_alt)
            if images_dir:
                for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                    alt_path = Path(images_dir) / f"{item_id}_alt{ext}"
                    if alt_path.exists():
                        img_html = (
                            f'<div>'
                            f'<img src="{img_src}" onerror="this.style.display=\'none\'">'
                            f'<img src="{alt_path}" onerror="this.style.display=\'none\'">'
                            f'</div>'
                        )
                        break

        html_parts.append(f"""<div class="item" id="item-{display_pos}">
  <div style="color:#999;font-size:12px">Item {display_pos + 1} of {total}</div>
  {img_html}
  <div class="item-name">{item_name}</div>
  <div class="buttons">
    <button class="btn-click" onclick="record('{item_id}', true, this)">Would Explore</button>
    <button class="btn-skip" onclick="record('{item_id}', false, this)">Skip</button>
  </div>
</div>
""")

    html_parts.append(f"""
<div id="results" style="display:none">
  <h3>Evaluation Complete</h3>
  <p>Copy the JSON below and save it to a file (e.g., <code>labels.json</code>).
  Then use it with <code>analyze_results.py</code>.</p>
  <button class="copy-btn" onclick="copyResults()">Copy to Clipboard</button>
  <pre id="results-json"></pre>
</div>

<script>
const results = {{}};
let count = 0;
const total = {total};

function record(id, clicked, btn) {{
  if (results[id] !== undefined) return;
  results[id] = clicked;
  btn.closest('.item').classList.add('done');
  count++;
  document.getElementById('progress').textContent = count + ' / ' + total + ' evaluated';
  if (count === total) {{
    document.getElementById('results').style.display = 'block';
    document.getElementById('results-json').textContent = JSON.stringify(results, null, 2);
    document.getElementById('progress').textContent = 'All done! Scroll down for results.';
  }}
}}

function copyResults() {{
  const text = document.getElementById('results-json').textContent;
  navigator.clipboard.writeText(text).then(() => {{
    document.querySelector('.copy-btn').textContent = 'Copied!';
    setTimeout(() => {{ document.querySelector('.copy-btn').textContent = 'Copy to Clipboard'; }}, 2000);
  }});
}}
</script>
</body></html>""")

    return "".join(html_parts), shuffle_key


def main():
    args = parse_args()

    # Load scores
    with open(args.scores) as f:
        scored_items = json.load(f)
    _log(f"Loaded {len(scored_items)} scored items from {args.scores}")

    # Optionally load catalog for image URLs
    catalog_lookup = {}
    if args.catalog:
        from _utils import load_items
        cat_items = load_items(args.catalog)
        catalog_lookup = {str(item.get("id", "")): item for item in cat_items}
        _log(f"Loaded catalog with {len(catalog_lookup)} items for image lookup")

    # Generate HTML
    html, shuffle_key = generate_html(
        scored_items,
        images_dir=args.images_dir,
        catalog_lookup=catalog_lookup,
        seed=args.seed,
    )

    # Save HTML
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    _log(f"Blind evaluation HTML saved to: {output_path}")

    # Save shuffle key
    key_path = output_path.with_name(output_path.stem + "_shuffle_key.json")
    with open(key_path, "w") as f:
        json.dump(shuffle_key, f, indent=2)
    _log(f"Shuffle key saved to: {key_path}")

    _log(f"\nOpen {output_path} in a browser to start the evaluation.")


if __name__ == "__main__":
    main()
