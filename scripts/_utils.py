"""
Shared utility functions for the LLM recommendation experiment CLI scripts.
"""

import base64
import mimetypes
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import yaml


def image_to_content_block(image_path):
    """Convert a local image file to an Anthropic API content block.

    Args:
        image_path: path to a local image file (str or Path).

    Returns:
        dict suitable for use in the Anthropic messages API content list.

    Raises:
        FileNotFoundError: if the image file does not exist.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime, "data": b64},
    }


def build_item_content_blocks(item, images_dir=None, temp_dir=None):
    """Build a list of content blocks (text + images) for one catalog item.

    Tries these image sources in order:
      1. item["hero_img_path"] — absolute or relative local path
      2. <images_dir>/<item_id>.{jpg,png,webp} — conventional naming
      3. item["hero_img_url"] — remote URL (downloaded to temp_dir)

    Also handles hero_img2_path / hero_img2_url for a second image.

    Args:
        item: dict with at least 'name' and ('price' or 'price_usd').
        images_dir: optional base directory to look for images by item id.
        temp_dir: optional directory to cache downloaded images.

    Returns:
        list of content block dicts (text and/or image).
    """
    blocks = []

    # Build text description
    item_id = item.get("id", "?")
    price = item.get("price", item.get("price_usd", "?"))
    if isinstance(price, (int, float)):
        price = f"${price}"
    text_desc = f"Item {item_id}: {item['name']} — {price}"
    if "visual_description" in item:
        text_desc += f"\nVisual: {item['visual_description']}"
    if "key_details" in item:
        text_desc += f"\nDetails: {item['key_details']}"
    blocks.append({"type": "text", "text": text_desc})

    # Resolve and attach images
    for path_key, url_key in [
        ("hero_img_path", "hero_img_url"),
        ("hero_img2_path", "hero_img2_url"),
    ]:
        img_path = _resolve_image(item, path_key, url_key, item_id, images_dir, temp_dir)
        if img_path:
            try:
                blocks.append(image_to_content_block(img_path))
            except FileNotFoundError:
                _log(f"Warning: image not found: {img_path}")

    return blocks


def _resolve_image(item, path_key, url_key, item_id, images_dir, temp_dir):
    """Try to find a local image path for an item. Returns path or None."""
    # 1. Explicit local path in item dict
    if path_key in item and item[path_key]:
        p = Path(item[path_key])
        if p.exists():
            return str(p)

    # 2. Convention-based lookup in images_dir (only for primary image)
    if images_dir and path_key == "hero_img_path":
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidate = Path(images_dir) / f"{item_id}{ext}"
            if candidate.exists():
                return str(candidate)
        # Also try _alt for second image
    if images_dir and path_key == "hero_img2_path":
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidate = Path(images_dir) / f"{item_id}_alt{ext}"
            if candidate.exists():
                return str(candidate)

    # 3. Download from URL
    if url_key in item and item[url_key]:
        return _download_image(item[url_key], item_id, path_key, temp_dir)

    # Also check bare "hero_img" / "hero_img2" keys (URL stored without _url suffix)
    bare_key = path_key.replace("_path", "")
    if bare_key in item and item[bare_key]:
        val = item[bare_key]
        if val.startswith("http://") or val.startswith("https://"):
            return _download_image(val, item_id, path_key, temp_dir)
        elif Path(val).exists():
            return val

    return None


def _download_image(url, item_id, path_key, temp_dir):
    """Download an image from a URL to a temp directory. Returns local path or None."""
    if not temp_dir:
        temp_dir = tempfile.mkdtemp(prefix="llm_recsys_imgs_")
    suffix = "_alt" if "2" in path_key else ""
    ext = ".jpg"
    for candidate_ext in [".jpg", ".png", ".webp", ".gif"]:
        if candidate_ext in url.lower():
            ext = candidate_ext
            break
    local_path = Path(temp_dir) / f"{item_id}{suffix}{ext}"
    if local_path.exists():
        return str(local_path)
    try:
        _log(f"Downloading image for {item_id}{suffix} ...")
        urllib.request.urlretrieve(url, str(local_path))
        return str(local_path)
    except Exception as e:
        _log(f"Warning: failed to download {url}: {e}")
        return None


def load_config(config_path=None):
    """Load configuration from a YAML file, with defaults.

    Args:
        config_path: path to config.yaml. If None, looks for config.yaml
                     in the same directory as this file.

    Returns:
        dict with keys: model, num_agents, tier_schema.
    """
    defaults = {
        "model": "claude-sonnet-4-5-20250929",
        "num_agents": 5,
        "tier_schema": {
            1: "Strong Match — high confidence the user would click to explore this item",
            2: "Moderate Match — aligns with several preferences but has notable gaps",
            3: "Weak Match — one or two alignment points, but overall not a fit",
            4: "No Match — conflicts with core preferences or hits anti-preferences",
        },
    }
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    if Path(config_path).exists():
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        for k, v in user_cfg.items():
            defaults[k] = v
    return defaults


def load_items(path):
    """Load items from a JSON file.

    Accepts either:
      - A JSON array of items directly, e.g. [{...}, {...}]
      - A JSON object with an "items" key, e.g. {"items": [{...}]}

    Returns:
        list of item dicts.
    """
    import json
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    raise ValueError(
        f"Expected a JSON array or object with 'items' key, got: {type(data).__name__}"
    )


def _log(msg):
    """Print a message to stderr (progress/debug output)."""
    print(msg, file=sys.stderr)
