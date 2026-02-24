#!/usr/bin/env python3
"""
Analyze recommendation results: compute precision, recall, lift, tier
calibration, and confusion matrix from LLM scores + user labels.

Optionally compares two scoring runs side-by-side (e.g., positive-only
vs. contrastive profiles).

Examples:
    # Single scoring analysis
    python scripts/analyze_results.py \
        --scores scores.json \
        --labels labels.json

    # Side-by-side comparison
    python scripts/analyze_results.py \
        --scores scores_contrastive.json \
        --labels labels.json \
        --scores-alt scores_posonly.json

    # Save report to file
    python scripts/analyze_results.py \
        --scores scores.json \
        --labels labels.json \
        --output report.txt
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute precision, recall, lift, and tier calibration from scores + labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Labels JSON format:\n"
            '  {"item_id": true, "item_id2": false, ...}\n'
            "  true = user would explore (click), false = user would skip.\n"
        ),
    )
    parser.add_argument(
        "--scores", required=True, metavar="PATH",
        help="JSON file of scored items (from score_catalog.py).",
    )
    parser.add_argument(
        "--labels", required=True, metavar="PATH",
        help="JSON file of user labels: {item_id: true/false}.",
    )
    parser.add_argument(
        "--scores-alt", default=None, metavar="PATH",
        help="Optional second scores file for side-by-side comparison.",
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Save the report to a file (in addition to printing to stdout).",
    )
    return parser.parse_args()


def load_scores(path):
    """Load scores JSON. Returns list of {id, name, tier, rationale}."""
    with open(path) as f:
        return json.load(f)


def load_labels(path):
    """Load user labels JSON. Returns dict {item_id_str: bool}."""
    with open(path) as f:
        data = json.load(f)
    # Normalize keys to strings
    return {str(k): bool(v) for k, v in data.items()}


def compute_metrics(scores, labels, name=""):
    """Compute full metrics for one scoring run.

    A "recommendation" is defined as Tier 1 or Tier 2.

    Returns:
        dict with keys: tp, fp, fn, tn, precision, recall, lift,
        base_rate, n_recs, n_total, tier_calibration, name.
    """
    results = []
    for item in scores:
        item_id = str(item["id"])
        tier = item.get("tier")
        clicked = labels.get(item_id)
        if clicked is None:
            # Item not in labels — skip
            continue
        results.append({
            "id": item_id,
            "name": item.get("name", ""),
            "tier": tier,
            "is_rec": tier in (1, 2),
            "user_clicked": clicked,
        })

    if not results:
        return None

    n_total = len(results)
    n_clicked = sum(1 for r in results if r["user_clicked"])
    base_rate = n_clicked / n_total if n_total > 0 else 0

    recs = [r for r in results if r["is_rec"]]
    non_recs = [r for r in results if not r["is_rec"]]

    tp = sum(1 for r in recs if r["user_clicked"])
    fp = sum(1 for r in recs if not r["user_clicked"])
    fn = sum(1 for r in non_recs if r["user_clicked"])
    tn = sum(1 for r in non_recs if not r["user_clicked"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    lift = precision / base_rate if base_rate > 0 else 0

    # Tier calibration
    tier_cal = {}
    for t in [1, 2, 3, 4]:
        tier_items = [r for r in results if r["tier"] == t]
        n = len(tier_items)
        clicks = sum(1 for r in tier_items if r["user_clicked"])
        tier_cal[t] = {
            "n_items": n,
            "n_clicked": clicks,
            "click_rate": clicks / n if n > 0 else 0,
        }

    # False positives and false negatives for error analysis
    false_positives = [r for r in recs if not r["user_clicked"]]
    false_negatives = [r for r in non_recs if r["user_clicked"]]

    return {
        "name": name,
        "n_total": n_total,
        "n_clicked": n_clicked,
        "base_rate": base_rate,
        "n_recs": len(recs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "lift": lift,
        "tier_calibration": tier_cal,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "all_results": results,
    }


def format_report(metrics, alt_metrics=None):
    """Format a text report from computed metrics.

    Args:
        metrics: dict from compute_metrics (primary scoring).
        alt_metrics: optional dict from compute_metrics (comparison scoring).

    Returns:
        Formatted report string.
    """
    lines = []
    sep = "=" * 60

    def section(title):
        lines.append("")
        lines.append(sep)
        lines.append(title)
        lines.append(sep)

    # ── Header ───────────────────────────────────────────────────
    lines.append(sep)
    lines.append("LLM RECOMMENDATION EXPERIMENT — ANALYSIS REPORT")
    lines.append(sep)
    lines.append(f"Total items evaluated: {metrics['n_total']}")
    lines.append(f"Total user clicks: {metrics['n_clicked']}")
    lines.append(f"Base click rate: {metrics['base_rate']:.1%}")

    # ── Tier Calibration ─────────────────────────────────────────
    def print_tier_cal(m):
        lines.append(f"{'Tier':<8} {'Items':>6} {'Clicks':>7} {'Click Rate':>11}")
        lines.append("-" * 34)
        for t in [1, 2, 3, 4]:
            cal = m["tier_calibration"][t]
            rate = f"{cal['click_rate']:.1%}" if cal["n_items"] > 0 else "N/A"
            lines.append(
                f"Tier {t:<3} {cal['n_items']:>6} {cal['n_clicked']:>7} {rate:>11}"
            )

    section(f"TIER CALIBRATION: {metrics.get('name', 'Primary')}")
    print_tier_cal(metrics)

    if alt_metrics:
        section(f"TIER CALIBRATION: {alt_metrics.get('name', 'Alternate')}")
        print_tier_cal(alt_metrics)

    # ── Precision / Recall / Lift ────────────────────────────────
    def print_metrics(m):
        lines.append(f"Recommendations (T1+T2): {m['n_recs']} items")
        lines.append(f"Non-recommendations (T3+T4): {m['n_total'] - m['n_recs']} items")
        lines.append("")
        lines.append(f"True Positives:  {m['tp']:>3}  (recommended & user clicked)")
        lines.append(f"False Positives: {m['fp']:>3}  (recommended but user skipped)")
        lines.append(f"False Negatives: {m['fn']:>3}  (not recommended but user clicked)")
        lines.append(f"True Negatives:  {m['tn']:>3}  (not recommended & user skipped)")
        lines.append("")
        lines.append(
            f"Precision: {m['precision']:.1%}  "
            f"(of items we recommended, how many did the user like?)"
        )
        lines.append(
            f"Recall:    {m['recall']:.1%}  "
            f"(of items the user liked, how many did we recommend?)"
        )
        lines.append(
            f"Lift:      {m['lift']:.2f}x  "
            f"(vs. {m['base_rate']:.1%} base click rate)"
        )

    section(f"METRICS: {metrics.get('name', 'Primary')}")
    print_metrics(metrics)

    if alt_metrics:
        section(f"METRICS: {alt_metrics.get('name', 'Alternate')}")
        print_metrics(alt_metrics)

    # ── Side-by-side comparison ──────────────────────────────────
    if alt_metrics:
        section("HEAD-TO-HEAD COMPARISON")
        header = f"{'Metric':<14} {metrics.get('name', 'Primary'):>18} {alt_metrics.get('name', 'Alternate'):>18}"
        lines.append(header)
        lines.append("-" * len(header))
        rows = [
            ("Precision", f"{metrics['precision']:.1%}", f"{alt_metrics['precision']:.1%}"),
            ("Recall", f"{metrics['recall']:.1%}", f"{alt_metrics['recall']:.1%}"),
            ("Lift", f"{metrics['lift']:.2f}x", f"{alt_metrics['lift']:.2f}x"),
            ("T1+T2 items", str(metrics['n_recs']), str(alt_metrics['n_recs'])),
            ("True Pos", str(metrics['tp']), str(alt_metrics['tp'])),
            ("False Pos", str(metrics['fp']), str(alt_metrics['fp'])),
            ("False Neg", str(metrics['fn']), str(alt_metrics['fn'])),
        ]
        for label, v1, v2 in rows:
            lines.append(f"{label:<14} {v1:>18} {v2:>18}")

        # Agreement analysis
        primary_by_id = {str(r["id"]): r["tier"] for r in metrics["all_results"]}
        alt_by_id = {str(r["id"]): r["tier"] for r in alt_metrics["all_results"]}
        common_ids = set(primary_by_id.keys()) & set(alt_by_id.keys())
        if common_ids:
            exact_agree = sum(
                1 for iid in common_ids if primary_by_id[iid] == alt_by_id[iid]
            )
            within_one = sum(
                1 for iid in common_ids
                if primary_by_id[iid] is not None
                and alt_by_id[iid] is not None
                and abs(primary_by_id[iid] - alt_by_id[iid]) <= 1
            )
            lines.append("")
            lines.append(f"Exact tier agreement: {exact_agree}/{len(common_ids)} ({exact_agree/len(common_ids):.1%})")
            lines.append(f"Within 1 tier: {within_one}/{len(common_ids)} ({within_one/len(common_ids):.1%})")

    # ── Error Analysis ───────────────────────────────────────────
    section(f"FALSE POSITIVES: {metrics.get('name', 'Primary')} — Recommended but user skipped")
    if metrics["false_positives"]:
        for r in metrics["false_positives"]:
            lines.append(f"  {r['id']}: {r['name']} (Tier {r['tier']})")
    else:
        lines.append("  (none)")

    section(f"FALSE NEGATIVES: {metrics.get('name', 'Primary')} — User clicked but not recommended")
    if metrics["false_negatives"]:
        for r in metrics["false_negatives"]:
            lines.append(f"  {r['id']}: {r['name']} (Tier {r['tier']})")
    else:
        lines.append("  (none)")

    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()

    scores = load_scores(args.scores)
    labels = load_labels(args.labels)

    print(f"Loaded {len(scores)} scores from {args.scores}", file=sys.stderr)
    print(f"Loaded {len(labels)} labels from {args.labels}", file=sys.stderr)

    # Determine name for primary scoring
    primary_name = Path(args.scores).stem
    metrics = compute_metrics(scores, labels, name=primary_name)
    if metrics is None:
        print("ERROR: No items matched between scores and labels.", file=sys.stderr)
        sys.exit(1)

    # Optional comparison
    alt_metrics = None
    if args.scores_alt:
        alt_scores = load_scores(args.scores_alt)
        alt_name = Path(args.scores_alt).stem
        alt_metrics = compute_metrics(alt_scores, labels, name=alt_name)
        print(f"Loaded {len(alt_scores)} alt scores from {args.scores_alt}", file=sys.stderr)

    report = format_report(metrics, alt_metrics)

    # Print to stdout
    print(report)

    # Optionally save to file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"\nReport saved to: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
