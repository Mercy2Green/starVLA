"""Plot offline LAST probe perturbation results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per_sample_jsonl", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_summary(path: str) -> List[Dict[str, str]]:
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def plot_bar(labels: List[str], values: List[float], ylabel: str, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_grouped(labels: List[str], series: Dict[str, List[float]], ylabel: str, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    names = list(series.keys())
    width = 0.8 / max(1, len(names))
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    for idx, name in enumerate(names):
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        ax.bar(offsets, series[name], width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def success_label(value: Any) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "success"}:
            return "success"
        if lowered in {"0", "false", "no", "failure", "fail"}:
            return "failure"
    return "success" if bool(value) else "failure"


def print_text_summary(summary_rows: List[Dict[str, str]]) -> None:
    mode_to_l1 = {
        row["mode"]: float(row["mean_action_delta_l1"])
        for row in summary_rows
        if row.get("mean_action_delta_l1") not in (None, "")
    }
    print("mean action_delta_l1 by mode:")
    for mode, value in sorted(mode_to_l1.items()):
        print(f"  {mode}: {value:.6f}")

    if "mask_top" in mode_to_l1 and "mask_low" in mode_to_l1:
        print(f"mask_top > mask_low: {mode_to_l1['mask_top'] > mode_to_l1['mask_low']}")
    if "crop_top" in mode_to_l1 and "mask_top" in mode_to_l1:
        relation = "more" if mode_to_l1["crop_top"] > mode_to_l1["mask_top"] else "less/equal"
        print(f"crop_top changes action {relation} than mask_top")


def run(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.per_sample_jsonl)
    summary = read_summary(args.summary_csv)

    labels = [row["mode"] for row in summary]
    l1 = [float(row["mean_action_delta_l1"]) for row in summary]
    plot_bar(labels, l1, "mean action_delta_l1", "Action Delta by Mode", output_dir / "action_delta_by_mode.png")

    domains = sorted({record.get("domain", "") for record in records})
    if len(domains) > 1:
        entropy_by_domain = []
        top5_by_domain = []
        top10_by_domain = []
        for domain in domains:
            subset = [record for record in records if record.get("domain", "") == domain and record.get("mode") == "none"]
            if not subset:
                subset = [record for record in records if record.get("domain", "") == domain]
            entropy_by_domain.append(mean([float(record["entropy"]) for record in subset if record.get("entropy") is not None]))
            top5_by_domain.append(mean([float(record["top5_mass"]) for record in subset if record.get("top5_mass") is not None]))
            top10_by_domain.append(mean([float(record["top10_mass"]) for record in subset if record.get("top10_mass") is not None]))
        plot_bar(domains, entropy_by_domain, "mean entropy", "LAST Entropy by Domain", output_dir / "entropy_by_domain.png")
        plot_grouped(
            domains,
            {"top5_mass": top5_by_domain, "top10_mass": top10_by_domain},
            "mean mass",
            "Top Mass by Domain",
            output_dir / "top_mass_by_domain.png",
        )

    if any("success" in record for record in records):
        grouped = defaultdict(list)
        for record in records:
            if record.get("mode") != "none" or record.get("entropy") is None or "success" not in record:
                continue
            grouped[success_label(record["success"])].append(float(record["entropy"]))
        if grouped:
            success_labels = sorted(grouped.keys())
            values = [mean(grouped[label]) for label in success_labels]
            plot_bar(
                success_labels,
                values,
                "mean entropy",
                "Success vs Failure Entropy",
                output_dir / "success_vs_failure_entropy.png",
            )

    print_text_summary(summary)
    print(f"Wrote figures to {output_dir}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
