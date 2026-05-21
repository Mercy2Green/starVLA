import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 18), color).save(path)


def test_build_last_probe_jsonl_and_dry_run_eval(tmp_path):
    image_dir = tmp_path / "images"
    _write_image(image_dir / "a.png", (255, 0, 0))
    _write_image(image_dir / "nested" / "b.jpg", (0, 255, 0))
    (image_dir / "ignore.txt").write_text("not an image", encoding="utf-8")

    input_jsonl = tmp_path / "samples.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "starVLA/tools/build_last_probe_jsonl.py"),
            "--image_dir",
            str(image_dir),
            "--output_jsonl",
            str(input_jsonl),
            "--lang",
            "pick up the object",
            "--domain",
            "debug",
            "--limit",
            "2",
        ],
        cwd=ROOT,
        check=True,
    )

    lines = input_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert all(Path(record["image"]).is_absolute() for record in records)
    assert {record["sample_id"] for record in records} == {"a", "nested/b"}

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("framework:\n  name: QwenOFT\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "starVLA/tools/evaluate_last_probe_offline.py"),
            "--config_yaml",
            str(config_yaml),
            "--input_jsonl",
            str(input_jsonl),
            "--output_dir",
            str(output_dir),
            "--last_checkpoint",
            str(tmp_path / "fake_last.pth"),
            "--modes",
            "none,mask_top,mask_low",
            "--dry_run_fake_policy",
            "--save_first_n_visualizations",
            "0",
        ],
        cwd=ROOT,
        check=True,
    )

    per_sample = output_dir / "per_sample.jsonl"
    summary = output_dir / "summary.csv"
    assert per_sample.exists()
    assert summary.exists()

    per_sample_records = [json.loads(line) for line in per_sample.read_text(encoding="utf-8").strip().splitlines()]
    assert len(per_sample_records) == 6
    assert {record["mode"] for record in per_sample_records} == {"none", "mask_top", "mask_low"}
    assert all(record["action_shape"] == [1, 4, 7] for record in per_sample_records)

    with summary.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {row["mode"] for row in rows} == {"none", "mask_top", "mask_low"}
    assert all(int(row["num_samples"]) == 2 for row in rows)
