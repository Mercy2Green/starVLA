"""Build a one-image-per-sample JSONL file for offline LAST probe experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--domain", default="debug")
    parser.add_argument("--task", default="")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--recursive", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=True)
    return parser.parse_args()


def collect_images(image_dir: Path, recursive: bool = True) -> list[Path]:
    pattern_iter = image_dir.rglob("*") if recursive else image_dir.glob("*")
    return sorted(path for path in pattern_iter if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def sample_id_for(path: Path, image_dir: Path) -> str:
    rel = path.relative_to(image_dir)
    return rel.with_suffix("").as_posix()


def build_jsonl(
    image_dir: str,
    output_jsonl: str,
    lang: str,
    domain: str = "debug",
    task: str = "",
    limit: int = 100,
    recursive: bool = True,
) -> int:
    root = Path(image_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"image_dir does not exist: {root}")

    images = collect_images(root, recursive=recursive)
    if limit is not None and limit >= 0:
        images = images[:limit]

    output_path = Path(output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for image_path in images:
            record = {
                "sample_id": sample_id_for(image_path, root),
                "image": str(image_path.resolve()),
                "lang": lang,
                "domain": domain,
                "task": task,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(images)


def main() -> None:
    args = parse_args()
    count = build_jsonl(
        image_dir=args.image_dir,
        output_jsonl=args.output_jsonl,
        lang=args.lang,
        domain=args.domain,
        task=args.task,
        limit=args.limit,
        recursive=args.recursive,
    )
    print(f"Wrote {count} samples to {Path(args.output_jsonl).expanduser().resolve()}")


if __name__ == "__main__":
    main()
