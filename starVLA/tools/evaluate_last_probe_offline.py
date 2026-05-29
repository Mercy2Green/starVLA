"""Offline causal perturbation runner for frozen LAST-ViT interventions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--last_checkpoint", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base_vlm", default=None)
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--modes", default="none,mask_top,mask_low,mask_random,crop_top")
    parser.add_argument("--framework_name", default="QwenOFT")
    parser.add_argument("--dry_run_fake_policy", action="store_true")
    parser.add_argument("--allow_untrained_action_head", action="store_true")
    parser.add_argument("--save_first_n_visualizations", type=int, default=10)
    return parser.parse_args()


def read_jsonl(path: str, max_samples: int) -> List[Dict[str, Any]]:
    records = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_samples is not None and max_samples > 0 and len(records) >= max_samples:
                break
    return records


def normalize_modes(modes: str) -> List[str]:
    parsed = [mode.strip() for mode in modes.split(",") if mode.strip()]
    if "none" not in parsed:
        parsed.insert(0, "none")
    return parsed


def load_images(image_field: Any) -> List[Image.Image]:
    paths = image_field if isinstance(image_field, list) else [image_field]
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB").copy())
    return images


def image_paths(image_field: Any) -> Any:
    if isinstance(image_field, list):
        return [str(Path(path).expanduser().resolve()) for path in image_field]
    return str(Path(image_field).expanduser().resolve())


def ensure_last_probe_cfg(cfg, args: argparse.Namespace, mode: str) -> None:
    if not hasattr(cfg, "framework"):
        cfg.framework = {}
    if not hasattr(cfg.framework, "last_probe") or cfg.framework.last_probe is None:
        cfg.framework.last_probe = {}
    cfg.framework.last_probe.enabled = True
    cfg.framework.last_probe.checkpoint_path = args.last_checkpoint
    cfg.framework.last_probe.checkpoint = args.last_checkpoint
    cfg.framework.last_probe.device = args.device
    cfg.framework.last_probe.intervention = mode
    cfg.framework.last_probe.log_jsonl = str(Path(args.output_dir).expanduser().resolve() / "last_probe_metrics.jsonl")


def apply_qwenvl_overrides(cfg, args: argparse.Namespace) -> None:
    if args.base_vlm is None and args.attn_implementation is None:
        return
    if not hasattr(cfg, "framework"):
        cfg.framework = {}
    if not hasattr(cfg.framework, "qwenvl") or cfg.framework.qwenvl is None:
        cfg.framework.qwenvl = {}
    if args.base_vlm is not None:
        base_vlm = Path(args.base_vlm).expanduser()
        cfg.framework.qwenvl.base_vlm = str(base_vlm.resolve()) if base_vlm.exists() else args.base_vlm
    if args.attn_implementation is not None:
        cfg.framework.qwenvl.attn_implementation = args.attn_implementation


def set_mode_on_model(model, cfg, args: argparse.Namespace, mode: str) -> None:
    ensure_last_probe_cfg(cfg, args, mode)
    if hasattr(model, "config"):
        ensure_last_probe_cfg(model.config, args, mode)


def load_checkpoint_conservative(model, checkpoint_path: str):
    path = Path(checkpoint_path).expanduser()
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        except pickle.UnpicklingError:
            state = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value

    incompatible = model.load_state_dict(cleaned, strict=False)
    print(f"Loaded checkpoint: {path}")
    print(f"Missing keys: {len(incompatible.missing_keys)}")
    print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")
    if incompatible.missing_keys:
        print("First missing keys:", incompatible.missing_keys[:20])
    if incompatible.unexpected_keys:
        print("First unexpected keys:", incompatible.unexpected_keys[:20])


def build_model(cfg, args: argparse.Namespace):
    cfg.framework.name = args.framework_name
    apply_qwenvl_overrides(cfg, args)
    ensure_last_probe_cfg(cfg, args, mode="none")
    if not args.checkpoint:
        print(
            "WARNING: no StarVLA/QwenOFT action checkpoint provided. Action head may be untrained; "
            "this run is implementation smoke only, not scientific evidence."
        )
    if args.framework_name == "QwenOFT":
        from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT

        model = Qwenvl_OFT(cfg)
    else:
        from starVLA.model.framework.base_framework import build_framework

        model = build_framework(cfg)
    if args.checkpoint:
        load_checkpoint_conservative(model, args.checkpoint)
    print_action_token_diagnostics(model)
    target_device = torch.device(args.device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        print(f"WARNING: requested device {args.device}, but CUDA is unavailable; keeping model on CPU.")
    else:
        model.to(target_device)
    model.eval()
    return model


def print_action_token_diagnostics(model) -> None:
    if not all(hasattr(model, attr) for attr in ("action_token", "action_token_id", "chunk_len", "qwen_vl_interface")):
        return
    tokenizer = model.qwen_vl_interface.processor.tokenizer
    tokenized = tokenizer(model.action_token, add_special_tokens=False)["input_ids"]
    repeated = tokenizer(model.action_token * model.chunk_len, add_special_tokens=False)["input_ids"]
    count = sum(1 for token_id in repeated if token_id == model.action_token_id)
    print(f"action token: {model.action_token}")
    print(f"action token id: {model.action_token_id}")
    print(f'tokenizer("{model.action_token}", add_special_tokens=False): {tokenized}')
    print(f"chunk_len: {model.chunk_len}")
    print(f"action token id count in repeated token string: {count} / {model.chunk_len}")
    if count < model.chunk_len:
        print("WARNING: action token may not be tokenized as expected; QwenOFT may fail to gather action token embeddings.")


def fake_policy_actions(images: List[Image.Image], lang: str, mode: str) -> np.ndarray:
    pixels = []
    for img in images:
        pixels.append(np.asarray(img, dtype=np.float32).mean() / 255.0)
    pixel_mean = float(np.mean(pixels)) if pixels else 0.0
    digest = hashlib.sha256((mode + "\n" + lang).encode("utf-8")).digest()
    mode_values = np.frombuffer(digest[:28], dtype=np.uint8).astype(np.float32).reshape(4, 7) / 255.0
    base = np.linspace(0.0, 0.7, 28, dtype=np.float32).reshape(4, 7)
    return (base + pixel_mean + 0.1 * mode_values)[None, ...]


def fake_probe_output(images: List[Image.Image], lang: str) -> Dict[str, torch.Tensor]:
    count_map = torch.zeros(1, 14, 14)
    digest = hashlib.sha256((str(len(images)) + "\n" + lang).encode("utf-8")).digest()
    center = digest[0] % 196
    count_map.view(1, -1)[0, center] = 10.0
    count_map.view(1, -1)[0, digest[1] % 196] += 3.0
    prob = count_map.flatten(1) / count_map.sum().clamp_min(1e-6)
    entropy = -(prob * (prob + 1e-6).log()).sum(dim=1)
    sorted_prob = prob.sort(dim=1, descending=True).values
    return {
        "count_map_raw": count_map,
        "entropy": entropy,
        "top5_mass": sorted_prob[:, :5].sum(dim=1),
        "top10_mass": sorted_prob[:, :10].sum(dim=1),
    }


def predict_actions(model, cfg, args: argparse.Namespace, sample: Dict[str, Any], images: List[Image.Image], mode: str):
    if args.dry_run_fake_policy:
        return fake_policy_actions(images, sample.get("lang", ""), mode), fake_probe_output(images, sample.get("lang", ""))

    set_mode_on_model(model, cfg, args, mode)
    example = {"image": images, "lang": sample["lang"]}
    if "state" in sample:
        example["state"] = sample["state"]
    with torch.inference_mode():
        output = model.predict_action(examples=[example])
    actions = np.asarray(output["normalized_actions"], dtype=np.float32)
    probe_out = getattr(model, "_last_probe_last_output", None)
    return actions, probe_out


def scalar_metric(probe_out: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if probe_out is None or key not in probe_out:
        return None
    value = probe_out[key]
    if torch.is_tensor(value):
        return float(value.flatten()[0].item())
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0])


def action_delta(actions: np.ndarray, baseline: np.ndarray) -> Dict[str, float]:
    diff = actions - baseline
    abs_diff = np.abs(diff)
    return {
        "action_delta_l1": float(abs_diff.mean()),
        "action_delta_l2": float(np.sqrt(np.square(diff).mean())),
        "action_delta_max": float(abs_diff.max()),
    }


def write_summary(records: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "mode",
        "num_samples",
        "mean_action_delta_l1",
        "std_action_delta_l1",
        "mean_action_delta_l2",
        "std_action_delta_l2",
        "mean_entropy",
        "mean_top5_mass",
        "mean_top10_mass",
    ]
    modes = sorted({record["mode"] for record in records})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for mode in modes:
            subset = [record for record in records if record["mode"] == mode]
            l1 = [record["action_delta_l1"] for record in subset]
            l2 = [record["action_delta_l2"] for record in subset]
            entropy = [record["entropy"] for record in subset if record["entropy"] is not None]
            top5 = [record["top5_mass"] for record in subset if record["top5_mass"] is not None]
            top10 = [record["top10_mass"] for record in subset if record["top10_mass"] is not None]
            writer.writerow(
                {
                    "mode": mode,
                    "num_samples": len(subset),
                    "mean_action_delta_l1": statistics.fmean(l1) if l1 else 0.0,
                    "std_action_delta_l1": statistics.pstdev(l1) if len(l1) > 1 else 0.0,
                    "mean_action_delta_l2": statistics.fmean(l2) if l2 else 0.0,
                    "std_action_delta_l2": statistics.pstdev(l2) if len(l2) > 1 else 0.0,
                    "mean_entropy": statistics.fmean(entropy) if entropy else "",
                    "mean_top5_mass": statistics.fmean(top5) if top5 else "",
                    "mean_top10_mass": statistics.fmean(top10) if top10 else "",
                }
            )


def save_visualization(sample: Dict[str, Any], images: List[Image.Image], probe_out: Dict[str, Any], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        from starVLA.model.modules.last_vit_probe import crop_top_last_region, mask_low_last_regions, mask_top_last_regions

        count_map = probe_out["count_map_raw"]
        img = images[0]
        mask_top = mask_top_last_regions([img], count_map[:1], ratio=0.25)[0]
        mask_low = mask_low_last_regions([img], count_map[:1], ratio=0.25)[0]
        crop_top = crop_top_last_region([img], count_map[:1], crop_scale=0.5)[0]

        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        panels = [
            ("original", img),
            ("last count", count_map[0].detach().cpu().numpy()),
            ("mask_top", mask_top),
            ("mask_low", mask_low),
            ("crop_top", crop_top),
        ]
        for ax, (title, data) in zip(axes, panels):
            ax.set_title(title)
            ax.axis("off")
            if title == "last count":
                ax.imshow(data, cmap="magma")
            else:
                ax.imshow(data)
        fig.suptitle(str(sample.get("sample_id", "")))
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Visualization skipped for {sample.get('sample_id', '')}: {exc}")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = normalize_modes(args.modes)
    cfg = OmegaConf.load(args.config_yaml)

    model = None
    if args.dry_run_fake_policy:
        print("WARNING: --dry_run_fake_policy uses deterministic fake actions and is not scientifically meaningful.")
    else:
        model = build_model(cfg, args)

    samples = read_jsonl(args.input_jsonl, max_samples=args.max_samples)
    per_sample_path = output_dir / "per_sample.jsonl"
    summary_path = output_dir / "summary.csv"
    all_records: List[Dict[str, Any]] = []

    with per_sample_path.open("w", encoding="utf-8") as out_f:
        for sample_index, sample in enumerate(samples):
            images = load_images(sample["image"])
            mode_actions: Dict[str, np.ndarray] = {}
            mode_probe: Dict[str, Any] = {}
            for mode in modes:
                actions, probe_out = predict_actions(model, cfg, args, sample, images, mode)
                mode_actions[mode] = actions
                mode_probe[mode] = probe_out

            baseline = mode_actions["none"]
            for mode in modes:
                deltas = action_delta(mode_actions[mode], baseline)
                probe_out = mode_probe.get(mode) or mode_probe.get("none")
                record = {
                    "sample_id": sample.get("sample_id", f"sample_{sample_index:06d}"),
                    "task": sample.get("task", ""),
                    "domain": sample.get("domain", ""),
                    "mode": mode,
                    **deltas,
                    "entropy": scalar_metric(probe_out, "entropy"),
                    "top5_mass": scalar_metric(probe_out, "top5_mass"),
                    "top10_mass": scalar_metric(probe_out, "top10_mass"),
                    "action_shape": list(mode_actions[mode].shape),
                    "image": image_paths(sample["image"]),
                    "lang": sample.get("lang", ""),
                }
                if "success" in sample:
                    record["success"] = sample["success"]
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                all_records.append(record)

            if args.save_first_n_visualizations > 0 and sample_index < args.save_first_n_visualizations:
                probe_out = mode_probe.get("none")
                if probe_out is not None:
                    save_visualization(
                        sample,
                        images,
                        probe_out,
                        output_dir / "visualizations" / f"{record['sample_id'].replace('/', '_')}.png",
                    )

    write_summary(all_records, summary_path)
    print(f"Wrote per-sample records to {per_sample_path}")
    print(f"Wrote summary to {summary_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
