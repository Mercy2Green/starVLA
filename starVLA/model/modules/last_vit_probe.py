"""Frozen LAST-ViT spatial probe and image-level interventions."""

from __future__ import annotations

import math
import os
import pickle
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models.vision_transformer import VisionTransformer

ImageBatch = Union[Image.Image, List[Image.Image], List[List[Image.Image]]]


def gaussian_kernel_1d(kernel_size: int, sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """LAST-ViT 使用的一维低通高斯核。"""
    values = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (values / sigma) ** 2)
    return kernel / torch.max(kernel)


def count_selected_indices_to_map(
    indices: torch.Tensor,
    grid_size: int = 14,
    num_patches: int = 196,
) -> torch.Tensor:
    """把 LAST top-k 选中的 patch 索引计数为 [B, grid, grid] 原始计数图。"""
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices = indices.squeeze(1)
    if indices.ndim != 2:
        raise ValueError(f"indices must have shape [B, C] or [B, 1, C], got {tuple(indices.shape)}")

    indices = indices.long()
    if indices.numel() and ((indices < 0).any() or (indices >= num_patches).any()):
        raise ValueError(f"indices values must be in [0, {num_patches})")

    batch_size = indices.shape[0]
    counts = torch.zeros(batch_size, num_patches, device=indices.device, dtype=torch.float32)
    counts.scatter_add_(1, indices, torch.ones_like(indices, dtype=torch.float32))
    return counts.view(batch_size, grid_size, grid_size)


def compute_map_metrics(count_map: torch.Tensor, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """计算空间集中度指标，输入可为 [H,W] 或 [B,H,W]。"""
    if count_map.ndim == 2:
        count_map = count_map.unsqueeze(0)
    if count_map.ndim != 3:
        raise ValueError(f"count_map must have shape [H, W] or [B, H, W], got {tuple(count_map.shape)}")

    flat = count_map.float().flatten(1)
    prob = flat / flat.sum(dim=1, keepdim=True).clamp_min(eps)
    entropy = -(prob * (prob + eps).log()).sum(dim=1)
    sorted_prob = prob.sort(dim=1, descending=True).values
    return {
        "entropy": entropy,
        "top5_mass": sorted_prob[:, :5].sum(dim=1),
        "top10_mass": sorted_prob[:, :10].sum(dim=1),
    }


def normalize_image_batch_structure(batch_images: ImageBatch) -> Tuple[List[Image.Image], Callable[[List[Image.Image]], ImageBatch]]:
    """展平 PIL / list[PIL] / list[list[PIL]]，并返回结构恢复函数。"""
    if isinstance(batch_images, Image.Image):
        return [batch_images], lambda flat: flat[0]

    if not isinstance(batch_images, list):
        raise TypeError(f"Unsupported image batch type: {type(batch_images)}")

    if all(isinstance(img, Image.Image) for img in batch_images):
        length = len(batch_images)

        def restore_list(flat: List[Image.Image]) -> List[Image.Image]:
            return list(flat[:length])

        return list(batch_images), restore_list

    if all(isinstance(sample, list) for sample in batch_images):
        lengths = [len(sample) for sample in batch_images]
        flat: List[Image.Image] = []
        for sample in batch_images:
            for img in sample:
                if not isinstance(img, Image.Image):
                    raise TypeError("Nested image batches must contain PIL.Image objects")
                flat.append(img)

        def restore_nested(flat_images: List[Image.Image]) -> List[List[Image.Image]]:
            restored = []
            offset = 0
            for length in lengths:
                restored.append(list(flat_images[offset : offset + length]))
                offset += length
            return restored

        return flat, restore_nested

    raise TypeError("batch_images must be PIL.Image, list[PIL.Image], or list[list[PIL.Image]]")


class LastViTProbe(nn.Module):
    """冻结的 LAST-ViT probe，只输出空间选择计数，不替换 Qwen-VL 视觉编码器。"""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        image_size: int = 224,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
    ) -> None:
        super().__init__()
        requested_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")

        self.device = requested_device
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.hidden_dim = hidden_dim
        self.missing_keys: List[str] = []
        self.unexpected_keys: List[str] = []

        self.model = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
        )
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def train(self, mode: bool = True) -> "LastViTProbe":
        super().train(False)
        self.model.eval()
        return self

    def load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"LAST-ViT checkpoint not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except pickle.UnpicklingError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        cleaned = {}
        for key, value in state_dict.items():
            key = key[6:] if key.startswith("model.") else key
            cleaned[key] = value
        incompatible = self.model.load_state_dict(cleaned, strict=False)
        self.missing_keys = list(incompatible.missing_keys)
        self.unexpected_keys = list(incompatible.unexpected_keys)

    @property
    def current_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _images_to_tensor(self, images: Sequence[Image.Image]) -> torch.Tensor:
        tensors = [self.preprocess(img.convert("RGB")) for img in images]
        return torch.stack(tensors, dim=0).to(self.current_device)

    @torch.inference_mode()
    def forward(self, images: ImageBatch) -> Dict[str, torch.Tensor]:
        flat_images, _ = normalize_image_batch_structure(images)
        if not flat_images:
            raise ValueError("LastViTProbe received an empty image batch")

        image_tensor = self._images_to_tensor(flat_images)
        device_type = self.current_device.type
        with torch.autocast(device_type=device_type, enabled=False):
            x = image_tensor.float()
            x = self.model._process_input(x)
            batch_size = x.shape[0]
            class_token = self.model.class_token.expand(batch_size, -1, -1)
            x = torch.cat([class_token, x], dim=1)
            x = self.model.encoder(x)

            patch_tokens = x[:, 1:].float()
            fft_tokens = torch.fft.fft(patch_tokens, dim=-1)
            kernel = gaussian_kernel_1d(self.hidden_dim, math.sqrt(self.hidden_dim), x.device, patch_tokens.dtype)
            kernel = kernel.view(1, 1, -1)
            low_pass = torch.fft.fftshift(fft_tokens, dim=-1) * kernel
            low_pass = torch.fft.ifftshift(low_pass, dim=-1)
            smoothed = torch.fft.ifft(low_pass, dim=-1).real
            diff = patch_tokens / torch.abs(smoothed - patch_tokens).clamp_min(1e-6)
            _, indices = torch.topk(diff, k=1, dim=1, largest=True)

            count_map_raw = count_selected_indices_to_map(indices, self.grid_size, self.num_patches)
            count_sum = count_map_raw.flatten(1).sum(dim=1).view(-1, 1, 1).clamp_min(1e-6)
            count_map = count_map_raw / count_sum
            count_map_224 = F.interpolate(
                count_map.unsqueeze(1),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            metrics = compute_map_metrics(count_map_raw)

        return {
            "count_map_raw": count_map_raw.detach().cpu(),
            "count_map": count_map.detach().cpu(),
            "count_map_224": count_map_224.detach().cpu(),
            "entropy": metrics["entropy"].detach().cpu(),
            "top5_mass": metrics["top5_mass"].detach().cpu(),
            "top10_mass": metrics["top10_mass"].detach().cpu(),
            "selected_indices": indices.squeeze(1).detach().cpu(),
        }


def _as_batched_maps(count_maps: torch.Tensor, expected: int) -> torch.Tensor:
    if count_maps.ndim == 2:
        count_maps = count_maps.unsqueeze(0)
    if count_maps.ndim != 3:
        raise ValueError(f"count_maps must have shape [H,W] or [B,H,W], got {tuple(count_maps.shape)}")
    if count_maps.shape[0] == 1 and expected > 1:
        count_maps = count_maps.expand(expected, -1, -1)
    if count_maps.shape[0] != expected:
        raise ValueError(f"count_maps batch {count_maps.shape[0]} does not match image count {expected}")
    return count_maps.float()


def _patch_boxes(width: int, height: int, grid_h: int, grid_w: int, patch_indices: Sequence[int]) -> List[Tuple[int, int, int, int]]:
    boxes = []
    for idx in patch_indices:
        row = int(idx) // grid_w
        col = int(idx) % grid_w
        x0 = round(col * width / grid_w)
        x1 = round((col + 1) * width / grid_w)
        y0 = round(row * height / grid_h)
        y1 = round((row + 1) * height / grid_h)
        boxes.append((x0, y0, x1, y1))
    return boxes


def _mask_images_by_rank(
    batch_images: ImageBatch,
    count_maps: torch.Tensor,
    ratio: float,
    largest: bool,
    random: bool = False,
    fill: Tuple[int, int, int] = (0, 0, 0),
) -> ImageBatch:
    flat_images, restore = normalize_image_batch_structure(batch_images)
    maps = _as_batched_maps(count_maps, len(flat_images))
    ratio = min(max(float(ratio), 0.0), 1.0)
    out = []
    for img, count_map in zip(flat_images, maps):
        grid_h, grid_w = count_map.shape
        total = grid_h * grid_w
        k = max(1, int(round(total * ratio))) if ratio > 0 else 0
        patched = img.copy()
        if k:
            if random:
                order = torch.randperm(total)[:k]
            else:
                order = torch.topk(count_map.flatten(), k=k, largest=largest).indices
            draw = ImageDraw.Draw(patched)
            draw_fill = fill
            if patched.mode not in ("RGB", "RGBA"):
                draw_fill = 0
            for box in _patch_boxes(patched.width, patched.height, grid_h, grid_w, order.tolist()):
                draw.rectangle(box, fill=draw_fill)
        out.append(patched)
    return restore(out)


def mask_top_last_regions(batch_images: ImageBatch, count_maps: torch.Tensor, ratio: float = 0.25) -> ImageBatch:
    return _mask_images_by_rank(batch_images, count_maps, ratio=ratio, largest=True)


def mask_low_last_regions(batch_images: ImageBatch, count_maps: torch.Tensor, ratio: float = 0.25) -> ImageBatch:
    return _mask_images_by_rank(batch_images, count_maps, ratio=ratio, largest=False)


def mask_random_regions(batch_images: ImageBatch, count_maps: torch.Tensor, ratio: float = 0.25) -> ImageBatch:
    return _mask_images_by_rank(batch_images, count_maps, ratio=ratio, largest=True, random=True)


def _weighted_center(count_map: torch.Tensor) -> Tuple[float, float]:
    grid_h, grid_w = count_map.shape
    weights = count_map.float().clamp_min(0)
    if weights.sum() <= 0:
        return (grid_w - 1) / 2.0, (grid_h - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32),
        indexing="ij",
    )
    weights = weights.cpu()
    cx = float((xx * weights).sum() / weights.sum())
    cy = float((yy * weights).sum() / weights.sum())
    return cx, cy


def _crop_around_map(img: Image.Image, count_map: torch.Tensor, crop_scale: float) -> Image.Image:
    crop_scale = min(max(float(crop_scale), 1e-3), 1.0)
    width, height = img.size
    crop_w = max(1, int(round(width * crop_scale)))
    crop_h = max(1, int(round(height * crop_scale)))
    grid_h, grid_w = count_map.shape
    cx_patch, cy_patch = _weighted_center(count_map)
    cx = (cx_patch + 0.5) * width / grid_w
    cy = (cy_patch + 0.5) * height / grid_h
    left = int(round(cx - crop_w / 2))
    top = int(round(cy - crop_h / 2))
    left = min(max(left, 0), max(width - crop_w, 0))
    top = min(max(top, 0), max(height - crop_h, 0))
    crop = img.crop((left, top, left + crop_w, top + crop_h))
    return crop.resize(img.size, Image.BICUBIC)


def crop_top_last_region(batch_images: ImageBatch, count_maps: torch.Tensor, crop_scale: float = 0.5) -> ImageBatch:
    flat_images, restore = normalize_image_batch_structure(batch_images)
    maps = _as_batched_maps(count_maps, len(flat_images))
    return restore([_crop_around_map(img, count_map, crop_scale) for img, count_map in zip(flat_images, maps)])


def multicrop_top_last_regions(
    batch_images: ImageBatch,
    count_maps: torch.Tensor,
    crop_scale: float = 0.5,
    crop_count: int = 2,
) -> ImageBatch:
    flat_images, restore = normalize_image_batch_structure(batch_images)
    maps = _as_batched_maps(count_maps, len(flat_images))
    out: List[Image.Image] = []
    for img, count_map in zip(flat_images, maps):
        out.append(img)
        top_indices = torch.topk(count_map.flatten(), k=max(0, int(crop_count)), largest=True).indices.tolist()
        grid_h, grid_w = count_map.shape
        for idx in top_indices:
            single_patch_map = torch.zeros_like(count_map)
            single_patch_map[idx // grid_w, idx % grid_w] = 1.0
            out.append(_crop_around_map(img, single_patch_map, crop_scale))

    # multicrop 会有意改变 view 数，单图/list 输入直接返回扩展后的 view 列表。
    if isinstance(batch_images, Image.Image) or (isinstance(batch_images, list) and all(isinstance(x, Image.Image) for x in batch_images)):
        return out

    if isinstance(batch_images, list) and all(isinstance(sample, list) for sample in batch_images):
        nested: List[List[Image.Image]] = []
        offset = 0
        for sample in batch_images:
            sample_out = []
            for _ in sample:
                sample_out.extend(out[offset : offset + 1 + max(0, int(crop_count))])
                offset += 1 + max(0, int(crop_count))
            nested.append(sample_out)
        return nested

    return restore(out)
