import os

import pytest
import torch
from PIL import Image

from starVLA.model.modules.last_vit_probe import (
    LastViTProbe,
    compute_map_metrics,
    count_selected_indices_to_map,
    crop_top_last_region,
    gaussian_kernel_1d,
    mask_low_last_regions,
    mask_random_regions,
    mask_top_last_regions,
)


def _image(size=(64, 48), color=(120, 80, 40)):
    return Image.new("RGB", size, color)


def _count_maps(batch_size=1):
    maps = torch.zeros(batch_size, 14, 14)
    maps[:, 6, 7] = 10.0
    maps[:, 0, 0] = 1.0
    return maps


def test_gaussian_kernel_shape_and_finite_values():
    kernel = gaussian_kernel_1d(768, 768**0.5)
    assert kernel.shape == (768,)
    assert torch.isfinite(kernel).all()
    assert torch.isclose(kernel.max(), torch.tensor(1.0))


def test_selected_indices_count_map_sums_to_hidden_dim():
    indices = torch.arange(768).view(1, 1, 768).remainder(196).repeat(2, 1, 1)
    count_map = count_selected_indices_to_map(indices, grid_size=14, num_patches=196)
    assert count_map.shape == (2, 14, 14)
    assert torch.allclose(count_map.sum(dim=(1, 2)), torch.full((2,), 768.0))


def test_map_metrics_are_finite_and_monotonic():
    count_map = _count_maps(batch_size=2)
    metrics = compute_map_metrics(count_map)
    assert torch.isfinite(metrics["entropy"]).all()
    assert torch.isfinite(metrics["top5_mass"]).all()
    assert torch.isfinite(metrics["top10_mass"]).all()
    assert torch.all(metrics["top10_mass"] >= metrics["top5_mass"])


@pytest.mark.parametrize(
    "fn",
    [
        mask_top_last_regions,
        mask_low_last_regions,
        mask_random_regions,
        crop_top_last_region,
    ],
)
def test_interventions_preserve_pil_size(fn):
    img = _image()
    out = fn([img], _count_maps(), ratio=0.2) if fn is not crop_top_last_region else fn([img], _count_maps(), crop_scale=0.5)
    assert isinstance(out, list)
    assert out[0].size == img.size
    assert isinstance(out[0], Image.Image)


@pytest.mark.parametrize("fn", [mask_top_last_regions, crop_top_last_region])
def test_nested_structure_is_preserved_for_mask_and_crop(fn):
    img1 = _image(color=(255, 0, 0))
    img2 = _image(color=(0, 255, 0))
    nested = [[img1, img2]]
    maps = _count_maps(batch_size=2)
    out = fn(nested, maps, ratio=0.2) if fn is mask_top_last_regions else fn(nested, maps, crop_scale=0.5)
    assert isinstance(out, list)
    assert isinstance(out[0], list)
    assert len(out) == 1
    assert len(out[0]) == 2
    assert out[0][0].size == img1.size
    assert out[0][1].size == img2.size


@pytest.mark.skipif(not os.getenv("LAST_VIT_TEST_CHECKPOINT"), reason="LAST_VIT_TEST_CHECKPOINT is not set")
def test_optional_checkpoint_probe_forward():
    probe = LastViTProbe(checkpoint_path=os.environ["LAST_VIT_TEST_CHECKPOINT"], device="cpu")
    output = probe([_image(size=(224, 224))])
    assert output["count_map_raw"].shape == (1, 14, 14)
    assert output["count_map_raw"].sum().item() == 768
    assert torch.allclose(output["count_map"].sum(dim=(1, 2)), torch.ones(1))
    assert output["selected_indices"].shape == (1, 768)
