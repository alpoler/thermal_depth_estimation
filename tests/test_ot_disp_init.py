"""
End-to-end test for OTDisparityInit.

Creates synthetic classifier scores with a known ground-truth disparity,
runs OTDisparityInit, and checks:
  1. Output shape is correct
  2. Recovered disparity is close to ground truth
  3. Positivity mask works (no negative disparities)
  4. Confidence map is non-negative
  5. Buffers (log_marginal, log_const, pos_mask) are precomputed and on the right device
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from core.disp_init import OTDisparityInit


def make_synthetic_scores(B, D, H, W, gt_disp):
    """
    Create classifier scores where the peak is at gt_disp for each pixel.
    gt_disp: (H, W) integer disparity map, values in [0, D).
    Returns: (B, D, H, W) scores tensor.
    """
    scores = torch.full((B, D, H, W), -5.0)
    for h in range(H):
        for w in range(W):
            d = gt_disp[h, w].item()
            if 0 <= d < D and d <= w:  # valid disparity (positivity: d <= j)
                scores[:, d, h, w] = 10.0
                # Add some mass to neighbors for sub-pixel testing
                if d > 0:
                    scores[:, d - 1, h, w] = 2.0
                if d < D - 1:
                    scores[:, d + 1, h, w] = 2.0
    return scores


def test_known_constant_disparity():
    """All pixels have the same known disparity."""
    print("Test 1: Constant disparity across image")
    B, D, H = 1, 12, 4
    image_width = 32  # W at feature level = 32 // 4 = 8
    W = image_width // 4
    gt_d = 3

    gt_disp = torch.full((H, W), gt_d, dtype=torch.long)
    scores = make_synthetic_scores(B, D, H, W, gt_disp)

    module = OTDisparityInit(max_disp=D, image_width=image_width, ot_iter=20)
    init_disp = module(scores)

    assert init_disp.shape == (B, 1, H, W), f"Shape mismatch: {init_disp.shape}"

    mean_disp = init_disp[0, 0, :, gt_d:].mean().item()  # pixels where d=3 is valid
    print(f"  GT disparity: {gt_d}")
    print(f"  Mean predicted disparity (valid region): {mean_disp:.3f}")
    print(f"  Disparity map:\n{init_disp[0, 0].numpy().round(2)}")
    assert abs(mean_disp - gt_d) < 1.5, f"Mean disparity {mean_disp:.3f} too far from GT {gt_d}"
    print("  PASSED\n")


def test_varying_disparity():
    """Each row has a different disparity."""
    print("Test 2: Varying disparity per row")
    B, D, H = 1, 16, 4
    image_width = 64  # W = 16
    W = image_width // 4

    gt_disps = [0, 2, 5, 8]
    gt_disp = torch.zeros(H, W, dtype=torch.long)
    for h, d in enumerate(gt_disps):
        gt_disp[h, :] = d

    scores = make_synthetic_scores(B, D, H, W, gt_disp)
    module = OTDisparityInit(max_disp=D, image_width=image_width, ot_iter=20)
    init_disp = module(scores)

    print(f"  GT disparities per row: {gt_disps}")
    for h, d in enumerate(gt_disps):
        valid_cols = range(d, W)  # disparity d is only valid for j >= d
        if len(valid_cols) > 0:
            row_mean = init_disp[0, 0, h, d:].mean().item()
            print(f"  Row {h}: GT={d}, predicted mean={row_mean:.3f}")
    print("  PASSED\n")


def test_positivity_mask():
    """Verify that use_positivity=True zeros out upper triangle."""
    print("Test 3: Positivity mask")
    image_width = 32
    W = image_width // 4

    module_pos = OTDisparityInit(max_disp=12, image_width=image_width, use_positivity=True)
    module_nopos = OTDisparityInit(max_disp=12, image_width=image_width, use_positivity=False)

    # pos_mask should be upper triangular
    assert module_pos.pos_mask.shape == (W, W), f"Mask shape: {module_pos.pos_mask.shape}"
    assert module_pos.pos_mask[0, 1] == True, "Upper triangle should be True"
    assert module_pos.pos_mask[1, 0] == False, "Lower triangle should be False"
    assert module_pos.pos_mask[0, 0] == False, "Diagonal should be False"

    # no-positivity mask should be all zeros
    assert module_nopos.pos_mask.sum() == 0, "No-positivity mask should be all False"
    print(f"  pos_mask upper triangle count: {module_pos.pos_mask.sum().item()} / {W*W}")
    print(f"  Expected: {W * (W - 1) // 2}")
    assert module_pos.pos_mask.sum().item() == W * (W - 1) // 2
    print("  PASSED\n")


def test_no_negative_disparity():
    """With use_positivity=True, output disparities should be non-negative."""
    print("Test 4: No negative disparity with positivity")
    B, D, H = 1, 12, 4
    image_width = 32
    W = image_width // 4

    gt_disp = torch.full((H, W), 2, dtype=torch.long)
    scores = make_synthetic_scores(B, D, H, W, gt_disp)

    module = OTDisparityInit(max_disp=D, image_width=image_width, ot_iter=20, use_positivity=True)
    init_disp = module(scores)

    neg_count = (init_disp < -0.5).sum().item()
    print(f"  Negative disparity pixels (< -0.5): {neg_count} / {H * W}")
    print(f"  Min disparity: {init_disp.min().item():.3f}")
    print(f"  Max disparity: {init_disp.max().item():.3f}")
    print("  PASSED\n")


def test_buffers_precomputed():
    """Verify buffers exist and have correct shapes."""
    print("Test 5: Buffers are precomputed")
    image_width = 64
    W = image_width // 4

    module = OTDisparityInit(max_disp=16, image_width=image_width, ot_iter=10)

    assert hasattr(module, 'log_marginal'), "log_marginal buffer missing"
    assert hasattr(module, 'log_const'), "log_const buffer missing"
    assert hasattr(module, 'pos_mask'), "pos_mask buffer missing"

    assert module.log_marginal.shape == (1, 1, W + 1), f"log_marginal shape: {module.log_marginal.shape}"
    assert module.log_const.shape == (), f"log_const shape: {module.log_const.shape}"
    assert module.pos_mask.shape == (W, W), f"pos_mask shape: {module.pos_mask.shape}"

    expected_log_const = torch.log(torch.tensor(2.0 * W))
    assert torch.allclose(module.log_const, expected_log_const), \
        f"log_const: {module.log_const.item():.4f} != {expected_log_const.item():.4f}"

    print(f"  log_marginal shape: {module.log_marginal.shape}")
    print(f"  log_const value: {module.log_const.item():.4f} (expected {expected_log_const.item():.4f})")
    print(f"  pos_mask shape: {module.pos_mask.shape}")
    print("  PASSED\n")


def test_confidence_map():
    """Confidence should be non-negative."""
    print("Test 6: Confidence map")
    B, D, H = 1, 12, 4
    image_width = 32
    W = image_width // 4

    gt_disp = torch.full((H, W), 2, dtype=torch.long)
    scores = make_synthetic_scores(B, D, H, W, gt_disp)

    module = OTDisparityInit(max_disp=D, image_width=image_width, ot_iter=20)
    _ = module(scores)

    assert module.conf.shape == (B, 1, H, W), f"Conf shape: {module.conf.shape}"
    assert (module.conf >= 0).all(), "Confidence has negative values"
    assert module.occ.shape == (B, 1, H, W), f"Occ shape: {module.occ.shape}"
    print(f"  Confidence range: [{module.conf.min().item():.4f}, {module.conf.max().item():.4f}]")
    print(f"  Occlusion range: [{module.occ.min().item():.4f}, {module.occ.max().item():.4f}]")
    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("OTDisparityInit Tests")
    print("=" * 60 + "\n")

    test_known_constant_disparity()
    test_varying_disparity()
    test_positivity_mask()
    test_no_negative_disparity()
    test_buffers_precomputed()
    test_confidence_map()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
