import torch
import torch.nn.functional as F


def sample_random_pairs(mask, n_pairs):
    """
    Sample random pairs of valid pixel indices with i != j guaranteed.

    Args:
        mask: (B, 1, H, W) binary mask of valid pixels
        n_pairs: number of pairs to sample per image

    Returns:
        idx_i, idx_j: (B, n_pairs) flat indices into H*W
    """
    B, _, H, W = mask.shape
    mask_flat = mask.view(B, H * W)

    idx_i = torch.zeros(B, n_pairs, dtype=torch.long, device=mask.device)
    idx_j = torch.zeros(B, n_pairs, dtype=torch.long, device=mask.device)

    for b in range(B):
        valid = mask_flat[b].nonzero(as_tuple=False).squeeze(1)
        n_valid = valid.shape[0]
        assert n_valid >= 2, f"Need at least 2 valid pixels, got {n_valid}"

        rand_idx_i = torch.randint(0, n_valid, (n_pairs,), device=mask.device)
        rand_idx_j = torch.randint(0, n_valid - 1, (n_pairs,), device=mask.device)
        rand_idx_j[rand_idx_j >= rand_idx_i] += 1

        idx_i[b] = valid[rand_idx_i]
        idx_j[b] = valid[rand_idx_j]

    return idx_i, idx_j


def ordinal_loss(pred, gt, mask, n_pairs=10000, delta=0.01):
    """
    Pairwise ordinal/ranking loss.

    For pixel pairs at similar GT depth (|delta_gt| < delta): L2 on predicted difference.
    For pixel pairs at different GT depth: linear penalty only when predicted
    ordering disagrees with GT ordering.

    Args:
        pred: (B, 1, H, W) estimated disparity
        gt:   (B, 1, H, W) ground-truth disparity
        mask: (B, 1, H, W) valid pixel mask
        n_pairs: number of random pairs per image
        delta: threshold for same-depth classification

    Returns:
        scalar loss
    """
    B, _, H, W = pred.shape
    pred_flat = pred.view(B, H * W)
    gt_flat = gt.view(B, H * W)

    idx_i, idx_j = sample_random_pairs(mask, n_pairs)

    pred_i = torch.gather(pred_flat, 1, idx_i)
    pred_j = torch.gather(pred_flat, 1, idx_j)
    gt_i = torch.gather(gt_flat, 1, idx_i)
    gt_j = torch.gather(gt_flat, 1, idx_j)

    delta_pred = pred_i - pred_j
    delta_gt = gt_i - gt_j

    same_depth = delta_gt.abs() < delta

    l2_term = delta_pred ** 2
    rank_term = F.relu(-delta_pred * torch.sign(delta_gt))

    loss = torch.where(same_depth, l2_term, rank_term)

    return loss.mean()
