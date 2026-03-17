import torch
from torch import nn
import torch.nn.functional as F


def ind2sub(idx, cols):
    r = idx / cols
    c = idx - r * cols
    return r, c


def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx


def edgeGuidedTripletSampling(inputs, targets, edges_img, thetas_img, masks, h, w):
    """
    Sample triplets (anchor, positive, negative) from edge regions.

    From 4-point scheme [a, b | edge | c, d]:
      - Triplet 1: anchor=b, positive=a, negative=c
      - Triplet 2: anchor=c, positive=d, negative=b

    Returns predicted disparity and mask values at anchor/pos/neg positions.
    """
    edges_max = edges_img.max()
    edges_mask = edges_img.ge(edges_max * 0.1)
    edges_loc = edges_mask.nonzero()

    inputs_edge = torch.masked_select(inputs, edges_mask)
    thetas_edge = torch.masked_select(thetas_img, edges_mask)
    minlen = inputs_edge.size()[0]

    if minlen == 0:
        empty = torch.zeros(0, device=inputs.device, dtype=inputs.dtype)
        return empty, empty, empty, empty, empty, empty, 0

    sample_num = minlen
    index_anchors = torch.randint(0, minlen, (sample_num,), dtype=torch.long, device=inputs.device)
    theta_anchors = torch.gather(thetas_edge, 0, index_anchors)
    row_anchors, col_anchors = ind2sub(edges_loc[index_anchors].squeeze(1), w)

    # 4 points at distances [2, 30] along gradient normal
    distance_matrix = torch.randint(2, 31, (4, sample_num), device=inputs.device)
    pos_or_neg = torch.ones(4, sample_num, device=inputs.device)
    pos_or_neg[:2, :] = -pos_or_neg[:2, :]
    distance_matrix = distance_matrix.float() * pos_or_neg

    col = col_anchors.unsqueeze(0).expand(4, sample_num).long() + \
        torch.round(distance_matrix * torch.cos(theta_anchors).unsqueeze(0)).long()
    row = row_anchors.unsqueeze(0).expand(4, sample_num).long() + \
        torch.round(distance_matrix * torch.sin(theta_anchors).unsqueeze(0)).long()

    col = col.clamp(0, w - 1)
    row = row.clamp(0, h - 1)

    # Flat indices for 4 points: a(0), b(1), c(2), d(3)
    pt_a = sub2ind(row[0, :], col[0, :], w)
    pt_b = sub2ind(row[1, :], col[1, :], w)
    pt_c = sub2ind(row[2, :], col[2, :], w)
    pt_d = sub2ind(row[3, :], col[3, :], w)

    # Triplet 1: anchor=b, positive=a, negative=c
    # Triplet 2: anchor=c, positive=d, negative=b
    anchor_idx = torch.cat((pt_b, pt_c), 0).long()
    pos_idx = torch.cat((pt_a, pt_d), 0).long()
    neg_idx = torch.cat((pt_c, pt_b), 0).long()

    inputs_anchor = torch.gather(inputs, 0, anchor_idx)
    inputs_pos = torch.gather(inputs, 0, pos_idx)
    inputs_neg = torch.gather(inputs, 0, neg_idx)

    masks_anchor = torch.gather(masks, 0, anchor_idx)
    masks_pos = torch.gather(masks, 0, pos_idx)
    masks_neg = torch.gather(masks, 0, neg_idx)

    return inputs_anchor, inputs_pos, inputs_neg, masks_anchor, masks_pos, masks_neg, sample_num


def randomTripletSampling(inputs, targets, masks, threshold, sample_num):
    """
    Random triplet sampling: group 3 valid pixels by GT ordering.
    For each triplet of pixels sorted by GT disparity (low, mid, high):
      anchor=mid, positive=low or high (closer in GT), negative=the other.
    """
    valid = targets.gt(threshold)
    inputs_valid = torch.masked_select(inputs, valid)
    targets_valid = torch.masked_select(targets, valid)
    masks_valid = torch.masked_select(masks, valid)
    num_valid = len(inputs_valid)

    if num_valid < 3 or sample_num == 0:
        empty = torch.zeros(0, device=inputs.device, dtype=inputs.dtype)
        return empty, empty, empty, empty, empty, empty

    sample_num = min(sample_num, num_valid // 3)
    perm = torch.randperm(num_valid, device=inputs.device)[:sample_num * 3]

    # Group into triplets of 3
    idx_0 = perm[0::3]
    idx_1 = perm[1::3]
    idx_2 = perm[2::3]

    t0, t1, t2 = targets_valid[idx_0], targets_valid[idx_1], targets_valid[idx_2]

    # Sort each triplet by GT value: anchor = middle, pos = closer, neg = farther
    gt_stack = torch.stack([t0, t1, t2], dim=1)  # (S, 3)
    inp_stack = torch.stack([inputs_valid[idx_0], inputs_valid[idx_1], inputs_valid[idx_2]], dim=1)
    mask_stack = torch.stack([masks_valid[idx_0], masks_valid[idx_1], masks_valid[idx_2]], dim=1)

    sorted_idx = gt_stack.argsort(dim=1)  # sort by GT: low, mid, high
    gt_sorted = gt_stack.gather(1, sorted_idx)
    inp_sorted = inp_stack.gather(1, sorted_idx)
    mask_sorted = mask_stack.gather(1, sorted_idx)

    # anchor = middle element, positive = closest in GT, negative = farthest
    anchor_inp = inp_sorted[:, 1]
    anchor_mask = mask_sorted[:, 1]
    anchor_gt = gt_sorted[:, 1]

    # Which is closer to anchor in GT: low or high?
    dist_low = (anchor_gt - gt_sorted[:, 0]).abs()
    dist_high = (gt_sorted[:, 2] - anchor_gt).abs()

    low_closer = dist_low <= dist_high
    pos_inp = torch.where(low_closer, inp_sorted[:, 0], inp_sorted[:, 2])
    neg_inp = torch.where(low_closer, inp_sorted[:, 2], inp_sorted[:, 0])
    pos_mask = torch.where(low_closer, mask_sorted[:, 0], mask_sorted[:, 2])
    neg_mask = torch.where(low_closer, mask_sorted[:, 2], mask_sorted[:, 0])

    return anchor_inp, pos_inp, neg_inp, anchor_mask, pos_mask, neg_mask


class EdgeguidedTripletLoss(nn.Module):
    """
    Drop-in replacement for EdgeguidedRankingLoss using PyTorch TripletMarginLoss.

    Uses same edge-guided 4-point sampling but forms proper (anchor, positive, negative)
    triplets. The margin ensures gradients stop once ordering is satisfied by margin,
    preventing unbounded pushing of predictions.
    """

    def __init__(self, point_pairs=10000, sigma=0.03, alpha=1.0, mask_value=-1e-8, margin=2.0):
        super().__init__()
        self.point_pairs = point_pairs
        self.sigma = sigma
        self.alpha = alpha
        self.mask_value = mask_value
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=1, reduction='none')

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def getEdge(self, images):
        n, c, h, w = images.size()
        a = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=images.device).view((1, 1, 3, 3))
        b = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=images.device).view((1, 1, 3, 3))
        if c == 3:
            gradient_x = F.conv2d(images[:, 0, :, :].unsqueeze(1), a)
            gradient_y = F.conv2d(images[:, 0, :, :].unsqueeze(1), b)
        else:
            gradient_x = F.conv2d(images, a)
            gradient_y = F.conv2d(images, b)
        edges = torch.sqrt(gradient_x.pow(2) + gradient_y.pow(2) + 1e-6)
        edges = F.pad(edges, (1, 1, 1, 1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1, 1, 1, 1), "constant", 0)
        return edges, thetas

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, inputs, targets, images, masks=None):
        if masks is None:
            masks = targets > self.mask_value

        edges_img, thetas_img = self.getEdge(images)

        n, c, h, w = targets.size()
        if n != 1:
            inputs = inputs.view(n, -1).float()
            targets = targets.view(n, -1).float()
            masks = masks.view(n, -1).float()
            edges_img = edges_img.view(n, -1).float()
            thetas_img = thetas_img.view(n, -1).float()
        else:
            inputs = inputs.contiguous().view(1, -1).float()
            targets = targets.contiguous().view(1, -1).float()
            masks = masks.contiguous().view(1, -1).float()
            edges_img = edges_img.contiguous().view(1, -1).float()
            thetas_img = thetas_img.contiguous().view(1, -1).float()

        loss = torch.zeros(1, dtype=torch.float32, device=inputs.device)

        for i in range(n):
            # Edge-guided triplet sampling
            eg_anchor, eg_pos, eg_neg, eg_m_a, eg_m_p, eg_m_n, sample_num = edgeGuidedTripletSampling(
                inputs[i, :], targets[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w)

            # Random triplet sampling
            #rs_anchor, rs_pos, rs_neg, rs_m_a, rs_m_p, rs_m_n = randomTripletSampling(
            #    inputs[i, :], targets[i, :], masks[i, :], self.mask_value, sample_num)

            if eg_anchor.numel() == 0:
                continue

            # Combine edge-guided + random triplets
            all_anchor = torch.cat([x for x in [eg_anchor] if x.numel() > 0], 0)
            all_pos = torch.cat([x for x in [eg_pos] if x.numel() > 0], 0)
            all_neg = torch.cat([x for x in [eg_neg] if x.numel() > 0], 0)
            all_mask = torch.cat([x for x in [eg_m_a * eg_m_p * eg_m_n] if x.numel() > 0], 0)

            # TripletMarginLoss expects (N, D) — D=1 for scalar disparity
            triplet_losses = self.triplet_loss(
                all_anchor.unsqueeze(1),
                all_pos.unsqueeze(1),
                all_neg.unsqueeze(1)
            )

            # Apply consistency mask (all 3 points must be valid)
            triplet_losses = triplet_losses * all_mask
            num_valid = all_mask.sum().clamp(min=1.0)
            loss = loss + triplet_losses.sum() / num_valid

        return loss[0] / max(n, 1)
