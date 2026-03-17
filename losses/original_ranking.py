import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def randomSampling(inputs, targets, masks, threshold, sample_num):

    # find A-B point pairs from predictions
    inputs_index = torch.masked_select(inputs, targets.gt(threshold))
    num_effect_pixels = len(inputs_index)

    # Guard: need at least 2 pixels to form pairs
    if num_effect_pixels < 2 or sample_num == 0:
        empty = torch.zeros(0, device=inputs.device, dtype=inputs.dtype)
        return empty, empty, empty, empty, empty, empty

    # Clamp sample_num so we don't index out of bounds
    sample_num = min(sample_num, num_effect_pixels // 2)
    shuffle_effect_pixels = torch.randperm(num_effect_pixels, device=inputs.device)
    inputs_A = inputs_index[shuffle_effect_pixels[0:sample_num*2:2]]
    inputs_B = inputs_index[shuffle_effect_pixels[1:sample_num*2:2]]
    # find corresponding pairs from GT
    target_index = torch.masked_select(targets, targets.gt(threshold))
    targets_A = target_index[shuffle_effect_pixels[0:sample_num*2:2]]
    targets_B = target_index[shuffle_effect_pixels[1:sample_num*2:2]]
    # only compute the losses of point pairs with valid GT
    consistent_masks_index = torch.masked_select(masks, targets.gt(threshold))
    consistent_masks_A = consistent_masks_index[shuffle_effect_pixels[0:sample_num*2:2]]
    consistent_masks_B = consistent_masks_index[shuffle_effect_pixels[1:sample_num*2:2]]

    # The amount of A and B should be the same!!
    if len(targets_A) > len(targets_B):
        targets_A = targets_A[:-1]
        inputs_A = inputs_A[:-1]
        consistent_masks_A = consistent_masks_A[:-1]

    return inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B
def ind2sub(idx, cols):
    r = idx // cols
    c = idx - r * cols
    return r, c

def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx

def edgeGuidedSampling(inputs, targets, edges_img, thetas_img, masks, h, w):

    # find edges
    edges_max = edges_img.max()
    edges_mask = edges_img.ge(edges_max*0.1)
    edges_loc = edges_mask.nonzero()

    inputs_edge = torch.masked_select(inputs, edges_mask)
    targets_edge = torch.masked_select(targets, edges_mask)
    thetas_edge = torch.masked_select(thetas_img, edges_mask)
    minlen = inputs_edge.size()[0]

    # Guard: if no edges found, return empty tensors
    if minlen == 0:
        empty = torch.zeros(0, device=inputs.device, dtype=inputs.dtype)
        return empty, empty, empty, empty, empty, empty, 0

    # find anchor points (i.e, edge points)
    sample_num = minlen
    index_anchors = torch.randint(0, minlen, (sample_num,), dtype=torch.long, device=inputs.device)
    anchors = torch.gather(inputs_edge, 0, index_anchors)
    theta_anchors = torch.gather(thetas_edge, 0, index_anchors)
    row_anchors, col_anchors = ind2sub(edges_loc[index_anchors].squeeze(1), w)
    ## compute the coordinates of 4-points,  distances are from [2, 30]
    distance_matrix = torch.randint(2, 21, (4,sample_num), device=inputs.device)
    pos_or_neg = torch.ones(4, sample_num, device=inputs.device)
    pos_or_neg[:2,:] = -pos_or_neg[:2,:]
    distance_matrix = distance_matrix.float() * pos_or_neg
    col = col_anchors.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix * torch.cos(theta_anchors).unsqueeze(0)).long()
    row = row_anchors.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix * torch.sin(theta_anchors).unsqueeze(0)).long()

    # constrain 0=<c<=w, 0<=r<=h
    # Note: index should minus 1
    col[col<0] = 0
    col[col>w-1] = w-1
    row[row<0] = 0
    row[row>h-1] = h-1

    # a-b, b-c, c-d
    a = sub2ind(row[0,:], col[0,:], w)
    b = sub2ind(row[1,:], col[1,:], w)
    c = sub2ind(row[2,:], col[2,:], w)
    d = sub2ind(row[3,:], col[3,:], w)
    A = torch.cat((a,b,c), 0)
    B = torch.cat((b,c,d), 0)

    inputs_A = torch.gather(inputs, 0, A.long())
    inputs_B = torch.gather(inputs, 0, B.long())
    targets_A = torch.gather(targets, 0, A.long())
    targets_B = torch.gather(targets, 0, B.long())
    masks_A = torch.gather(masks, 0, A.long())
    masks_B = torch.gather(masks, 0, B.long())

    return inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B, sample_num

class EdgeguidedRankingLoss(nn.Module):
    def __init__(self, point_pairs=10000, sigma=0.03, alpha=1.0, mask_value=-1e-8, max_disp=192.0):
        super(EdgeguidedRankingLoss, self).__init__()
        self.point_pairs = point_pairs # number of point pairs
        self.sigma = sigma # used for determining the ordinal relationship between a selected pair
        self.alpha = alpha # used for balancing the effect of = and (<,>)
        self.mask_value = mask_value
        self.max_disp = max_disp
        self.diagnostics = {}  # populated each forward pass for TensorBoard logging

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def getEdge(self, images):
        # Force float32 to prevent overflow in gradient_x.pow(2) under AMP float16
        n,c,h,w = images.size()
        a = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=images.device).view((1,1,3,3))
        b = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=images.device).view((1,1,3,3))
        if c == 3:
            gradient_x = F.conv2d(images[:,0,:,:].unsqueeze(1), a)
            gradient_y = F.conv2d(images[:,0,:,:].unsqueeze(1), b)
        else:
            gradient_x = F.conv2d(images, a)
            gradient_y = F.conv2d(images, b)
        edges = torch.sqrt(torch.pow(gradient_x,2)+ torch.pow(gradient_y,2) + 1e-6)
        edges = F.pad(edges, (1,1,1,1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1,1,1,1), "constant", 0)

        return edges, thetas

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, inputs, targets, images, masks=None):
        if masks == None:
            masks = targets > self.mask_value
        # find edges from RGB
        edges_img, thetas_img = self.getEdge(images)

        #=============================
        n,c,h,w = targets.size()
        # Stay in float32 — no .double() casting to avoid mixed-precision NaN
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

        # initialization
        loss = torch.zeros(1, dtype=torch.float32, device=inputs.device)


        for i in range(n):
            # Edge-Guided sampling
            inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B, sample_num = edgeGuidedSampling(inputs[i,:], targets[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w)
            # Random Sampling
            random_sample_num = sample_num
            #random_inputs_A, random_inputs_B, random_targets_A, random_targets_B, random_masks_A, random_masks_B = randomSampling(inputs[i,:], targets[i, :], masks[i, :], self.mask_value, random_sample_num)

            # Skip if both samplers returned empty
            #if inputs_A.numel() == 0 and random_inputs_A.numel() == 0:
            #    continue

            # Combine EGS + RS
            #inputs_A = torch.cat((inputs_A), 0)
            #inputs_B = torch.cat((inputs_B), 0)
            #targets_A = torch.cat((targets_A), 0)
            #targets_B = torch.cat((targets_B ), 0)
            #masks_A = torch.cat((masks_A), 0)
            #masks_B = torch.cat((masks_B), 0)
            targets_A = torch.clamp(targets_A, min=1e-8)
            targets_B = torch.clamp(targets_B, min=1e-8)

            log_A = torch.log(targets_A)
            log_B = torch.log(targets_B)
            log_ratio = log_A - log_B

            # Define thresholds in log space
            log_thresh_upper = torch.log(torch.tensor(1.0 + self.sigma, device=inputs.device))
            log_thresh_lower = torch.log(torch.tensor(1.0 / (1.0 + self.sigma), device=inputs.device))

            # 2. Update Label Logic
            mask_eq = (log_ratio < log_thresh_upper) & (log_ratio > log_thresh_lower)
                        #GT ordinal relationship
            #target_ratio = torch.div(targets_A+1e-6, targets_B+1e-6)
            #mask_eq = target_ratio.lt(1.0 + self.sigma) * target_ratio.gt(1.0/(1.0+self.sigma))
            labels = torch.zeros_like(log_ratio)
            labels[log_ratio >= log_thresh_upper] = 1
            labels[log_ratio <= log_thresh_lower] = -1

            # consider forward-backward consistency checking, i.e, only compute losses of point pairs with valid GT
            consistency_mask = masks_A * masks_B

            diff_ab = inputs_A - inputs_B

            # Equal pairs: predictions should be close
            equal_loss = F.smooth_l1_loss(inputs_A, inputs_B, reduction='none', beta=1.0) * mask_eq.float() * consistency_mask

            # Unequal pairs: enforce correct ordering (no normalization — softplus is numerically stable)
            logit = (-diff_ab) * labels
            unequal_loss = F.softplus(logit) * (~mask_eq).float() * consistency_mask

            # Average only over active (non-masked) pairs to avoid diluting gradients
            n_equal = (mask_eq.float() * consistency_mask).sum().clamp(min=1)
            n_unequal = ((~mask_eq).float() * consistency_mask).sum().clamp(min=1)

            if equal_loss.numel() > 0:
                loss = loss + self.alpha * (equal_loss.sum() / n_equal) + 1.0 * (unequal_loss.sum() / n_unequal)

        final_loss = loss[0] / max(n, 1)


        return final_loss
