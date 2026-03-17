import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as kf
class EdgeGuidedRankingLoss(nn.Module):
    def __init__(self, point_pairs=10000, sigma=0.03, alpha=1.0, mask_value=-1e-8):
        super().__init__()
        self.point_pairs = point_pairs
        self.sigma = sigma
        self.alpha = alpha
        self.mask_value = mask_value
        
        # Sobel filters for edge detection
        self.register_buffer('sobel_x', torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3))

    def get_edge(self, images):
            # 1. Setup
            B, C, H, W = images.shape
            img_gray = images[:, 0:1, :, :] if C == 3 else images
            #img_gray = kf.median_blur(img_gray, kernel_size=(11, 11))
            #img_blurred = kf.gaussian_blur2d(img_gray, kernel_size=(3, 3), sigma=(2.0, 2.0))
            # Prepare output container
            batch_edges = []
            batch_thetas = []

            # 2. Iterate over the batch
            # We process each image individually to allow unique thresholds
            for i in range(B):
                # Extract single image (1, 1, H, W)
                current_img = img_gray[i:i+1]
                
                # --- Dynamic Threshold Calculation per Image ---
                # Calculate magnitude to find the quantile
                grads = kf.spatial_gradient(current_img, order=1)
                g_mag = torch.sqrt(grads[:, :, 0]**2 + grads[:, :, 1]**2 + 1e-6)
                
                # Flatten to find quantile
                flat_mag = g_mag.view(-1)
                high_val = torch.quantile(flat_mag, 0.99) # Scalar Tensor
                
                low_val = torch.quantile(flat_mag, 0.98) # Scalar Tensor
                high_val = torch.clamp(high_val, min=0.002, max=0.99)
                low_val = torch.clamp(low_val, min=0.001, max=0.98)
                if high_val <= low_val:
                            high_val = low_val + 0.001
                # --- Run Canny on Single Image ---
                # We use .item() to convert the 0-dim tensor to a float
                # This bypasses the "Boolean value of Tensor" error
                _, edges = kf.canny(
                    current_img, 
                    low_threshold=low_val.item(), 
                    high_threshold=high_val.item()
                )
                
                # Compute Thetas
                gx = grads[:, :, 0]
                gy = grads[:, :, 1]
                thetas = torch.atan2(gy, gx)
                
                batch_edges.append(edges)
                batch_thetas.append(thetas)

            # 3. Stack results back into a batch
            edges = torch.cat(batch_edges, dim=0)
            thetas = torch.cat(batch_thetas, dim=0)

            return edges, thetas



    # def get_edge(self, images):
    #     # Handle grayscale or RGB
    #     img_gray = images[:, 0:1, :, :] if images.shape[1] == 3 else images
    #     #bs = img_gray.shape[0]
    #     #sigma_color = torch.full((bs,), 0.1, device=img_gray.device)
    #     #sigma_space = torch.full((bs, 2), 1.5, device=img_gray.device)
    #     #magnitude, edges = kf.canny(img_gray, low_threshold=0.05, high_threshold=0.15)
    #     #sigma_space = torch.tensor([1.5], device=img_gray.device).repeat(img_gray.shape[0])
    #     #img_gray = kf.bilateral_blur(img_gray,kernel_size=(7,7),sigma_color=sigma_color,sigma_space=sigma_space)
    #     gx = F.conv2d(img_gray, self.sobel_x, padding=1)
    #     gy = F.conv2d(img_gray, self.sobel_y, padding=1)
        
    #     edges = torch.sqrt(gx**2 + gy**2)
    #     thetas = torch.atan2(gy, gx)
    #     return edges, thetas

    def forward(self, inputs, targets, images, masks=None):
        N, C, H, W = targets.shape
        if masks is None:
            masks = targets > self.mask_value
        
        # 1. Compute Edges and Thetas (Vectorized)
        edges, thetas = self.get_edge(images) # Log transform to enhance edges in depth maps
        
        # Flatten tensors for sampling: (N, H*W)

        flat_targets = targets.view(N, -1)
        flat_masks = masks.view(N, -1)
        flat_edges = edges.view(N, -1)
        flat_thetas = thetas.view(N, -1)
        
        # 2. Vectorized Sampling Strategy
        # Identify valid edge pixels (threshold > 10% of max per image)
        #q = 0.9
        #edge_thresholds = torch.quantile(flat_edges, q, dim=1, keepdim=True)
        #valid_edges = flat_edges >= edge_thresholds
        #edge_max_vals = flat_edges.max(dim=1, keepdim=True)[0]
        #valid_edges = flat_edges >= (edge_max_vals * 0.1)

        #edge_max_vals = flat_edges.max(dim=1, keepdim=True)[0]
        #valid_edges = flat_edges >= (edge_max_vals * 0.1)
        
        # Create sampling probabilities (uniform over valid edges)
        probs = flat_edges.float() + 1e-6 # Avoid zero sum
        
        # Sample anchor indices for the whole batch: (N, S)
        # We sample 'point_pairs' anchors per image
        sample_num = int(flat_edges.sum(dim=1).max().item())
        anchor_indices = torch.multinomial(probs, sample_num, replacement=False)
        
        # Gather anchor data
        # row/col of anchors
        anchor_rows = anchor_indices // W
        anchor_cols = anchor_indices % W
        anchor_thetas = torch.gather(flat_thetas, 1, anchor_indices)
        
        # 3. Generate Candidate Points (A, B, C, D) along gradient
        # Logic: sample 4 points at random distances [2, 30] along the normal
        # Shape: (4, N, S)
        distances = torch.randint(2, 31, (4, N, sample_num), device=inputs[0].device).float()
        
        # Directions: first 2 points negative, next 2 positive (matches original logic)
        signs = torch.tensor([-1.0, -1.0, 1.0, 1.0], device=inputs[0].device).view(4, 1, 1)
        distances = distances * signs
        
        # Compute offsets
        # cos/sin shape: (1, N, S) -> broadcast to (4, N, S)
        delta_cols = (distances * torch.cos(anchor_thetas).unsqueeze(0)).long()
        delta_rows = (distances * torch.sin(anchor_thetas).unsqueeze(0)).long()
        
        # Calculate new coordinates for 4 points
        cols = torch.clamp(anchor_cols.unsqueeze(0) + delta_cols, 0, W - 1)
        rows = torch.clamp(anchor_rows.unsqueeze(0) + delta_rows, 0, H - 1)
        
        # Convert back to flat indices: (4, N, S)
        sample_indices = rows * W + cols
        
        # 4. Form Pairs (A-B, B-C, C-D)
        # A_idx comes from row 0,1,2; B_idx comes from row 1,2,3
        # 1. Slice to get components
        # Shape: (3, N, S)
        A_components = sample_indices[:3] 
        B_components = sample_indices[1:]

        # 2. Permute to put Batch first: (N, 3, S)
        # This ensures that when we flatten, we keep each image's data together
        A_components = A_components.permute(1, 2, 0)
        B_components = B_components.permute(1, 2, 0)

        # 3. Reshape to (N, 3*S)
        idx_A = A_components.reshape(N, -1)
        idx_B = B_components.reshape(N, -1)

        #rs_probs = flat_masks.float() + 1e-6
        
        # We need 2 points per pair -> 2 * S samples
        #rs_indices = torch.multinomial(rs_probs, sample_num * 2, replacement=True)
        
        # Split into A and B (N, S)
        #idx_A_rs = rs_indices[:, 0::2]
        #idx_B_rs = rs_indices[:, 1::2]

        #idx_A = torch.cat([idx_A, idx_A_rs], dim=1)
        #idx_B = torch.cat([idx_B, idx_B_rs], dim=1)

        # Gather values for all pairs
        targets_A = torch.gather(flat_targets, 1, idx_A)
        targets_B = torch.gather(flat_targets, 1, idx_B)
        masks_A = torch.gather(flat_masks, 1, idx_A)
        masks_B = torch.gather(flat_masks, 1, idx_B)
        consistency = masks_A * masks_B
        target_ratio = torch.div(targets_A + 1e-6, targets_B + 1e-6)
        mask_eq = (target_ratio < (1.0 + self.sigma)) & (target_ratio > (1.0 / (1.0 + self.sigma)))
        labels = torch.zeros_like(target_ratio)
        labels[target_ratio >= (1.0 + self.sigma)] = 1
        labels[target_ratio <= (1.0 / (1.0 + self.sigma))] = -1
            
        valid_eq = mask_eq& consistency
        valid_uneq = (~mask_eq) & consistency
        loss_gamma = 0.9
        n_predictions = len(inputs)
        loss = torch.tensor(0.0, device=inputs[0].device, requires_grad=True)   
        for i in range(n_predictions):
            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            flat_inputs = inputs[i].view(N, -1)

            inputs_A = torch.gather(flat_inputs, 1, idx_A)
            inputs_B = torch.gather(flat_inputs, 1, idx_B)
            
            
            

            equal_loss= F.smooth_l1_loss(inputs_A, inputs_B, reduction='none', beta=1.0) * valid_eq.float()
            #equal_loss = (inputs_A - inputs_B).pow(2) * valid_eq.float()

            logit = (-inputs_A + inputs_B) * labels
            
            unequal_loss = F.softplus(logit) * valid_uneq.float()

            num_eq = valid_eq.sum().float().clamp(min=1.0)
            num_uneq = valid_uneq.sum().float().clamp(min=1.0)
            
            step_loss = self.alpha * (equal_loss.sum() / num_eq) + (unequal_loss.sum() / num_uneq)
            loss = loss + (i_weight * step_loss)

        return loss
    

def test_loss_example():
    # 1. Configuration
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Instantiate the Loss
    # Parameters taken from __init__ defaults in ranking_loss.py
    criterion = EdgeGuidedRankingLoss(point_pairs=5000) 
    criterion.to(device)

    # 3. Create Dummy Inputs
    
    # inputs: The output from your depth estimation model (e.g., DepthNet)
    # Random values generally normalized or in a specific depth range
    inputs = torch.rand((batch_size, 1, height, width), requires_grad=True).to(device)
    
    # targets: Ground truth depth maps (e.g., from Kinect or Lidar)
    targets = torch.rand((batch_size, 1, height, width)).to(device)
    
    # images: The RGB images input to the network (Normalized like in demo.py)
    images = torch.rand((batch_size, 3, height, width)).to(device)
    
    # masks: Valid pixels (1 for valid, 0 for invalid/missing depth)
    # For example, assume 90% of pixels have valid ground truth
    masks = (torch.rand((batch_size, 1, height, width)) > 0.1).float().to(device)

    # 4. Calculate Loss
    loss = criterion(inputs, targets, images, masks)

    print(f"Loss value: {loss.item()}")

if __name__ == "__main__":
    test_loss_example()