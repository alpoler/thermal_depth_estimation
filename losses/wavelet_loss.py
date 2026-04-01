import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.io import savemat

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()

class HFDTeacherBlock(nn.Module):
    def __init__(self, patch_size=16, stride=16):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.register_buffer('running_sigma', torch.tensor(0.0))
        self.register_buffer('running_grad', torch.tensor(0.0))
        self.momentum = 0.9
        # Initialize pytorch_wavelets DWT/IDWT
        # We need J=3 to match the original 3-level decomposition
        try:
            from pytorch_wavelets import DWTForward, DWTInverse,DTCWTForward, DTCWTInverse
        except ImportError:
            raise ImportError("Please install pytorch-wavelets: pip install pytorch-wavelets")
            
        self.dwt = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        self.idwt = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        
        # Sobel for global gradient
        self.register_buffer('sobel_x', torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))

    def get_stats_maps(self, x):
        # 1. Local Mean and Variance via Pooling
        mu = F.avg_pool2d(x, self.patch_size, stride=self.stride)
        mu_sq = F.avg_pool2d(x**2, self.patch_size, stride=self.stride)
        var = mu_sq - mu**2
        
        # 2. Local Gradient Magnitude
        grad_x = F.conv2d(F.pad(x, (1,1,1,1), mode='replicate'), self.sobel_x)
        grad_y = F.conv2d(F.pad(x, (1,1,1,1), mode='replicate'), self.sobel_y)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        # Average gradient per "patch" area
        mean_grad = F.avg_pool2d(grad_mag, self.patch_size, stride=self.stride)
        
        return var, mean_grad

    def update_thresholds(self, current_var_mean, current_grad_mean):
            """Updates the running mean buffers using momentum."""
            if self.training:
                # Apply momentum update: running = momentum * running + (1 - momentum) * current
                self.running_sigma = (self.momentum * self.running_sigma) + \
                                    ((1 - self.momentum) * current_var_mean.detach())
                self.running_grad = (self.momentum * self.running_grad) + \
                                    ((1 - self.momentum) * current_grad_mean.detach())

    def get_masks(self, x):
        B, C, H, W = x.shape
        var_map, grad_map = self.get_stats_maps(x)
        current_sigma_mean = var_map.mean()
        current_grad_mean = grad_map.mean()
        self.update_thresholds(current_sigma_mean, current_grad_mean)
        t_sigma = self.running_sigma.detach()
        t_grad = self.running_grad.detach()
        # 2. Define masks at the "patch resolution"
        # Level 3: Both high
        m3 = ((var_map > t_sigma) & (grad_map > t_grad)).float()
        # Level 2: Either high (exclusive of level 3)
        m2 = (((var_map > t_sigma) | (grad_map > t_grad)).float() - m3).clamp(0, 1)
        # Level 1: Rest
        m1 = (1.0 - (m3 + m2)).clamp(0, 1)

        # Upsample masks back to image resolution to use as blending weights
        # Use 'nearest' to maintain the "patchy" logic you had before
        M1 = F.interpolate(m1, size=(H, W), mode='nearest')
        M2 = F.interpolate(m2, size=(H, W), mode='nearest')
        M3 = F.interpolate(m3, size=(H, W), mode='nearest')
        
        return M1, M2, M3

    def forward(self, x, masks=None):
        with torch.amp.autocast('cuda', enabled=False):
            # Ensure input is float32 for wavelet and sobel operations
            x = x.float()
            B, C, H, W = x.shape
            
            # 1. Get Patch-based stats maps (Downsampled)
            if masks is None:
                M1, M2, M3 = self.get_masks(x)
            else:
                M1, M2, M3 = masks

            # 3. Global Wavelet Decomposition
            # yl is the lowest frequency (approx) at level 3
            # yh is a list of [level1_coeffs, level2_coeffs, level3_coeffs]
            # In pytorch_wavelets, yh[0] is the finest scale, yh[-1] is the coarsest.
            yl, yh = self.dwt(x)

            # 4. Reconstruct only the High Frequencies (LL=0)
            
            # Reconstruct Level 1 HF (Use only level 3 coeffs - Coarsest Detail)
            # We want to capture large-scale structures for smooth regions.
            # Passing [0, 0, yh[2]] to idwt.
            zero_h0 = torch.zeros_like(yh[0])
            zero_h1 = torch.zeros_like(yh[1])
            #zero_h2 = torch.zeros_like(yh[2])
            zero_ll3 = torch.zeros_like(yl)
            h1 = self.idwt((zero_ll3, [zero_h0, zero_h1, yh[2]]))
            
            # Reconstruct Level 2 HF (Use level 2 and level 3 coeffs - Mid + Coarse Detail)
            # We want to capture mid-level structures.
            # Passing [0, yh[1], yh[2]] to idwt.
            h2 = self.idwt((zero_ll3, [zero_h0, yh[1], yh[2]]))
            
            # Reconstruct Level 3 HF (Use level 1, 2, 3 coeffs)
            # Coarsest level is Level 3 (yh[2]), matching yl.
           
            # Passing full yh list [finest, mid, coarsest]
            h3 = self.idwt((zero_ll3, [yh[0], yh[1], yh[2]]))

            # 5. Final Blend (The "Vectorized" logic)
            # Instead of looping through patches, we multiply the full image by the mask
            out = (h1 * M1) + (h2 * M2) + (h3 * M3)
            
            return out, (M1, M2, M3)
        
# Example Usage
if __name__ == "__main__":
    # Create a dummy depth map (Batch 1, 1 Channel, 256x256)
    #/mnt/my_disk/alper/thermal/zero_shot_thermal/_2021-08-13-16-08-46/000928.npz
    #/mnt/my_disk/alper/thermal/zero_shot_thermal/_2021-08-13-17-06-04/007378_vis.png
    img = cv2.imread("/mnt/my_disk/alper/thermal/sync_data/_2021-08-13-17-06-04/thr/img_left/007300.png",flags=cv2.IMREAD_UNCHANGED)
    val_min = np.percentile(img,2)
    val_max = np.percentile(img,98)
    img = np.clip(img,min=val_min,max=val_max)
    img = (img-val_min)/(val_max-val_min+1e-8)
    img = (img*255).astype(np.uint8)
    data_dict= np.load("/mnt/my_disk/alper/thermal/zero_shot_thermal/_2021-08-13-17-06-04/007300.npz")
    data_dict = {key: data_dict[key] for key in data_dict.files}
    savemat("test_data.mat", data_dict)    
    disparity = data_dict["rgb_psuedo_disparity"] # It is thermal
    
    disparity = disparity[None,None,...]
    disparity = torch.from_numpy(disparity).float()
    # Initialize ALWT module
    alwt = HFDTeacherBlock(patch_size=12,stride=8)
    
    # Forward pass
    hf_map,a = alwt(disparity)
    
    hf_map = hf_map[0,0,...].numpy()
    hf_map = (hf_map-hf_map.min())/(hf_map.max()-hf_map.min())
    disparity = (disparity[0,0] - disparity.min())/(disparity.max()-disparity.min())
    disparity = (disparity*255).numpy().astype(np.uint8)
    hf_map = (hf_map*255).astype(np.uint8)
    cv2.imwrite("thr_img.png",img)
    cv2.imwrite("disp.png", disparity)
    cv2.imwrite("wave_map.png",hf_map)
    print(f"Input shape: {disparity.shape}")
    print(f"Output HF Map shape: {hf_map.shape}")
    print("Standard deviation of HF Map (should be lower if LL removed):", hf_map.std())