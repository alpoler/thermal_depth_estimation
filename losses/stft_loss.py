
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class STFTLoss(nn.Module):
    def __init__(self, window_size=32, stride=16):
        """
        STFT Loss Module.
        
        Args:
            window_size (int): Size of the window (and patch). Default: 32.
            stride (int): Stride of the window. Default: 16 (50% overlap).
        """
        super(STFTLoss, self).__init__()
        self.window_size = window_size
        self.stride = stride
        
        # Create Hamming window
        # multidimensional window: outer product of 1D windows
        window1d = torch.hamming_window(window_size)
        window2d = window1d[:, None] * window1d[None, :]
        self.register_buffer('window', window2d)

    def forward(self, input, target):
        """
        Compute STFT Loss between input and target.
        
        Args:
            input (torch.Tensor): Predicted map (B, C, H, W) or (B, H, W).
            target (torch.Tensor): Ground truth map (B, C, H, W) or (B, H, W).
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure 4D input
        if input.dim() == 3:
            input = input.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Check if input and target shapes match
        if input.shape != target.shape:
             raise ValueError(f"Input and target shapes must match. Got {input.shape} and {target.shape}")

        B, C, H, W = input.shape
        
        # Unfold input and target into patches using F.unfold
        # Unfold extracts sliding local blocks from a batched input tensor.
        # It flattens each patch into a column.
        # Output shape: (B, C * kernel_size * kernel_size, L)
        input_patches = F.unfold(input, kernel_size=self.window_size, stride=self.stride)
        target_patches = F.unfold(target, kernel_size=self.window_size, stride=self.stride)
        
        # Rearrange to (B * L, C, window_size, window_size) for applying window and FFT
        # Permute to (B, L, C * window_size * window_size) first
        input_patches = input_patches.transpose(1, 2)
        target_patches = target_patches.transpose(1, 2)
        
        # Reshape to (B * L, C, window_size, window_size)
        input_patches = input_patches.reshape(-1, C, self.window_size, self.window_size)
        target_patches = target_patches.reshape(-1, C, self.window_size, self.window_size)
       
        # Apply window
        window = self.window.to(input.device)
        input_patches = input_patches * window
        target_patches = target_patches * window
        
        # Compute FFT (real FFT)
        # torch.fft.rfft2 computes the 2D real-to-complex DFT
        # Returns a complex tensor of shape (..., window_size, window_size // 2 + 1)
        input_fft = torch.fft.rfft2(input_patches)
        target_fft = torch.fft.rfft2(target_patches)
        
        # 2. Compute Log-Magnitude (Your placement was correct!)
        mag_input = torch.abs(input_fft)
        mag_target = torch.abs(target_fft)
        mag_input = torch.log(mag_input + 1e-8)
        mag_target = torch.log(mag_target + 1e-8)

        # 3. High-Frequency Filtering (The Fix)
        # Instead of slicing, we zero out the low frequencies.
        # "cutoff" determines how many low-freq bins to ignore.
        cutoff = 2 
        h, w = mag_input.shape[-2:]
        
        # Create a mask of ones (keep everything by default)
        mask = torch.ones_like(mag_input)
        
        # Zero out Top-Left (Positive Low Freqs)
        mask[..., :cutoff, :cutoff] = 0
        
        # Zero out Bottom-Left (Negative Low Freqs in Height)
        # Note: rfft only has positive freqs in Width, so we only handle Height here.
        mask[..., -cutoff:, :cutoff] = 0
        
        # Apply the mask
        pred_high_input = mag_input * mask
        pred_high_target = mag_target * mask

        # 4. Compute Loss
        # Smooth L1 is great here.
        loss = F.smooth_l1_loss(pred_high_input, pred_high_target, reduction='mean')
        
        return loss