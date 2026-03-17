# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch,pdb,os,sys
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import re
import sys
sys.path.append("/home/akayabasi/foundation_stereo_thermal/")
from torch.distributions import Beta
from core.utils.utils import bilinear_sampler
from core.extractor import ResidualBlock
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, dx=None):
        self.num_levels = num_levels
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []
        self.dx = dx

        # all pairs correlation
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d).contiguous()

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)


    def __call__(self, disp, coords, low_memory=False):
        b, _, h, w = disp.shape
        self.dx = self.dx.to(disp.device)
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            x0 = self.dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl, low_memory=low_memory)
            geo_volume = geo_volume.reshape(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + self.dx   # X on right image
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl, low_memory=low_memory)
            init_corr = init_corr.reshape(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        out_pyramid = torch.cat(out_pyramid, dim=-1)
        return out_pyramid.permute(0, 3, 1, 2).contiguous()   #(B,C,H,W)


    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.reshape(B, D, H, W1)
        fmap2 = fmap2.reshape(B, D, H, W2)
        with torch.cuda.amp.autocast(enabled=False):
          corr = torch.einsum('aijk,aijh->ajkh', F.normalize(fmap1.float(), dim=1), F.normalize(fmap2.float(), dim=1))
        corr = corr.reshape(B, H, W1, 1, W2)
        return corr
    
class LBPEncoder(nn.Module):
    """
    Computes the modified Local Binary Patterns (LBP) of an image using custom neighbor offsets.
    """
    def __init__(self, args):
        super(LBPEncoder, self).__init__()
        self.args = args
        self.lbp_neighbor_offsets = self._parse_offsets(self.args.lbp_neighbor_offsets)

        self._build_lbp_kernel()
        self.sigmoid = nn.Sigmoid()
    
    def _build_lbp_kernel(self):
        # Determine the kernel size based on the maximum offset
        self.num_neighbors = len(self.lbp_neighbor_offsets)
        self.max_offset = int(np.abs(self.lbp_neighbor_offsets).max())
        self.kernel_size = 2 * self.max_offset + 1
        self.padding = self.max_offset

        # Initialize the convolution layer for depthwise convolution
        self.lbp_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.num_neighbors,
            kernel_size=self.kernel_size,
            padding=self.padding,
            padding_mode="replicate",
            bias=False,
            groups=1  # Since in_channels=1, groups=1 makes it depthwise
        )

        self.lbp_weight = torch.zeros(self.num_neighbors, 1, 
                                    self.kernel_size, self.kernel_size).float()
        center_y, center_x = self.max_offset, self.max_offset
        for idx, (dy, dx) in enumerate(self.lbp_neighbor_offsets):
            # Compute the position in the kernel for the neighbor
            y, x = center_y + dy, center_x + dx
            if 0 <= y < self.kernel_size and 0 <= x < self.kernel_size:
                self.lbp_weight[idx, 0, y, x] = 1.0
                self.lbp_weight[idx, 0, center_y, center_x] = -1.0
            else:
                raise ValueError(f"Offset ({dy}, {dx}) is out of kernel bounds.")
        
        # Assign the weight to the convolution layer
        self.lbp_conv.weight = nn.Parameter(self.lbp_weight)
        self.lbp_conv.weight.requires_grad = False  # Ensure weights are not updated during training
    
    def _parse_offsets(self, offsets_str):
        """
        Parses a string to extract neighbor offsets.

        Parameters:
            offsets_str (str): String defining neighbor offsets, e.g., "(-1,-1), (1,1), (-1,1), (1,-1)"

        Returns:
            list of tuples: List of neighbor offsets.
        """
        # extract coordinate pairs
        pattern = r'\((-?\d+),\s*(-?\d+)\)'
        matches = re.findall(pattern, offsets_str)
        if not matches:
            raise ValueError(offsets_str + ": not suppoted format, please check it!")
        offsets = [(int(y), int(x)) for y, x in matches]
        return np.array(offsets)
        
        
    def forward(self, img):
        """
        Parameters:
            img (torch.Tensor): Grayscale image tensor of shape [N, 1, H, W].
        Returns:
            torch.Tensor: Modified LBP image of shape [N, C, H, W].
        """
        with torch.no_grad():
            # Apply convolution to compute differences directly
            differences = self.lbp_conv(img)  # Shape: [1, N, H, W] due to padding

            # Apply sigmoid to the differences to get encoding values between 0 and 1
            encoding = self.sigmoid(differences)  # Shape: [1, N, H, W]
        return encoding
    
class BetaModulator(nn.Module):
    def __init__(self, args, lbp_dim, hidden_dim=None, norm_fn='batch'):
        super(BetaModulator, self).__init__()
        self.norm_fn = norm_fn
        self.modulation_ratio = args.modulation_ratio
        # self.conv_depth = nn.Sequential(
        #     nn.Conv2d(8, 16, kernel_size=1, padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True),
        # )
        # self.conv_disp = nn.Sequential(
        #     nn.Conv2d(8, 16, kernel_size=1, padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True),
        # )
        if hidden_dim is None:
            hidden_dim = lbp_dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(lbp_dim*2, hidden_dim*2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1, bias=True),
        )
        down_dim = 64 if hidden_dim*2<64 else 128
        self.down = nn.Sequential(
            ResidualBlock(hidden_dim*2, down_dim, self.norm_fn, stride=2),
            ResidualBlock(down_dim, 128, self.norm_fn, stride=1)
        )
        self.up   = nn.ConvTranspose2d(128, hidden_dim*2, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim*4, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.Softplus(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1, padding=0, bias=False),
            nn.Softplus(),
        )
    
    def forward(self, lbp_disp, lbp_depth, out_distribution=False):
        x1 = self.conv1( torch.cat([lbp_disp, lbp_depth], dim=1) )
        x2 = self.up(self.down(x1))
        beta_paras = self.conv2( torch.cat([x1,x2], dim=1) ) + 1  # enforcing alpha>=1, beta>=1

        # build Beta distribution
        alpha, beta = torch.split(beta_paras, 1, dim=1)
        distribution = Beta(alpha, beta)

        if self.training:
            modulation = distribution.rsample()
        else:
            modulation = distribution.mean
        
        if not out_distribution:
            return modulation
        return modulation, distribution