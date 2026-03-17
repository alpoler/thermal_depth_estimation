import torch
import torch.nn as nn
from topologylayer.nn import LevelSetLayer2D

class TopologyLayerLoss(nn.Module):
    def __init__(self, patch_size=(64, 64), theta=0.1):
        super(TopologyLayerLoss, self).__init__()
        self.theta = theta
        self.patch_size = patch_size
        
        # LevelSetLayer2D is designed for image grids
        # size: the dimensions of the input grid
        # sublevel=True: tracks features from min to max (rising water analogy)
        self.topo_layer = LevelSetLayer2D(size=patch_size, sublevel=True)

    def forward(self, h_student):
        # 1. Dimension Check
        if h_student.dim() == 4:
            h_student = h_student.squeeze(1)
            
        H, W = h_student.shape[-2], h_student.shape[-1]
        
        # 2. Random Patch Cropping
        # To keep training fast, we only compute topology on a small patch
        top = torch.randint(0, H - self.patch_size[0], (1,)).item()
        left = torch.randint(0, W - self.patch_size[1], (1,)).item()
        
        patch = h_student[..., top:top+self.patch_size[0], left:left+self.patch_size[1]].contiguous()
        
        loss_topo = torch.tensor(0.0, device=h_student.device)
        batch_size = patch.shape[0]

        # 3. Batch Processing
        for i in range(batch_size):
            # The layer returns a list of diagrams (dgms)
            # dgms[0] corresponds to 0D features (connected components)
            dgms, is_sublevel = self.topo_layer(patch[i])
            dg_0d = dgms[0]
            
            if dg_0d.shape[0] > 0:
                births = dg_0d[:, 0]
                deaths = dg_0d[:, 1]
                
                # Filter out the global minimum that lives to infinity
                valid_mask = deaths < float('inf')
                
                if valid_mask.any():
                    # Persistence is the lifespan (death - birth)
                    persistence = deaths[valid_mask] - births[valid_mask]
                    
                    # Target short-lived features (topological noise) where persistence < theta
                    short_lived_mask = persistence < self.theta
                    loss_topo += torch.sum(persistence[short_lived_mask])

        return loss_topo / batch_size