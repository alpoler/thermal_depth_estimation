import gc
import os
import os.path as osp
from argparse import ArgumentParser

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mmengine.config import Config
from models.stereo_model import FoundationLighting
from dataloaders import build_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from core.foundation_stereo import FoundationStereo as FoundationStereoOriginal
from core.foundation_stereo_lbp import FoundationStereo as FoundationStereoLBP
from utils.visualization import visualize_depth, visualize_image
from metrics.eval_metric import compute_depth_errors, compute_disp_errors


def parse_args():
    parser = ArgumentParser()

    # configure file (same options as train.py)
    parser.add_argument('--config', default="/home/akayabasi/thermal_depth_estimation/23-51-11/Base_Sup_Stereo_Depth.yaml",
                        help='config file path')
    parser.add_argument('--out_dir', type=str, default='/mnt/my_disk/alper/thermal_depth_exps',
                        help='base output directory for saving predictions')
    parser.add_argument('--exp_name', type=str, default='predict', help='experiment name')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--ckpt_path', type=str,
                        default="/mnt/my_disk/alper/thermal_depth_exps/exp_unfrozen_cnet/ckpt_epoch=74_step=191400.ckpt",
                        help='pretrained checkpoint path for model backbone')
    parser.add_argument('--resume', type=str, default=None,
                        help='lightning checkpoint to load (contains full training state)')

    # prediction-specific options
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test', 'test_day', 'test_night', 'test_rain'],
                        help='data split to run prediction on')
    parser.add_argument('--sequence', type=str, default=None,
                        help='run on a specific sequence name (overrides split)')
    parser.add_argument('--save_raw', action='store_true',
                        help='save raw disparity/depth as .npy files')

    return parser.parse_args()


def save_prediction_visualization(left_vis, pred_depth, pred_disp,
                                  save_path, seq_name, img_name):
    """Save a matplotlib figure with input | depth | disparity."""
    # Convert tensors to numpy
    img_np = left_vis.cpu().numpy()
    if img_np.shape[0] == 1:
        img_np = np.repeat(img_np, 3, axis=0)
    img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, 3)
    img_np = np.clip(img_np, 0, 1)

    depth_np = np.nan_to_num(pred_depth.squeeze().cpu().numpy())
    disp_np = np.nan_to_num(pred_disp.squeeze().cpu().numpy())

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].imshow(img_np)
    axes[0].set_title('Input (Thermal)')
    axes[0].axis('off')

    d_vmin = np.percentile(depth_np[depth_np > 0], 2) if (depth_np > 0).any() else 0
    d_vmax = np.percentile(depth_np[depth_np > 0], 98) if (depth_np > 0).any() else 1
    im_depth = axes[1].imshow(depth_np, cmap='turbo', vmin=d_vmin, vmax=d_vmax)
    axes[1].set_title('Predicted Depth (m)')
    axes[1].axis('off')
    fig.colorbar(im_depth, ax=axes[1], fraction=0.046, pad=0.04)

    disp_vmin = np.percentile(disp_np[disp_np > 0], 2) if (disp_np > 0).any() else 0
    disp_vmax = np.percentile(disp_np[disp_np > 0], 98) if (disp_np > 0).any() else 1
    im_disp = axes[2].imshow(disp_np, cmap='jet', vmin=disp_vmin, vmax=disp_vmax)
    axes[2].set_title('Predicted Disparity (px)')
    axes[2].axis('off')
    fig.colorbar(im_disp, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f'{seq_name} / {img_name}', fontsize=14)
    fig.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    out_file = osp.join(save_path, f'{seq_name}_{img_name}.png')
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_file


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(osp.join(args.config))

    # ---- Foundation Stereo config (same as train.py) ----
    cfg_foundation_stereo = OmegaConf.load('/home/akayabasi/thermal_depth_estimation/23-51-11/cfg.yaml')
    if 'vit_size' not in cfg_foundation_stereo:
        cfg_foundation_stereo['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg_foundation_stereo[k] = args.__dict__[k]
    # Derive image_width from the training modality's train_size
    train_modality = cfg.dataset[cfg.dataset.list[0]].train.modality
    cfg_foundation_stereo['image_width'] = cfg.dataset[cfg.dataset.list[0]][train_modality].train_size[1]
    args_foundation_stereo = OmegaConf.create(cfg_foundation_stereo)

    # ---- Build backbone model (same logic as train.py) ----
    model_type = cfg.model.get('model_type', 'foundation_stereo')
    if model_type == 'foundation_stereo_lbp':
        print("Initializing FoundationStereoLBP model...")
        model = FoundationStereoLBP(args_foundation_stereo)
    else:
        print("Initializing FoundationStereoOriginal model...")
        model = FoundationStereoOriginal(args_foundation_stereo)

    seed_everything(args.seed)

    # ---- Load model via Lightning interface ----
    lightning_model = FoundationLighting.load_from_checkpoint(
        args.ckpt_path,
        opt=cfg,
        model=model,
        strict=False,
    )
    print(f"Loaded checkpoint from {args.ckpt_path}")

    # ---- Determine split and build dataset ----
    # Override test split if --sequence is given
    if args.sequence is not None:
        cfg.dataset.MS2['test_env'] = args.sequence
        data_split = 'test'
    else:
        split_map = {
            'val': 'train_val',
            'test': 'test',
            'test_day': 'test',
            'test_night': 'test',
            'test_rain': 'test',
        }
        if args.split in ('test_day', 'test_night', 'test_rain'):
            cfg.dataset.MS2['test_env'] = args.split
        data_split = split_map[args.split]

    dataset = build_dataset(cfg.dataset, cfg.model.eval_mode, split=data_split)

    if args.split == 'val':
        eval_dataset = dataset['val']['depth']
    else:
        eval_dataset = dataset['test']['depth']

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers_per_gpu,
        pin_memory=True,
        drop_last=False,
        persistent_workers=cfg.workers_per_gpu > 0,
    )

    print(f"Running prediction on '{args.split}' split: {len(eval_loader)} samples")

    # ---- Output directory ----
    out_dir = cfg.get('out_dir', args.out_dir)
    exp_name = cfg.get('exp_name', args.exp_name)
    save_base = osp.join(args.out_dir, exp_name, f'predictions_{args.split}')
    vis_dir = osp.join(save_base, 'visualizations')
    raw_dir = osp.join(save_base, 'raw')
    os.makedirs(vis_dir, exist_ok=True)
    if args.save_raw:
        os.makedirs(raw_dir, exist_ok=True)

    # ---- Run prediction ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    lightning_model = lightning_model.to(device)
    lightning_model.eval()

    # Access the underlying dataset's sample list to get image paths
    base_dataset = eval_dataset
    modality = cfg.dataset.MS2.val.modality if args.split == 'val' else cfg.dataset.MS2.test.modality
    has_samples = hasattr(base_dataset, 'samples') and modality in base_dataset.samples

    all_errs = []
    valid_iters = cfg.model.valid_iters

    print(f"Saving visualizations to: {vis_dir}")
    amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if use_amp else torch.nn.Identity()
    with torch.no_grad(), amp_ctx:
        for batch_idx, batch in enumerate(eval_loader):
            left_img = batch["tgt_left"].to(device)
            right_img = batch["tgt_right"].to(device)
            left_vis = batch["tgt_left_eh"]
            depth_gt = batch["tgt_depth_gt"]
            disp_gt = batch["tgt_disp_gt"]
            focal = batch["focal"]
            baseline = batch["baseline"]

            psuedo_depth = ((focal[..., None, None] * baseline[..., None]) /
                            batch["thr_psuedo_disparity"]).clamp(min=1e-3)
            psuedo_depth = psuedo_depth.unsqueeze(1).to(device)

            # Inference (model returns init_disp, pred_disp_pyramid)
            pred_disp = lightning_model.inference_disp(left_img, right_img, psuedo_depth)
            pred_depth = baseline[0] * focal[0] / (pred_disp.cpu() + 1e-10)

            # Compute metrics
            errs_disp = compute_disp_errors(disp_gt, pred_disp.cpu())
            errs_depth = compute_depth_errors(depth_gt, pred_depth,
                                              dataset=cfg.dataset.list[0], align=False)
            errs = {
                'abs_rel': errs_depth[1], 'sq_rel': errs_depth[2],
                'rmse': errs_depth[4], 'rmse_log': errs_depth[5],
                'a1': errs_depth[6], 'a2': errs_depth[7], 'a3': errs_depth[8],
                'epe': errs_disp[0], 'd1': errs_disp[1],
                'thres1': errs_disp[2], 'thres2': errs_disp[3], 'thres3': errs_disp[4],
            }
            all_errs.append(errs)

            # Determine sequence name and image name from dataset
            if has_samples:
                sample_info = base_dataset.samples[modality][batch_idx]
                img_path = str(sample_info['tgt_img_left'])
                parts = img_path.replace('\\', '/').split('/')
                try:
                    sync_idx = parts.index('sync_data')
                    seq_name = parts[sync_idx + 1]
                    img_name = osp.splitext(parts[-1])[0]
                except ValueError:
                    seq_name = f'seq_{batch_idx}'
                    img_name = f'{batch_idx:06d}'
            else:
                seq_name = args.split
                img_name = f'{batch_idx:06d}'

            # Save visualization
            save_prediction_visualization(
                left_vis[0], pred_depth[0], pred_disp[0].cpu(),
                osp.join(vis_dir, seq_name), seq_name, img_name
            )

            # Save raw arrays
            if args.save_raw:
                raw_sample_dir = osp.join(raw_dir, seq_name)
                os.makedirs(raw_sample_dir, exist_ok=True)
                np.save(osp.join(raw_sample_dir, f'{img_name}_disp.npy'),
                        pred_disp[0].cpu().float().numpy())
                np.save(osp.join(raw_sample_dir, f'{img_name}_depth.npy'),
                        pred_depth[0].cpu().float().numpy())

            if (batch_idx + 1) % 50 == 0:
                print(f'  [{batch_idx + 1}/{len(eval_loader)}] '
                      f'EPE={errs["epe"]:.3f} abs_rel={errs["abs_rel"]:.4f}')

    # ---- Aggregate and print metrics ----
    print('\n' + '=' * 60)
    print(f'Results on {args.split} ({len(all_errs)} samples)')
    print('=' * 60)
    metric_names = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3',
                    'epe', 'd1', 'thres1', 'thres2', 'thres3']
    means = {}
    for m in metric_names:
        vals = np.array([e[m] for e in all_errs])
        means[m] = vals.mean()
        print(f'  {m:12s}: {means[m]:.4f}')

    # Save metrics to file
    metrics_file = osp.join(save_base, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f'Split: {args.split}\n')
        f.write(f'Samples: {len(all_errs)}\n')
        f.write(f'Config: {args.config}\n')
        f.write(f'Checkpoint: {args.ckpt_path}\n')
        f.write(f'Resume: {args.resume}\n\n')
        for m in metric_names:
            f.write(f'{m:12s}: {means[m]:.6f}\n')
    print(f'\nMetrics saved to: {metrics_file}')
    print(f'Visualizations saved to: {vis_dir}')
