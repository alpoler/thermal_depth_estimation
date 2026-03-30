import os
import os.path as osp
import math
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mmengine.config import Config
from models.stereo_model import FoundationLighting
from dataloaders import build_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from core.foundation_stereo import FoundationStereo as FoundationStereoOriginal, FoundationStereoFaithful
from core.foundation_stereo_lbp import FoundationStereo as FoundationStereoLBP
from utils.visualization import visualize_depth, visualize_image
from metrics.eval_metric import compute_depth_errors, compute_disp_errors


def parse_args():
    parser = ArgumentParser()

    # configure file (same options as train.py)
    parser.add_argument('--config', default="/home/akayabasi/thermal_depth_estimation/23-51-11/Base_Sup_Stereo_Depth.yaml",
                        help='config file path')
    parser.add_argument('--out_dir', type=str, default='/mnt/mydisk/alper/thermal_depth_exps',
                        help='base output directory for saving predictions')
    parser.add_argument('--exp_name', type=str, default='predict', help='experiment name')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--ckpt_path', type=str,
                        default="/mnt/mydisk/alper/thermal_depth_exps/exp_wavelet_loss_faithful/ckpt_epoch=38_step=74646.ckpt",
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
                        help='save raw disparity/depth/uncertainty as .npy files')

    return parser.parse_args()


def inference_with_uncertainty(model, left_img, right_img, valid_iters, psuedo_depth=None):
    """Run inference and return disparity + uncertainty (variance) if available."""
    B, C, H, W = left_img.shape
    if C == 1:
        left_img = left_img.repeat_interleave(3, axis=1)
        right_img = right_img.repeat_interleave(3, axis=1)

    init_disp, pred_disp_pyramid, log_scale_preds = model.disp_net(
        left_img, right_img, iters=valid_iters, psuedo_depth=psuedo_depth
    )
    pred_disp = pred_disp_pyramid[-1]
    uncertainty = log_scale_preds[0] if log_scale_preds is not None and len(log_scale_preds) > 0 else None
    return pred_disp, uncertainty


def save_uncertainty_visualization(left_vis, pred_depth, pred_disp, uncertainty,
                                   save_path, seq_name, img_name):
    """Save an interactive Plotly HTML with input | depth | disparity | uncertainty.
    Hover over any pixel to see exact values."""
    # Convert tensors to numpy
    img_np = left_vis.cpu().numpy()
    if img_np.shape[0] == 1:
        img_np = np.repeat(img_np, 3, axis=0)
    img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, 3)
    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

    depth_np = np.nan_to_num(pred_depth.squeeze().cpu().numpy())
    disp_np = np.nan_to_num(pred_disp.squeeze().cpu().numpy())

    has_unc = uncertainty is not None
    n_cols = 4 if has_unc else 3
    subplot_titles = ['Input (Thermal)', 'Predicted Depth (m)', 'Predicted Disparity (px)']
    if has_unc:
        subplot_titles.append('Uncertainty (Variance)')

    fig = make_subplots(rows=n_cols, cols=1, subplot_titles=subplot_titles,
                        vertical_spacing=0.06)

    # Input image
    fig.add_trace(go.Image(z=img_np), row=1, col=1)

    # Depth heatmap — hover shows depth in meters
    fig.add_trace(
        go.Heatmap(z=np.flipud(depth_np), colorscale='Turbo',
                   zmin=np.percentile(depth_np[depth_np > 0], 2) if (depth_np > 0).any() else 0,
                   zmax=np.percentile(depth_np[depth_np > 0], 98) if (depth_np > 0).any() else 1,
                   colorbar=dict(title='m', len=1.0 / n_cols, y=1 - 1.5 / n_cols * 0.95),
                   hovertemplate='x: %{x}<br>y: %{y}<br>depth: %{z:.2f} m<extra></extra>'),
        row=2, col=1)

    # Disparity heatmap — hover shows disparity in pixels
    fig.add_trace(
        go.Heatmap(z=np.flipud(disp_np), colorscale='Jet',
                   zmin=np.percentile(disp_np[disp_np > 0], 2) if (disp_np > 0).any() else 0,
                   zmax=np.percentile(disp_np[disp_np > 0], 98) if (disp_np > 0).any() else 1,
                   colorbar=dict(title='px', len=1.0 / n_cols, y=1 - 2.5 / n_cols * 0.95),
                   hovertemplate='x: %{x}<br>y: %{y}<br>disp: %{z:.2f} px<extra></extra>'),
        row=3, col=1)

    # Uncertainty heatmap — hover shows variance value
    if has_unc:
        unc_np = np.nan_to_num(uncertainty.squeeze().cpu().numpy())
        fig.add_trace(
            go.Heatmap(z=np.flipud(unc_np), colorscale='Inferno',
                       zmin=np.percentile(unc_np, 2),
                       zmax=np.percentile(unc_np, 98),
                       colorbar=dict(title='var', len=1.0 / n_cols, y=1 - 3.5 / n_cols * 0.95),
                       hovertemplate='x: %{x}<br>y: %{y}<br>variance: %{z:.4f}<extra></extra>'),
            row=4, col=1)

    H, W = depth_np.shape
    fig.update_layout(
        title=dict(text=f'{seq_name} / {img_name}', x=0.5),
        height=max(300 * n_cols, H * n_cols),
        width=max(800, W * 1.3),
        margin=dict(l=10, r=80, t=60, b=10),
    )

    # Match aspect ratios for all subplots
    for i in range(1, n_cols + 1):
        xaxis = f'xaxis{i}' if i > 1 else 'xaxis'
        yaxis = f'yaxis{i}' if i > 1 else 'yaxis'
        fig.layout[xaxis].update(showticklabels=False)
        fig.layout[yaxis].update(showticklabels=False, scaleanchor=xaxis.replace('axis', ''))

    os.makedirs(save_path, exist_ok=True)
    out_file = osp.join(save_path, f'{seq_name}_{img_name}.html')
    fig.write_html(out_file)
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
    args_foundation_stereo = OmegaConf.create(cfg_foundation_stereo)

    # ---- Build backbone model (same logic as train.py) ----
    model_type = cfg.model.get('model_type', 'foundation_stereo')
    use_faithful_loss = cfg.loss.get('faithful_loss', False)
    if model_type == 'foundation_stereo_lbp':
        print("Initializing FoundationStereoLBP model...")
        model = FoundationStereoLBP(args_foundation_stereo)
    elif use_faithful_loss:
        print("Initializing FoundationStereoFaithful model...")
        model = FoundationStereoFaithful(args_foundation_stereo)
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

    # ---- Run prediction with Lightning Trainer ----
    # We use manual inference loop for uncertainty extraction (Trainer.predict doesn't
    # easily expose extra outputs), but we keep the same device/precision setup.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lightning_model = lightning_model.to(device)
    lightning_model.eval()

    # Access the underlying dataset's sample list to get image paths
    # The dataset stores samples[modality] as a list of dicts with 'tgt_img_left' paths
    base_dataset = eval_dataset
    modality = cfg.dataset.MS2.val.modality if args.split == 'val' else cfg.dataset.MS2.test.modality
    has_samples = hasattr(base_dataset, 'samples') and modality in base_dataset.samples

    all_errs = []
    valid_iters = cfg.model.valid_iters

    print(f"Saving visualizations to: {vis_dir}")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
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

            # Inference with uncertainty
            pred_disp, uncertainty = inference_with_uncertainty(
                lightning_model, left_img, right_img, valid_iters, psuedo_depth
            )
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
                # Extract sequence and image name from path like:
                # .../sync_data/urban00/thr/img_left/00001.png
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
            save_uncertainty_visualization(
                left_vis[0], pred_depth[0], pred_disp[0].cpu(), uncertainty,
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
                if uncertainty is not None:
                    np.save(osp.join(raw_sample_dir, f'{img_name}_uncertainty.npy'),
                            uncertainty[0].cpu().float().numpy())

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
