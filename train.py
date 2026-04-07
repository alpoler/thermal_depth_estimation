import os
import os.path as osp

from argparse import ArgumentParser

from mmengine.config import Config
from models.stereo_model import FoundationLighting  
from dataloaders import build_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning.strategies import DDPStrategy
from core.foundation_stereo import FoundationStereo as FoundationStereoOriginal
from core.foundation_stereo_lbp import FoundationStereo as FoundationStereoLBP
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.distributed as dist
import torch

def parse_args():
    parser = ArgumentParser()

    # configure file
    parser.add_argument('--config',default="/home/akayabasi/thermal_depth_estimation/23-51-11/Base_Sup_Stereo_Depth.yaml", help='config file path')
    parser.add_argument('--out_dir' , type=str, default='checkpoints')
    parser.add_argument('--exp_name', type=str, default='test_', help='experiment name')
    parser.add_argument('--num_gpus', type=int, default=3, help='number of gpus')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--ckpt_path', type=str, default="/mnt/my_disk/alper/foundation_stereo_pretrained/model_best_bp2.pth",
                        help='pretrained checkpoint path to load')
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')

    return parser.parse_args()

if __name__ == '__main__':

    # parse args
    args = parse_args()
    # parse cfg
    cfg = Config.fromfile(osp.join(args.config))
    
    ckpt_dir = args.ckpt_path
    cfg_foundation_stereo = OmegaConf.load(f'/home/akayabasi/thermal_depth_estimation/23-51-11/cfg.yaml')
    if 'vit_size' not in cfg_foundation_stereo:
        cfg_foundation_stereo['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg_foundation_stereo[k] = args.__dict__[k]
    # Derive image_width from the training modality's train_size
    train_modality = cfg.dataset[cfg.dataset.list[0]].train.modality
    cfg_foundation_stereo['image_width'] = cfg.dataset[cfg.dataset.list[0]][train_modality].train_size[1]
    args_foundation_stereo = OmegaConf.create(cfg_foundation_stereo)

    model_type = cfg.model.get('model_type', 'foundation_stereo')
    if model_type == 'foundation_stereo_lbp':
        print(f"Initializing FoundationStereoLBP model...")
        model = FoundationStereoLBP(args_foundation_stereo)
        strict_loading = False
    else:
        print(f"Initializing FoundationStereoOriginal model...")
        model = FoundationStereoOriginal(args_foundation_stereo)
        strict_loading = True

    ckpt = torch.load(ckpt_dir)
    if strict_loading:
        # Allow missing disp_init buffers (newly added, safe to re-initialize)
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        if missing:
            print(f"[INFO] Missing keys: {sorted(missing)}")
        if unexpected:
            print(f"[INFO] Unexpected keys: {sorted(unexpected)}")
    else:
        model.load_state_dict(ckpt['model'], strict=False)
    
    # show information
    print(f'Now training with {args.config}...')

    # configure seed
    seed_everything(args.seed)

    # prepare data loader & ckpt_callback
    dataset = build_dataset(cfg.dataset, cfg.model.eval_mode, split='train_val')

    train_loader = DataLoader(dataset['train'],
                              batch_size=cfg.imgs_per_gpu,
                              shuffle=True,
                              num_workers=cfg.workers_per_gpu,
                              pin_memory=True,
                              drop_last=True,
                              persistent_workers=cfg.workers_per_gpu > 0)

    # define ckpt_callback
    val_loaders = []
    checkpoint_callbacks = []
    out_dir = cfg.get('out_dir', args.out_dir)
    exp_name = cfg.get('exp_name', args.exp_name)
    work_dir = osp.join(out_dir, exp_name)
    os.makedirs(work_dir, exist_ok=True)

    if 'depth' in cfg.model.eval_mode: 
      val_loader_  = DataLoader(dataset['val']['depth'],
                                batch_size=cfg.imgs_per_gpu,
                                shuffle=False,
                                num_workers=cfg.workers_per_gpu,
                                pin_memory=True,
                                drop_last=True,
                                persistent_workers=cfg.workers_per_gpu > 0)

      callback_   = ModelCheckpoint(dirpath=work_dir,
                                    save_weights_only=False,
                                    monitor='val_loss',
                                    mode='min',
                                    save_top_k=1,
                                    filename='ckpt_{epoch:02d}_{step}')
                                    # every_n_epochs=cfg.checkpoint_epoch_interval)

      val_loaders.append(val_loader_)             
      checkpoint_callbacks.append(callback_)                        

    print('{} samples found for training'.format(len(train_loader)))
    for idx, val_loader in enumerate(val_loaders):
      print('{} samples found for validatioin set {}'.format(len(val_loader), idx))

    # build model
    model = FoundationLighting(opt=cfg,model=model)

    # training
    # gradient clipping is handled manually when using aligned_mtl
    use_aligned_mtl = cfg.loss.get('grad_method', 'none') == 'aligned_mtl'
    trainer_kwargs = dict(
        strategy=DDPStrategy(find_unused_parameters=True) if args.num_gpus > 1 else None,
        accelerator="gpu",
        devices=args.num_gpus,
        default_root_dir=work_dir,
        num_nodes=1,
        num_sanity_val_steps=5,
        max_epochs=cfg.total_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        limit_train_batches=cfg.batch_lim_per_epoch,
        callbacks=checkpoint_callbacks,
        benchmark=True,
        precision="bf16",
        sync_batchnorm=True,
    )
    if not use_aligned_mtl:
        trainer_kwargs['gradient_clip_val'] = 1.0
        trainer_kwargs['gradient_clip_algorithm'] = 'norm'
    trainer = Trainer(**trainer_kwargs)
    try:
        resume_from_ckpt = None
        if args.resume is not None:
             ckpt = torch.load(args.resume)
             if 'optimizer_states' not in ckpt or not ckpt['optimizer_states']:
                 print(f"Propagating weights from {args.resume} (Optimizer state not found - Warm Start)")
                 model.load_state_dict(ckpt['state_dict'], strict=False)
             else:
                 print(f"Resuming full state from {args.resume}")
                 resume_from_ckpt = args.resume

        trainer.fit(model, train_loader, val_dataloaders=val_loader, ckpt_path=resume_from_ckpt)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()