import torch
import numpy as np
from losses.original_ranking import EdgeguidedRankingLoss
from losses.triplet_ranking_loss import EdgeguidedTripletLoss
from losses.gradient_loss import StableGradientLoss
from losses.stft_loss import STFTLoss
from pytorch_lightning import LightningModule
from metrics.eval_metric import compute_depth_errors, compute_disp_errors
from utils.visualization import *
from losses.wavelet_loss import HFDTeacherBlock, CharbonnierLoss
from losses.curvature_loss import CurvatureLoss
from losses.pairwise_ordinal_loss import ordinal_loss as pairwise_ordinal_loss
from losses.aligned_mtl import AlignedMTL
from losses.dtcwt_loss import DTCWTSubbandLoss
from core.utils.utils import rescale_modulation
import robust_loss_pytorch.adaptive

class StereoDepthBaseModule(LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.optim_opt = opt.optim
        self.train_iters = opt.model.train_iters
        self.valid_iters = opt.model.valid_iters
        self.dataset_name = opt.dataset.list[0]
        self.automatic_optimization = True
        self.pseudo_disparity = opt.loss.pseudo_disparity
        self.ordinal_loss_on = opt.loss.ordinal_loss
        self.gradient_loss_on = opt.loss.gradient_loss
        self.stft_loss_on = opt.loss.get('stft_loss', False)
        self.use_triplet_loss = opt.loss.get('triplet_loss', False)
        if self.ordinal_loss_on:
            if self.use_triplet_loss:
                self.ordinal_criteria = EdgeguidedTripletLoss(margin=opt.loss.get('triplet_margin', 2.0))
            else:
                self.ordinal_criteria = EdgeguidedRankingLoss()
        if self.gradient_loss_on:
            self.gradient_criterion = StableGradientLoss(3)
        if self.stft_loss_on:
            self.stft_criterion = STFTLoss(window_size=16,stride=8)
        self.wavelet_loss_on = opt.loss.get('wavelet_loss', False)
        if self.wavelet_loss_on:
            self.wavelet_criterion = HFDTeacherBlock(patch_size=16, stride=12)
            self.charbonnier_loss = CharbonnierLoss()
        self.curvature_loss_on = opt.loss.get('curvature_loss', False)
        self.curvature_loss_weight = opt.loss.get('curvature_loss_weight', 1.0)
        if self.curvature_loss_on:
            self.curvature_criterion = CurvatureLoss(scales=3, use_log=False)
        self.pairwise_ordinal_loss_on = opt.loss.get('pairwise_ordinal_loss', False)
        self.pairwise_ordinal_loss_weight = opt.loss.get('pairwise_ordinal_loss_weight', 1.0)
        self.pairwise_ordinal_n_pairs = opt.loss.get('pairwise_ordinal_n_pairs', 10000)
        self.pairwise_ordinal_delta = opt.loss.get('pairwise_ordinal_delta', 0.01)
        self.use_aligned_mtl = opt.loss.get('grad_method', 'none') == 'aligned_mtl'
        if self.use_aligned_mtl:
            self.aligned_mtl = AlignedMTL(weights=opt.loss.get('aligned_mtl_weights', [0.5, 0.5]))
            self.automatic_optimization = False
        self.dtcwt_loss_on = opt.loss.get('dtcwt_loss', False)
        self.dtcwt_loss_weight = opt.loss.get('dtcwt_loss_weight', 1.0)    
        if self.dtcwt_loss_on:
            dtcwt_J = opt.loss.get('dtcwt_scales', 3)
            dtcwt_alpha = opt.loss.get('dtcwt_alpha', 2.0)
            dtcwt_levels = opt.loss.get('dtcwt_levels', None)
            if dtcwt_levels is not None:
                dtcwt_levels = [int(l) for l in dtcwt_levels]
            self.dtcwt_criterion = DTCWTSubbandLoss(J=dtcwt_J, alpha=dtcwt_alpha, levels=dtcwt_levels)
        self.robust_loss_on = opt.loss.get('robust_loss', False)
        self.robust_loss_weight = opt.loss.get('robust_loss_weight', 1.0)
        if self.robust_loss_on:
            self.adaptive_loss = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
                num_dims=1, float_dtype=np.float32, device='cpu')
    def get_optimize_param(self):
        pass
    
    def get_losses(self):
        pass
    def psuedo_disparity_loss(self):
        pass
    def configure_optimizers(self):
        optim_params = self.get_optimize_param()

        if self.robust_loss_on:
            optim_params.append({
                'params': self.adaptive_loss.parameters(),
                'lr': self.optim_opt.learning_rate,
            })

        if self.optim_opt.optimizer == 'Adam' :
            optimizer = torch.optim.Adam(optim_params)
        elif self.optim_opt.optimizer == 'AdamW' :
            optimizer = torch.optim.AdamW(optim_params)
        elif self.optim_opt.optimizer == 'SGD' :
            optimizer = torch.optim.SGD(optim_params)

        if self.optim_opt.scheduler == 'CosineAnnealWarm' :
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                            T_0=self.optim_opt.CosineAnnealWarm.T_0, 
                                            T_mult=self.optim_opt.CosineAnnealWarm.T_mult,
                                            eta_min=self.optim_opt.CosineAnnealWarm.eta_min)
            return [optimizer], [scheduler]
        
        elif self.optim_opt.scheduler == 'CosineAnnealWithWarmup':
            eta_min = self.optim_opt.CosineAnnealWithWarmup.eta_min
            total_steps = self.trainer.estimated_stepping_batches
            # warmup_epochs overrides warmup_steps if provided
            warmup_epochs = self.optim_opt.CosineAnnealWithWarmup.get('warmup_epochs', None)
            if warmup_epochs is not None:
                steps_per_epoch = total_steps // self.trainer.max_epochs
                warmup_steps = warmup_epochs * steps_per_epoch
            else:
                warmup_steps = self.optim_opt.CosineAnnealWithWarmup.warmup_steps
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-1, total_iters=warmup_steps)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min)
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]),
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [scheduler]

        elif self.optim_opt.scheduler == 'OneCycleLR':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.optim_opt.learning_rate,
                    total_steps=self.trainer.estimated_stepping_batches,
                    pct_start=self.optim_opt.OneCycleLR.pct_start,
                    div_factor=self.optim_opt.OneCycleLR.div_factor,
                    final_div_factor=self.optim_opt.OneCycleLR.final_div_factor,
                    cycle_momentum=self.optim_opt.OneCycleLR.cycle_momentum,
                ),
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [scheduler]

        return [optimizer], [scheduler]

    def forward(self, left_img, right_img,iters,psuedo_depth=None):
        # in lightning, forward defines the prediction/inference actions for training
        B,C,H,W = left_img.shape
        if C == 1: # for the single-channel input
            left_img  = left_img.repeat_interleave(3, axis=1)
            right_img  = right_img.repeat_interleave(3, axis=1)

        predictions = self.disp_net(left_img, right_img,iters,psuedo_depth=psuedo_depth)
        return predictions

    def inference_disp(self, left_img, right_img, psuedo_depth=None):
        B,C,H,W = left_img.shape
        if C == 1:
            left_img  = left_img.repeat_interleave(3, axis=1)
            right_img  = right_img.repeat_interleave(3, axis=1)

        init_disp,pred_disp = self.disp_net(left_img, right_img, psuedo_depth=psuedo_depth)
        return pred_disp

    def training_step(self, batch, batch_idx):
        left_img = batch["tgt_left"]
        right_img = batch["tgt_right"]
        disp_gt = batch["tgt_disp_gt"]
        psuedo_disparity_gt=batch["thr_psuedo_disparity"]
        valid_regions_gt=batch["valid_psuedo_disparity"]
        focal     = batch["focal"]
        baseline  = batch["baseline"]
        psuedo_depth= ((focal[...,None,None]*baseline[...,None])) / psuedo_disparity_gt.clamp(min=1e-5)
        psuedo_depth = psuedo_depth.clamp(max=80)

        # network forward
        outputs = self.forward(left_img, right_img,iters=self.train_iters,psuedo_depth=psuedo_depth.unsqueeze(1))
        init_disp, predictions = outputs
        loss_depth = self.get_losses(init_disp, predictions, disp_gt)

        if self.pseudo_disparity:
            loss_reg = self.psuedo_disparity_loss(psuedo_disparity_gt, valid_regions_gt, predictions, init_disp, left_img)
        else:
            loss_reg = None

        if self.use_aligned_mtl and loss_reg is not None:
            opt = self.optimizers()
            trainable_params = [p for p in self.disp_net.parameters() if p.requires_grad]

            opt.zero_grad()
            self.manual_backward(loss_depth, retain_graph=True)
            grads_depth = torch.cat([p.grad.flatten() if p.grad is not None
                                     else torch.zeros(p.numel(), device=p.device)
                                     for p in trainable_params])

            opt.zero_grad()
            self.manual_backward(loss_reg)
            grads_reg = torch.cat([p.grad.flatten() if p.grad is not None
                                   else torch.zeros(p.numel(), device=p.device)
                                   for p in trainable_params])

            combined = self.aligned_mtl(grads_depth, grads_reg)

            idx = 0
            for p in trainable_params:
                length = p.numel()
                p.grad = combined[idx:idx + length].view(p.shape)
                idx += length

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            opt.step()

            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()

            total_loss = loss_depth + loss_reg
        else:
            total_loss = loss_depth
            if loss_reg is not None:
                total_loss = total_loss + loss_reg

        # record log
        self.log('train/total_loss', total_loss)
        self.log('train/depth_loss', loss_depth)
        if loss_reg is not None:
            self.log('train/reg_loss', loss_reg)

        if self.use_aligned_mtl and loss_reg is not None:
            return None
        return total_loss


    def validation_step(self, batch, batch_idx):

        left_img  = batch["tgt_left"]
        right_img = batch["tgt_right"]
        left_vis  = batch["tgt_left_eh"]
        depth_gt  = batch["tgt_depth_gt"]
        disp_gt   = batch["tgt_disp_gt"]
        focal     = batch["focal"]
        baseline  = batch["baseline"]

        psuedo_depth = ((focal[...,None,None]*baseline[...,None]) / batch["thr_psuedo_disparity"]).clamp(min=1e-3)
        psuedo_depth = psuedo_depth.unsqueeze(1)

        pred_disp  = self.inference_disp(left_img, right_img,psuedo_depth)
        pred_depth = baseline[0] * focal[0] / (pred_disp +1e-10)

        errs_disp  = compute_disp_errors(disp_gt, pred_disp)
        errs_depth = compute_depth_errors(depth_gt, pred_depth, dataset=self.dataset_name,align=False)

        errs = {'abs_rel': errs_depth[1], 'sq_rel': errs_depth[2], 
                'rmse': errs_depth[4], 'rmse_log': errs_depth[5],
                'a1': errs_depth[6], 'a2': errs_depth[7], 'a3': errs_depth[8],
                'epe' : errs_disp[0], 'd1' : errs_disp[1], 'thres1' : errs_disp[2],
                'thres2' : errs_disp[3], 'thres3' : errs_disp[4]}

        # plot
        if batch_idx < 2:
            if left_vis[0].size(-1) != pred_depth[0].size(-1):
                C,H,W = left_vis[0].size()
                pred_depth = torch.nn.functional.interpolate(pred_depth, [H, W], mode='nearest')
                pred_disp  = torch.nn.functional.interpolate(pred_disp, [H, W], mode='nearest')

            vis_img = visualize_image(left_vis[0])  # (3, H, W)
            vis_disp = visualize_depth(pred_disp[0].squeeze())  # (3, H, W)
            vis_depth = visualize_depth(pred_depth[0].squeeze())  # (3, H, W)
            stack = torch.cat([vis_img, vis_disp, vis_depth], dim=1).unsqueeze(0)  # (1, 3, 2*H, W)
            self.logger.experiment.add_images(
                'val/img_disp_depth_{}'.format(batch_idx), stack, self.current_epoch)
        return errs

    def validation_epoch_end(self, outputs):
        mean_rel    = np.array([x['abs_rel'] for x in outputs]).mean()
        mean_sq_rel = np.array([x['sq_rel'] for x in outputs]).mean()
        mean_rmse   = np.array([x['rmse'] for x in outputs]).mean()
        mean_rmse_log = np.array([x['rmse_log'] for x in outputs]).mean()

        mean_a1 = np.array([x['a1'] for x in outputs]).mean()
        mean_a2 = np.array([x['a2'] for x in outputs]).mean()
        mean_a3 = np.array([x['a3'] for x in outputs]).mean()

        mean_epe = np.array([x['epe'] for x in outputs]).mean()
        mean_d1 = np.array([x['d1'] for x in outputs]).mean()
        mean_th1 = np.array([x['thres1'] for x in outputs]).mean()
        mean_th2 = np.array([x['thres2'] for x in outputs]).mean()
        mean_th3 = np.array([x['thres3'] for x in outputs]).mean()

        self.log('val_loss', mean_epe, prog_bar=True)
        self.log('val/abs_rel', mean_rel)
        self.log('val/sq_rel', mean_sq_rel)
        self.log('val/rmse', mean_rmse)
        self.log('val/rmse_log', mean_rmse_log)
        self.log('val/a1', mean_a1)
        self.log('val/a2', mean_a2)
        self.log('val/a3', mean_a3)  

        self.log('val/epe', mean_epe)
        self.log('val/d1', mean_d1)
        self.log('val/th1', mean_th1)
        self.log('val/th2', mean_th2)
        self.log('val/th3', mean_th3)

    def test_step(self, batch_data, batch_idx, dataloader_idx=0):
        return self.validation_step(batch_data, batch_idx, dataloader_idx)

    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)
    
class FoundationLighting(StereoDepthBaseModule):
    def __init__(self, opt,model):
        super().__init__(opt)
        self.save_hyperparameters(ignore=['model'])

        # Network
        self.disp_net = model
        self.max_disp = opt.model.max_disp
        self.criterion = torch.nn.functional.smooth_l1_loss

        # Epsilon scheduling for OT disp_init: eps_t = eps_end^(step/total_steps)
        self.eps_end = opt.model.get('ot_eps_end', 0.5)

    def on_train_batch_start(self, batch, batch_idx):
        if hasattr(self.disp_net, 'disp_init') and hasattr(self.disp_net.disp_init, 'ot'):
            total_steps = self.trainer.estimated_stepping_batches
            t = self.global_step / max(total_steps - 1, 1)
            eps = self.eps_end ** t
            self.disp_net.disp_init.epsilon = eps
            self.disp_net.disp_init.ot.eps = eps
            if self.global_step % 100 == 0:
                self.log('ot_epsilon', eps)

    def get_optimize_param(self):
        unfrozed_keywords = ["update_block", "feature.deconv32_16", "feature.deconv16_8", "feature.deconv8_4","feature.conv4",
                              "cost_agg", "corr_feature_att", "corr_stem","cnet.conv2","cnet.outputs04","cnet.outputs08","cnet.outpus16","classifier"]

        # Unfreeze dustbin_cost if using dustbin OT
        from core.disp_init import DustbinOTDisparityInit
        if isinstance(self.disp_net.disp_init, DustbinOTDisparityInit):
            unfrozed_keywords.append("disp_init")

        # 2. Iterate through the network and freeze/unfreeze based on name
        for name, param in self.disp_net.named_parameters():
            if any(keyword in name for keyword in unfrozed_keywords):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 3. Pass ONLY the trainable parameters (requires_grad=True) to the optimizer
        optim_params = [
            {
                'params': filter(lambda p: p.requires_grad, self.disp_net.parameters()),
                'lr': self.optim_opt.learning_rate,
                'weight_decay': self.optim_opt.weight_decay
            },
        ]
       # optim_params = [
       #     {'params': self.disp_net.parameters(), 'lr': self.optim_opt.learning_rate, 'weight_decay': self.optim_opt.weight_decay},
       # ]
        return optim_params

    def inference_disp(self, left_img, right_img, psuedo_depth=None):
        B,C,H,W = left_img.shape
        if C == 1:
            left_img  = left_img.repeat_interleave(3, axis=1)
            right_img  = right_img.repeat_interleave(3, axis=1)

        outputs = self.disp_net(left_img, right_img,iters=self.valid_iters,psuedo_depth=psuedo_depth)
        init_disp, pred_disp_pyramid = outputs
        return pred_disp_pyramid[-1]


    
    def get_losses(self, disp_init_pred, disp_preds, disp_gt):


        mask = (disp_gt < self.max_disp) & (disp_gt > 0)
        valid = mask.float()

        disp_gt = disp_gt.unsqueeze(1)
        mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
        valid = ((valid >= 0.5) & (mag < self.max_disp)).unsqueeze(1)

        valid_bool = valid.bool()
        if valid_bool.sum() == 0:
            return torch.tensor(0.0, device=disp_init_pred.device, requires_grad=True)

        disp_init_pred = torch.nn.functional.interpolate(disp_init_pred, scale_factor=4, mode='bilinear', align_corners=True) * 4
        disp_loss = 1.0 * torch.nn.functional.smooth_l1_loss(disp_init_pred[valid_bool], disp_gt[valid_bool], reduction='mean')
        loss_gamma = 0.9
        n_predictions = len(disp_preds)

        for i in range(n_predictions):
            # Adjust gamma based on total number of steps to keep weights consistent
            adjusted_loss_gamma = loss_gamma ** (15 / max(n_predictions - 1, 1))

            # Exponentially increasing weight: Early steps have low weight, final step has high weight
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)

            if self.robust_loss_on:
                residuals = (disp_preds[i][valid_bool] - disp_gt[valid_bool]).unsqueeze(-1)
                i_loss = self.adaptive_loss.lossfun(residuals)
            else:
                # Absolute difference (L1 Loss)
                i_loss = (disp_preds[i][valid_bool] - disp_gt[valid_bool]).abs()
            disp_loss += i_weight * i_loss.mean()

        return disp_loss

    def weighted_sequence_loss(self, preds, target, criterion, gamma=0.9):
        loss_gamma = gamma
        n_predictions = len(preds)
        loss = 0.0
        
        for i in range(n_predictions):
            # Adjust gamma based on total number of steps to keep weights consistent
            adjusted_loss_gamma = loss_gamma ** (15 / max(n_predictions - 1, 1))
            
            # Exponentially increasing weight: Early steps have low weight, final step has high weight
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            #modulation_weight = rescale_modulation(i, n_predictions,"sigmoid", 1.0)  
            i_loss = criterion(preds[i], target)
            loss += i_weight * i_loss
        
        return loss

    def psuedo_disparity_loss(self,psuedo_disparity_gt,valid_regions_gt,predictions,init_disp,left_img):

        #if torch.any(valid_regions_gt):
        disp_loss = torch.zeros((), device=psuedo_disparity_gt.device, dtype=torch.float32, requires_grad=True)
        if self.ordinal_loss_on:
            loss_gamma = 0.9
            n_predictions = len(predictions)
            pseudo_mask = (psuedo_disparity_gt > 0) & (psuedo_disparity_gt < 256)

            for i in range(n_predictions):
                # Adjust gamma based on total number of steps to keep weights consistent
                adjusted_loss_gamma = loss_gamma ** (15 / max(n_predictions - 1, 1))

                # Exponentially increasing weight: Early steps have low weight, final step has high weight
                i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
                #modulation_weight = rescale_modulation(i, n_predictions,"sigmoid", 1.0)
                # Clamp predictions to valid range before ranking loss to prevent negative disparities
                clamped_pred = predictions[i].clamp(0, self.max_disp)
                i_loss = self.ordinal_criteria(clamped_pred, psuedo_disparity_gt.unsqueeze(1), left_img, masks=pseudo_mask.unsqueeze(1))
                disp_loss = disp_loss +  i_weight * i_loss

            self.log('train/ranking_loss', disp_loss.detach(), prog_bar=True)

        if self.gradient_loss_on:
            def grad_crit(pred, target):
                    return self.gradient_criterion(pred, target.unsqueeze(1), valid_regions_gt.unsqueeze(1)).mean()

            disp_loss = disp_loss + self.weighted_sequence_loss(predictions, psuedo_disparity_gt, grad_crit, gamma=0.9)   

        if self.stft_loss_on:
            # 1. Init Disp (upsampled)
            
            target_masked = psuedo_disparity_gt.unsqueeze(1) 
           
            def stft_crit(pred, target):
                return self.stft_criterion(pred, target)

            stft_seq_loss = self.weighted_sequence_loss(predictions, target_masked, stft_crit, gamma=0.9)
            disp_loss = disp_loss + stft_seq_loss
        if self.dtcwt_loss_on:
            target_masked = psuedo_disparity_gt.unsqueeze(1)

            def dtcwt_crit(pred, target):
                return self.dtcwt_criterion(pred, target)

            dtcwt_loss = self.weighted_sequence_loss(predictions, target_masked, dtcwt_crit, gamma=0.9)
            disp_loss = disp_loss + self.dtcwt_loss_weight * dtcwt_loss
            self.log('train/dtcwt_loss', dtcwt_loss.detach(), prog_bar=False)    
            
        if self.wavelet_loss_on:
            # 1. Init Disp (upsampled)
            target_masked = psuedo_disparity_gt.unsqueeze(1)
            
            # Pre-compute target HF map and mask
            # We assume target structure is consistent for the batch
            target_hf, (M1, M2, M3) = self.wavelet_criterion(target_masked)

            mask = (torch.abs(target_hf) > 0.5) #& valid_regions_gt.unsqueeze(1).bool() 
            
            def masked_charbonnier_loss(pred_hf, target_hf, mask):
                # Apply mask to select valid pixels
                # mask is 1 where we want to compute loss (assuming standard mask usage)
                # target_hf and pred_hf are full images.
                # If mask is boolean or 0/1, we can select:
                if mask.sum() < 1.0:
                    return torch.tensor(0.0, device=pred_hf.device, requires_grad=True)
                
                # Select only masked pixels (flattening happens here)
                pred_valid = pred_hf[mask]
                target_valid = target_hf[mask]
                
                # Apply Charbonnier Loss on valid pixels
                return torch.nn.functional.smooth_l1_loss(pred_valid, target_valid)

            #if init_disp is not None:
            #    init_disp_up = torch.nn.functional.interpolate(init_disp, scale_factor=4, mode='bilinear', align_corners=True) * 4
            #    pred_hf_init, _ = self.wavelet_criterion(init_disp_up, masks=(M1, M2, M3))


                #loss_init = masked_charbonnier_loss(pred_hf_init, target_hf, mask)
                #disp_loss = loss_init
            
            def wavelet_crit(pred, target):
                # Target passed here is target_masked. We already computed target_hf and mask globally for this batch step
                # But to be safe and follow the interface:
                # actually target is the same psuedo_disparity_gt which is static
                pred_hf, _ = self.wavelet_criterion(pred, masks=(M1, M2, M3))
                # target_hf and mask are already computed above. Reuse them.
                return masked_charbonnier_loss(pred_hf, target_hf, mask)
            
            disp_loss = disp_loss +self.weighted_sequence_loss(predictions, target_masked, wavelet_crit, gamma=0.9)

        if self.curvature_loss_on:
            target_masked = psuedo_disparity_gt.unsqueeze(1)
            curv_mask = torch.ones_like(valid_regions_gt.unsqueeze(1).float())

            def curv_crit(pred, target):
                return self.curvature_criterion(pred, target, mask=curv_mask)

            curv_loss = self.weighted_sequence_loss(predictions, target_masked, curv_crit, gamma=0.9)
            disp_loss = disp_loss + self.curvature_loss_weight * curv_loss
            self.log('train/curvature_loss', curv_loss.detach(), prog_bar=False)

        if self.pairwise_ordinal_loss_on:
            target_masked = psuedo_disparity_gt.unsqueeze(1)
            pseudo_mask = (psuedo_disparity_gt > 0) & (psuedo_disparity_gt < 256)

            def pw_ordinal_crit(pred, target):
                return pairwise_ordinal_loss(
                    pred, target,
                    mask=pseudo_mask.unsqueeze(1),
                    n_pairs=self.pairwise_ordinal_n_pairs,
                    delta=self.pairwise_ordinal_delta,
                )

            pw_ord_loss = self.weighted_sequence_loss(predictions, target_masked, pw_ordinal_crit, gamma=0.9)
            disp_loss = disp_loss + self.pairwise_ordinal_loss_weight * pw_ord_loss
            self.log('train/pairwise_ordinal_loss', pw_ord_loss.detach(), prog_bar=False)

        return disp_loss
        # else:
        #    disp_loss= predictions[0].sum() * 0.0
        #return disp_loss


    def disp2depth(self,disp,baseline,fx,uv):
        valid_disp = (disp >= 1)
        disp = disp.clamp(min=1)
        depth = (baseline*fx)/(disp+1e-6)
        uv = uv*(depth.flatten(2))
        return uv,valid_disp

    def create_local_coordinate(self,H,W,device,dtype):
        y_range = torch.arange(H,device=device,dtype=dtype)
        x_range = torch.arange(W,device=device,dtype=dtype)
        yy,xx = torch.meshgrid(y_range,x_range,indexing="ij")
        grid = torch.stack([xx,yy,torch.ones_like(xx)],dim=-1) 
        return grid.unsqueeze(0)
    
    def soft_splat_rgb_to_thermal_batch(self,u_thr, v_thr, thr_disparity, thr_depth, H_thr, W_thr):
        """
        Batched Soft Splatting.
        Inputs are (B, N) or (B, H, W) flattened to (B, N)
        output: (B, H_thr, W_thr)
        """
        B = thr_depth.shape[0]
        dtype = thr_depth.dtype
        device = thr_depth.device
        
        u_thr = u_thr.view(B, -1)
        v_thr = v_thr.view(B, -1)
        thr_disparity = thr_disparity.view(B, -1)
        thr_depth = thr_depth.view(B, -1)
        
        # 1. Bounds Check
        # Keep points strictly inside the frame [0, W-1] x [0, H-1]
        mask = (u_thr >= 0) & (u_thr <= W_thr - 1.001) & \
            (v_thr >= 0) & (v_thr <= H_thr - 1.001) & (thr_depth > 0)
        
        # We need to process only valid points, but carrying the batch index is crucial
        # Create Batch Indices
        batch_ids = torch.arange(B, device=device).unsqueeze(1).repeat(1, u_thr.shape[1])
        
        # Apply Mask
        b_idx = batch_ids[mask]
        u = u_thr[mask]
        v = v_thr[mask]
        val = thr_depth[mask]
        disp = thr_disparity[mask]
        
        if u.numel() == 0:
            return torch.zeros((B, H_thr, W_thr), device=device, dtype=dtype)
        
        # 2. Compute 4 Neighbors
        u0 = torch.floor(u).long()
        v0 = torch.floor(v).long()
        u1 = u0 + 1
        v1 = v0 + 1
        
        # Clamp indices to be safe
        u0 = torch.clamp(u0, 0, W_thr - 1)
        v0 = torch.clamp(v0, 0, H_thr - 1)
        u1 = torch.clamp(u1, 0, W_thr - 1)
        v1 = torch.clamp(v1, 0, H_thr - 1)
        
        # 3. Bilinear Weights
        right_w = u - u0.float()
        left_w  = 1.0 - right_w
        bottom_w = v - v0.float()
        top_w    = 1.0 - bottom_w
        
        w_tl = left_w * top_w
        w_tr = right_w * top_w
        w_bl = left_w * bottom_w
        w_br = right_w * bottom_w
        
        # 4. Depth Importance (Soft Z-Buffer)
        beta = 0.1
        importance = torch.exp(beta * disp)
        
        w_tl = w_tl * importance
        w_tr = w_tr * importance
        w_bl = w_bl * importance
        w_br = w_br * importance
        
        # 5. Scatter Add
        # We flatten the whole batch output to (B * H_thr * W_thr)
        # Stride is H_thr * W_thr
        stride = H_thr * W_thr
        
        idx_base = b_idx * stride
        
        idx_tl = idx_base + v0 * W_thr + u0
        idx_tr = idx_base + v0 * W_thr + u1
        idx_bl = idx_base + v1 * W_thr + u0
        idx_br = idx_base + v1 * W_thr + u1
        
        all_indices = torch.cat([idx_tl, idx_tr, idx_bl, idx_br], dim=0) # (4N,)
        all_weights = torch.cat([w_tl, w_tr, w_bl, w_br], dim=0)         # (4N,)
        all_values  = val.repeat(4)                                      # (4N,)
        
        out_val = torch.zeros(B * H_thr * W_thr, device=device, dtype=dtype)
        out_w   = torch.zeros(B * H_thr * W_thr, device=device, dtype=dtype)
        
        # Weight the values
        weighted_values = all_values * all_weights
        
        out_val.index_add_(0, all_indices, weighted_values)
        out_w.index_add_(0, all_indices, all_weights)
        
        # 6. Normalize
        valid_pixels = out_w > 1e-6
        final_image = torch.zeros_like(out_val)
        final_image[valid_pixels] = out_val[valid_pixels] / out_w[valid_pixels]
        
        return final_image.reshape(B, 1,H_thr, W_thr),valid_pixels.reshape(B, 1,H_thr, W_thr)