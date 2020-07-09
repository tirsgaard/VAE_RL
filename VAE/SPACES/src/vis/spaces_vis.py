from torch.utils.tensorboard import SummaryWriter
import imageio
import numpy as np
import torch
from utils import spatial_transform
from .utils import bbox_in_one
from attrdict import AttrDict
from torchvision.utils import make_grid
from torch.utils.data import Subset, DataLoader


class SpacesVis:
    def __init__(self):
        pass
    
    
    @torch.no_grad()
    def train_vis(self, writer: SummaryWriter, log, global_step, mode, num_batch=10):
        """
        """
        
        B = num_batch
        
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                log[key] = value.detach().cpu()
                if isinstance(log[key], torch.Tensor) and log[key].ndim > 0:
                    log[key] = log[key][:num_batch]
        log = AttrDict(log)
        # Only include first batch element
        # (L, 3, H, W)
        fg_box = bbox_in_one(
            log.fg[0], log.z_pres[0], log.z_scale[0], log.z_shift[0]
        )
        sizes = {"B": log.fg.shape[0],
                 "L": log.fg.shape[1],
                 "H": log.fg.shape[3],
                 "W": log.fg.shape[4],
                 "K": log.comps.shape[1]}
        
        # (L, 1, 3, H, W)
        # We only look at first image in batch
        imgs = log.imgs[0, :, None]
        fg = log.fg[0, :, None]
        recon = log.y[0, :, None]
        fg_box = fg_box[:, None]
        bg = log.bg.view([sizes["B"], sizes["L"], sizes["K"], sizes["H"], sizes["W"]])[0, :, None]
        # (L, K, 3, H, W)
        comps = log.comps.view([sizes["B"], sizes["L"], sizes["K"], 3, sizes["H"], sizes["W"]])[0]
        # (L, K, 3, H, W)
        masks = log.masks.view([sizes["B"], sizes["L"], sizes["K"], 1, sizes["H"], sizes["W"]])[0].expand_as(comps)
        masked_comps = comps * masks
        alpha_map = log.alpha_map[0,:, None].expand_as(imgs)
        grid = torch.cat([imgs, recon, fg, fg_box, bg, masked_comps, masks, comps, alpha_map], dim=1)
        nrow = grid.size(1)
        L, N, _, H, W = grid.size()
        grid = grid.view(L*N, 3, H, W)
        
        grid_image = make_grid(grid, nrow, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/#0-separations', grid_image, global_step)
        
        grid_image = make_grid(log.imgs[0,:], 5, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/1-image', grid_image, global_step)
        
        grid_image = make_grid(log.y[0,:], 5, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/2-reconstruction_overall', grid_image, global_step)
        
        grid_image = make_grid(log.bg[0,:], 5, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/3-background', grid_image, global_step)
        
        mse = (log.y[0,:] - log.imgs[0,:]) ** 2
        mse = mse.flatten(start_dim=1).sum(dim=1)
        mse_diff = mse[0]-mse[-1] # Negative if better
        log_like, kl_z_what, kl_z_where, kl_z_pres, kl_z_depth, kl_bg = (
            log['log_like'].mean(), log['kl_z_what'].mean(), log['kl_z_where'].mean(),
            log['kl_z_pres'].mean(), log['kl_z_depth'].mean(), log['kl_bg'].mean()
        )
        # Store differences, to check for difference, when using recurrent structure
        log_like_diff = log['log_like'][0,0]-log['log_like'][0,-1]
        kl_z_what_diff = log['kl_z_what'][0,0]-log['kl_z_what'][0,-1]
        kl_z_where_diff = log['kl_z_where'][0,0]-log['kl_z_where'][0,-1]
        kl_z_pres_diff = log['kl_z_pres'][0,0]-log['kl_z_pres'][0,-1]
        kl_z_depth_diff = log['kl_z_depth'][0,0]-log['kl_z_depth'][0,-1]
        kl_bg_diff = log['kl_bg'].view([sizes["B"], sizes["L"]])[0,0]-log['kl_bg'].view([sizes["B"], sizes["L"]])[0,-1]
        
        loss_boundary = log.boundary_loss.mean()
        loss = log.loss.mean()
        
        count = log.z_pres.flatten(start_dim=1).sum(dim=1).mean(dim=0)
        writer.add_scalar(f'{mode}/mse', mse.mean(dim=0).item(), global_step=global_step)
        writer.add_scalar(f'{mode}/loss', loss, global_step=global_step)
        writer.add_scalar(f'{mode}/count', count, global_step=global_step)
        writer.add_scalar(f'{mode}/log_like', log_like.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/loss_boundary', loss_boundary.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/What_KL', kl_z_what.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/Where_KL', kl_z_where.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/Pres_KL', kl_z_pres.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/Depth_KL', kl_z_depth.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/Bg_KL', kl_bg.item(), global_step=global_step)
        
        # Also add differences statistics
        writer.add_scalar(f'{mode}/differences/mse', mse_diff.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/differences/log_like', log_like_diff.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/differences/What_KL', kl_z_what_diff.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/differences/Where_KL', kl_z_where_diff.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/differences/Pres_KL', kl_z_pres_diff.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/differences/Depth_KL', kl_z_depth_diff.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/differences/Bg_KL', kl_bg_diff.item(), global_step=global_step)
    @torch.no_grad()
    def show_vis(self, model, dataset, indices, path, device):
        dataset = Subset(dataset, indices)
        dataloader = DataLoader(dataset, batch_size=len(indices), shuffle=False)
        data = next(iter(dataloader))
        data = data.to(device)
        loss, log = model(data, 100000000)
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                log[key] = value.detach().cpu()
        log = AttrDict(log)
        # (B, 3, H, W)
        fg_box = bbox_in_one(
            log.fg, log.z_pres, log.z_scale, log.z_shift
        )
        # (L, 1, 3, H, W)
        imgs = log.imgs[0, :, None]
        fg = log.fg[0, :, None]
        recon = log.y[0, :, None]
        fg_box = fg_box[0, :, None]
        bg = log.bg[0, :, None]
        # (L, K, 3, H, W)
        comps = log.comps[0,:]
        # (L, K, 3, H, W)
        masks = log.masks.expand_as(comps)
        masked_comps = comps * masks
        alpha_map = log.alpha_map[0, :, None].expand_as(imgs)
        grid = torch.cat([imgs, recon,  fg, fg_box, bg, masked_comps, masks, comps, alpha_map], dim=1)
        nrow = grid.size(1)
        L, N, _, H, W = grid.size()
        grid = grid.view(L*N, 3, H, W)
        
        # (3, H, W)
        grid_image = make_grid(grid, nrow, normalize=False, pad_value=1)
        
        # (H, W, 3)
        image = torch.clamp(grid_image, 0.0, 1.0)
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        imageio.imwrite(path, image)
