import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict
from .arch import arch
from .fg import SpacesFg
from .bg import SpaceBg

def log_stacker(logs_stacked):
    """
    Function for stacking the sequences of logs generated from the fg model
    Input: 
        logs_stacked: A list of dictionaries. Each entry should be of form (B,..)
    
    Output:
        new_logs: A dictionary with the entries stacked in a sequence.
                      The resulting dimensions of each entry is (B, L,..)
    """
    new_logs = {}
    for element in logs_stacked:
        for key in element:
            try:
                new_logs[key].append(element[key]) # Try adding key
                
            except:
                new_logs[key] = [element[key]]
            
    # Now for each element in dictionary stack
    for element in new_logs:
        new_logs[element] = torch.stack(new_logs[element], dim=1)
    
    return new_logs

class Spaces(nn.Module):
    
    def __init__(self, cfg):
        nn.Module.__init__(self)
        
        self.fg_module = SpacesFg()
        self.bg_module = SpaceBg()
        if 'cuda' in cfg.device:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        
    def get_h_0(self, sizes):
        h_prev = {}
        h_scale = torch.zeros(1, sizes["B"] , sizes["G"]**2)
        h_shift = torch.zeros(1, sizes["B"] , sizes["G"]**2)
        h_pres = torch.zeros(1, sizes["B"] , sizes["G"]**2)
        h_depth = torch.zeros(1, sizes["B"] , sizes["G"]**2)
        h_start_what = torch.zeros(sizes["B"]*sizes["G"]**2, arch.recurrent_dim ,1,1)
        
        h_prev["h_pres"], h_prev["h_depth"], h_prev["h_scale"], h_prev["h_shift"], h_prev["h_what"] = h_pres, h_depth, h_scale, h_shift, h_start_what
        return h_prev
        
    def forward(self, x, global_step):
        """
        Inference.
        
        :param x: (B, L, 3, H, W)
        :param global_step: global training step
        :return:
            loss: a scalor. Note it will be better to return (B*L,)
            log: a dictionary for visualization
        """
        # For easy reshaping
        sizes = {"B": x.shape[0],
                 "L": x.shape[1],
                 "H": x.shape[3],
                 "W": x.shape[4],
                 "G": arch.G}
        
        # Background extraction
        # Background is not sequentially encoded, so Batch size and sequence length can be combined
        # (B*L, 3, H, W), (B*L, 3, H, W), (B*L,)
        bg_likelihood, bg, kl_bg, log_bg = self.bg_module(x.view([-1, 3, sizes["H"], sizes["W"]]), global_step)
        # Divide batch size and sequence length again
        bg_likelihood = bg_likelihood.view([sizes["B"], sizes["L"], 3, sizes["H"], sizes["W"]])
        try:
            bg = bg.view([sizes["B"], sizes["L"], 3, sizes["H"], sizes["W"]])
        except:
                print(x.shape)
                print(bg.shape)
                print(x.view([-1, 3, sizes["H"], sizes["W"]]).shape)
        kl_bg = kl_bg.view([sizes["B"], sizes["L"]])
        
        # Foreground extraction
        # This has to be sequentially
        # Predefine first passed on information
        h_prev = get_h_0(sizes)
        
        fg_likelihood = torch.zeros(sizes["B"], sizes["L"], 3, sizes["H"], sizes["W"])
        fg            = torch.zeros(sizes["B"], sizes["L"], 3, sizes["H"], sizes["W"])
        alpha_map     = torch.zeros(sizes["B"], sizes["L"], 1, sizes["H"], sizes["W"])
        kl_fg         = torch.zeros(sizes["B"], sizes["L"])
        loss_boundary = torch.zeros(sizes["B"], sizes["L"])
        log_fg = []
        
        # Now run for each element in sequence
        for l in range(sizes["L"]):
            log_fg.append(None) # This is such a stupid way of adding function returns to list
            fg_likelihood[:,l], fg[:,l], alpha_map[:,l], kl_fg[:,l], loss_boundary[:,l], log_fg[-1], h_prev = self.fg_module(x[:,l], global_step, h_prev)
        log_fg = log_stacker(log_fg)
        
        # Fix alpha trick
        if global_step and global_step < arch.fix_alpha_steps:
            alpha_map = torch.full_like(alpha_map, arch.fix_alpha_value)
            
        # Compute final mixture likelhood
        # (B, L, 3, H, W)
        fg_likelihood = (fg_likelihood + (alpha_map + 1e-5).log())
        bg_likelihood = (bg_likelihood + (1 - alpha_map + 1e-5).log())
        # (B, L, 2, 3, H, W)
        log_like = torch.stack((fg_likelihood, bg_likelihood), dim=2)
        # (B, L, 3, H, W)
        log_like = torch.logsumexp(log_like, dim=2)
        # (B, L,)
        log_like = log_like.flatten(start_dim=2).sum(2)

        # Take mean as reconstruction # (L, B)
        y = alpha_map * fg + (1.0 - alpha_map) * bg
        
        # Elbo
        elbo_seq = log_like - kl_bg - kl_fg
        # Combine sequences by sum since in log space
        elbo = elbo_seq.mean(1)
        loss_boundary = loss_boundary.sum(1)
        # Mean over batch
        loss = (-elbo + loss_boundary).mean()
        
        log = {
            'imgs': x,
            'y': y,
            # (B,)
            'mse': ((y-x)**2).flatten(start_dim=2).sum(dim=2),
            'log_like': log_like
        }
        log.update(log_fg)
        log.update(log_bg)
        
        return loss, log
    
    
    
    
    def inference(self, x, h_prev, reset = False):
        """
        Inference
        
        :param x: (B, 3, H, W)
        :param global_step: global training step
        :return:
            loss: a scalor. Note it will be better to return (B*L,)
            log: a dictionary for visualization
        """
        # For easy reshaping
        sizes = {"B": x.shape[0],
                 "L": x.shape[1],
                 "H": x.shape[3],
                 "W": x.shape[4],
                 "G": arch.G}
        
        # Background extraction
        # Background is not sequentially encoded, so Batch size and sequence length can be combined
        # (B*L, 3, H, W), (B*L, 3, H, W), (B*L,)
        global_step = 100000
        z_mask_loc, z_mask_scale, z_comp_loc_reshape, z_comp_scale_reshape = self.bg_module.inference(x.view([-1, 3, sizes["H"], sizes["W"]]), global_step)
        # Divide batch size and sequence length again
        #bg = bg.view([sizes["B"], sizes["L"], 3, sizes["H"], sizes["W"]])
        
        # Foreground extraction
        # This has to be sequentially
        # Predefine first passed on information
        if reset:
            h_prev = self.get_h_0(sizes)
        
        # Now run for each element in sequence
        Z_infs = []
        h_s = []
        for i in range(x.shape[1]):
            Z_infer, h, h_prev = self.fg_module.inference(x[:,i], h_prev)
            Z_infs.append(Z_infer)
            h_s.append(h)
            
        
        Z_infer = torch.cat(list(Z_infer.values()),dim=2)
        h_s = torch.cat(h_s,dim=2)
        background = torch.cat([z_mask_loc, z_mask_scale, z_comp_loc_reshape, z_comp_scale_reshape], dim = 0)
        background = background.permute(1,0).unsqueeze(0)
        # Combine different inputs
        Z_infer = torch.cat([Z_infer, h_s, background], dim = 2)
        return Z_infer, h_prev
