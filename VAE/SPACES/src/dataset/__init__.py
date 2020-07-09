from .atari import Atari
from .obj3d import Obj3D
from .own_atari import own_atari
from .atari_from_numpy import atari_from_numpy
from torch.utils.data import DataLoader
import torch

__all__ = ['get_dataset', 'get_dataloader']

def get_dataset(cfg, mode):
    assert mode in ['train', 'val', 'test']
    if cfg.dataset == 'ATARI':
        mode = 'validation' if mode == 'val' else mode
        return Atari(cfg.dataset_roots.ATARI, mode, gamelist=cfg.gamelist)
    elif cfg.dataset == 'OBJ3D_SMALL':
        return Obj3D(cfg.dataset_roots.OBJ3D_SMALL, mode)
    elif cfg.dataset == 'OBJ3D_LARGE':
        return Obj3D(cfg.dataset_roots.OBJ3D_LARGE, mode)
    elif cfg.dataset == 'Space_invaders':
        return own_atari(cfg.dataset_roots.Space_invaders, mode)
    elif cfg.dataset == 'Riverraid_seq':
        return atari_from_numpy(cfg.dataset_roots.Riverraid_seq, mode, cfg.train.start_seq_length, cfg.train.end_seq_length, cfg.train.increase_seq)
    elif cfg.dataset == 'Riverraid_seq':
        return atari_from_numpy(cfg.dataset_roots.Riverraid_seq, mode, cfg.train.start_seq_length, cfg.train.end_seq_length, cfg.train.increase_seq)

def get_dataloader(cfg, mode):
    assert mode in ['train', 'val', 'test']
    
    batch_size = getattr(cfg, mode).batch_size
    shuffle = True if mode == 'train' else False
    num_workers = getattr(cfg, mode).num_workers
    
    dataset = get_dataset(cfg, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn = custom_collate)
    return dataloader, dataset
    

def custom_collate(batched_seq):
    
    elem = batched_seq[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
            
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batched_seq[0]
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
    
    try:
        batched_seq = torch.stack(batched_seq, 0, out = out)
        return batched_seq
    except:
        # This is the case of different sequence lengths
        # We will just cut the sequences to have the same length as the shortest
        min_length = min(seq.shape[0] for seq in batched_seq)
        batched_seq = [seq[0:min_length] for seq in batched_seq]
        batched_seq = torch.stack(batched_seq, 0, out = out)
        return batched_seq