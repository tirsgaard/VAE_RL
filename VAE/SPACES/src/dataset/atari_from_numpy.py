from torch.utils.data import Dataset
import os.path as osp
from torchvision import transforms
from skimage import io
import os
import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class atari_from_numpy(Dataset):
    def __init__(self, root, mode, start_seq_len, end_seq_len, increase_seq):
        # checkpointdir = os.checkpointdir.join(root, mode)
        assert mode in ['train', 'val', 'test']
        self.root = root
        self.mode = mode
        self.seq_start_len = start_seq_len
        self.seq_end_len = end_seq_len
        self.increase_seq = increase_seq
        self.cur_seq_len = self.seq_start_len
        self.global_step = 1
        assert os.path.exists(root), 'Path {} does not exist'.format(root)
        
        
        self.array_paths = []
        array_dir = os.path.join(self.root, mode)
        for file in os.scandir(array_dir):
            array_dir = file.path
            if 'npy' in array_dir:
                self.array_paths.append(array_dir)
            else:
                print("No .npy file found")
        
        
        self.dataset_array = np.load(self.array_paths[0], allow_pickle=True) # Load the dataset into memmory
        
        # Also construct index array, each element is a list consisting of two elements,
        #  which sequence it is stored in, and the range of the sequence
        self.index_array = np.empty(self.__len__(), dtype="object")
        i = 0
        for j in range(self.dataset_array.shape[0]):
            for k in range(self.dataset_array[j].shape[0]-self.seq_end_len):
                self.index_array[i] = [j,k]
                i += 1
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
    @property
    def bb_path(self):
        path = osp.join(self.root, self.mode, 'bb')
        assert osp.exists(path), f'Bounding box path {path} does not exist.'
        return path
    
    def __getitem__(self, index_global):
        self.global_step += 1 # This is for keeping track of when to increase sequence length
        seq_index, index_local = self.index_array[index_global]
        self.check_seq_len()
        imgs = self.dataset_array[seq_index][index_local:(index_local+self.cur_seq_len)] # (L, H, W, 3)
        img_tensors = []
        for i in range(imgs.shape[0]):
            img_tensors.append(self.transform(imgs[i]))
        imgs = torch.stack(img_tensors, 0)
        return imgs
    
    def check_seq_len(self):
        self.cur_seq_len = min(self.seq_start_len + int(np.floor(self.global_step/self.increase_seq)), self.seq_end_len)
    
    def __len__(self):
        possible_seq = 0
        for i in range(self.dataset_array.shape[0]):
            possible_seq += self.dataset_array[i].shape[0]-self.seq_end_len # do not add seqeunces to short
        
        return possible_seq
    
    def set_seq_length(self, seq_length):
        # Select another sequence length
        
        # We just have to reconstruct mapping array
        self.seq_length = seq_length
        self.index_array = np.empty(self.__len__(), dtype="object")
        i = 0
        for j in range(self.dataset_array.shape[0]):
            for k in range(self.dataset_array[j].shape[0]-self.seq_length):
                self.index_array[i] = [j,[range(k,k+self.seq_length)]]
                i += 1
        
    

