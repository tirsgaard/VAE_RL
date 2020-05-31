import numpy as np
import torch
import torchvision.transforms

from tempfile import mkdtemp
import os.path as path


def phi_transformer(S, n_phi, im_size=[84,84], n_channels = 1):
    ### Function for transforming images
    # S is the batch of input images, and n_phi is the number of stacked images
    if (n_channels == 1):
        pre_process = torchvision.transforms.Compose(
            [torchvision.transforms.ToPILImage(),
             torchvision.transforms.Grayscale(num_output_channels=1),
             torchvision.transforms.Resize(im_size),
             torchvision.transforms.ToTensor()])
    elif (n_channels == 3):
        pre_process = torchvision.transforms.Compose(
            [torchvision.transforms.ToPILImage(),
             torchvision.transforms.Resize(im_size),
             torchvision.transforms.ToTensor()])

    lists = []
    if (n_phi==1):
        lists = [pre_process(S)]
    else:
        for i in range(n_phi):
            lists.append(pre_process(S[i]))
        
    S = torch.stack(lists)
    S = torch.squeeze(S)
    S = torch.unsqueeze(S, dim=0)
    S = S.numpy()
    S = (255*S).astype("uint8")
    return S





class Replay_buffer:
    # Class for storing replay information
    # Beacuse of atari wrapper we store in tensorflow formaet i.e (H,W,K)
    def __init__(self, n_stored, S_size, n_phi, S_dtype="uint8", save_type="lazy_frames"):
        self.save_type = save_type
        self.counter = 0
        self.max_capacity = n_stored
        self.S_size = S_size
        self.n_phi = n_phi
        out_size = (self.max_capacity,) + (self.S_size) + (self.n_phi,)      
        
        if (save_type=="lazy_frames"): # Use lazy frames to save memory
            self.S_array = np.zeros(self.max_capacity, dtype="object")
            self.S_array_next = np.zeros(self.max_capacity, dtype="object")
            
        elif (save_type=="disk"):
            filename1 = '../../cache_hdd/newfile3.dat' # path.join(mkdtemp(), '~/cache_hdd/newfile1.dat')
            filename2 = '../../cache_hdd/newfile4.dat' #path.join(mkdtemp(), '~/cache_hdd/newfile2.dat')
            self.S_array = np.memmap(filename1, dtype=S_dtype, mode='w+', shape=out_size)
            self.S_array_next = np.memmap(filename2, dtype=S_dtype, mode='w+', shape=out_size)
            
            # Cache lists for storing data until batch should be returned
            self.counter_cache = np.zeros(self.max_capacity)
            self.counter_list = []
            self.a_list = []
            self.r_list = []
            self.done_list = []
            self.S_list = []
            self.S_next_list = []
            
        else:
            self.S_array = np.zeros(out_size, dtype=S_dtype)
            self.S_array_next = np.zeros(out_size, dtype=S_dtype)
            
        self.a_array = np.zeros(n_stored, dtype="int")
        self.r_array = np.zeros(n_stored, dtype = "double")
        self.done_array = np.zeros(n_stored, dtype = "bool")
        self.S_dtype = S_dtype
        
        
    def flush_cache(self):
        ### Flush elements stored in cache into harddrive stored array
        
        # Insert stored elements into correct arrays
        self.S_array[self.counter_list], self.S_array_next[self.counter_list] = self.S_list, self.S_next_list
        self.a_array[self.counter_list] = self.a_list
        self.r_array[self.counter_list] = self.r_list
        self.done_array[self.counter_list] = self.done_list
        
        # Reset lists
        self.counter_cache[self.counter_list] = 0
        self.counter_list = []
        self.a_list = []
        self.r_list = []
        self.done_list = []
        self.S_list = []
        self.S_next_list = []
        
    def should_flush(self, indexes):
        ### Checks if any of the indexed elements is in cached array
        ###    From my calculations, this should on average happen every
        ###    ~110 parameter update for a cache size of 10**6, after replay arrays are filled
        if (len(self.counter_list) > 10**5):
            # This should NEVER happen in reality, chance is too low
            return True
        
        return (sum(self.counter_cache[indexes])  >  0)
    
    def cache_replay(self, replay, index):
        ### Adds replay to cache
        
        # note what element has been cached
        self.counter_cache[index] = 1
        self.counter_list.append(index)
        
        # Add replay to caches
        self.a_list.append(replay[1])
        self.r_list.append(replay[2])
        self.done_list.append(replay[4])
        self.S_list.append(replay[0])
        self.S_next_list.append(replay[3])
        
        
    def add_replay(self, replay):
        # Input is S, a, r, S_next, done
        if (self.max_capacity>self.counter):
            # Case where added there is room for replays
            index = self.counter
            self.counter += 1
        else:
            # Case where a replay has to be supstituted
            index = np.random.randint(self.counter)
            
        ## Save replay to disk
        if (self.save_type=="disk"):
            # Instead cache replay
            self.cache_replay(replay, index)
            return
            
        self.S_array[index], self.S_array_next[index] = replay[0], replay[3]
        
        
        self.a_array[index] = replay[1]
        self.r_array[index] = replay[2]
        self.done_array[index] = replay[4]
    
    def return_batch(self, batch_size):
        # Get index for batch
        indexes = np.random.randint(self.counter, size=batch_size)
        
        if (self.save_type=="disk"):
            # Check if cache should be flushed
            if self.should_flush(indexes):
                self.flush_cache()
        
        # Define arrays for return
        S_array = self.S_array[indexes]
        S_next_array = self.S_array_next[indexes]
        
        if self.save_type == "lazy_frames": # If lazy frames, convert to numpy and permute
            S_array = np.stack(S_array, axis=0)
            S_next_array = np.stack(S_next_array, axis=0)
        
        S_array = np.transpose(S_array, (0, 3, 1, 2))
        S_next_array = np.transpose(S_next_array, (0, 3, 1, 2))    
        a_array = self.a_array[indexes]
        r_array = self.r_array[indexes]
        done_array = self.done_array[indexes]
        return S_array, a_array, r_array, S_next_array, done_array
