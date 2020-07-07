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
    def __init__(self, n_stored, S_size, n_phi, S_dtype="uint8", save_type="lazy_frames", cache=True, save_location = None, resume = False):
        self.cache = cache
        self.save_type = save_type
        self.save_location = save_location
        self.counter = 0
        # We store an extra element to combine S and S_next array
        self.max_capacity = n_stored+1 
        self.S_size = S_size
        self.n_phi = n_phi
        self.oldest_index = 0
        out_size = (self.max_capacity,) + (self.S_size) + (self.n_phi,)      
        
        if (save_type=="lazy_frames"): # Use lazy frames to save memory
            self.S_array = np.zeros(self.max_capacity, dtype="object")
            
        elif (save_type=="disk"):
            # Check if replay exists
            if not resume:
                self.S_array = np.lib.format.open_memmap(self.save_location+"S_array1.npy", dtype=S_dtype, mode='w+', shape=out_size)
                
            if self.cache:
                # Cache lists for storing data until batch should be returned
                self.counter_cache = np.zeros(self.max_capacity) # For storing which elements are cahced
                self.counter_list = [] # For keeping track of cached elements
                # The rest is for storing the replays
                self.a_list = []
                self.r_list = []
                self.done_list = []
                self.S_list = []
                self.S_next_list = []
            
        self.a_array = np.zeros(n_stored, dtype ="int")
        self.r_array = np.zeros(n_stored, dtype = "double")
        self.done_array = np.zeros(n_stored, dtype = "bool")
        self.S_dtype = S_dtype
        
        if resume:
            self.load()
        
    def save_buffer(self):
        if self.cache:
            self.flush_cache() # flush latest changes
        destination = self.save_location + "replay_files"
        if (self.save_type == "disk"):
            np.savez(destination, self.a_array, self.r_array, self.done_array, self.oldest_index, self.counter)
        else:
            np.savez(destination, self.a_array, self.r_array, self.done_array, self.oldest_index, self.counter, self.S_array)
        
        
    def load(self):
        destination = self.save_location + "replay_files.npz"
        arrays = np.load(destination)
        self.a_array = arrays['arr_0']
        self.r_array = arrays['arr_1']
        self.done_array = arrays['arr_2']
        self.oldest_index = int(arrays['arr_3'])
        self.counter = int(arrays['arr_4'])
        if (self.save_type != "disk"):
            self.S_array = arrays['arr_5']
        else:
            self.S_array = np.load(self.save_location+"S_array1.npy", mmap_mode='r+')
            
            
    def get_oldest_index(self):
        index = self.oldest_index
        # Iterate oldest index
        self.oldest_index = (self.oldest_index+1) % self.max_capacity
        return index
        
    def flush_cache(self):
        ### Flush elements stored in cache into harddrive stored array
        # Check if cache is empty
        if len(self.counter_list) == 0:
            return

        # Insert stored elements into correct arrays
        self.S_array[self.counter_list], self.S_array[np.array(self.counter_list)+1] = self.S_list, self.S_next_list
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
            assert (len(self.counter_list) > 10**5)
            return True
        
        return (sum(self.counter_cache[indexes])  >  0)
    
    def cache_replay(self, replay, index):
        ### Adds replay to cache
        
        # note what element has been cached
        self.counter_cache[index] = 1
        self.counter_list.append(index) # this is for resetting counter_cache
        
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
            index = self.get_oldest_index()
            
        ## Save replay to disk
        if ((self.save_type=="disk") & (self.cache)):
            # Instead cache replay
            self.cache_replay(replay, index)
            return
        
        # This is a one-line, since it is stored on a disk
        self.S_array[index], self.S_array[index+1] = replay[0], replay[3]
        self.a_array[index] = replay[1]
        self.r_array[index] = replay[2]
        self.done_array[index] = replay[4]
    
    def return_batch(self, batch_size):
        # Get index for batch
        indexes = np.random.randint(self.counter, size=batch_size)
        
        if ((self.save_type=="disk") & (self.cache)):
            # Check if cache should be flushed
            if self.should_flush(indexes):
                self.flush_cache()
        
        # Define arrays for return
        S_array = self.S_array[indexes]
        S_next_array = self.S_array[indexes+1]
        
        if self.save_type == "lazy_frames": # If lazy frames, convert to numpy and permute
            S_array = np.stack(S_array, axis=0)
            S_next_array = np.stack(S_next_array, axis=0)
        
        S_array = np.transpose(S_array, (0, 3, 1, 2))
        S_next_array = np.transpose(S_next_array, (0, 3, 1, 2))    
        a_array = self.a_array[indexes]
        r_array = self.r_array[indexes]
        done_array = self.done_array[indexes]
        return S_array, a_array, r_array, S_next_array, done_array