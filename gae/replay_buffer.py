import random
import numpy as np 

class ReplayBuffer(object):

    def __init__(self, max_size):
        self.max_size = max_size
        self.cur_size = 0
        self.buffer = {}
        self.init_length = 0

    def __len__(self):
        return self.cur_size

    def seed_buffer(self, episodes):
        self.init_length = len(episodes)
        self.add(episodes, np.ones(self.init_length))
    
    def add (self, episodes, *args):
        # add episodes to buffer 
        idx = 0
        while self.cur_size < self.max_size and idx < len(episodes):
            self.buffer[self.cur_size] = episodes[idx]
            idx += 1
            self.cur_size += 1
        if idx < len(episodes):
            remove_idxs = self.remove_n(len(episodes)-idx)
            for remove)idx in remove_idxs:
                self.buffer[remove_idx] = episodes[idx]
                idx += 1
        assert len(self.buffer) == self.cur_size

    def remove_n(self, n):
        # return n items for removal -- naive random removal
        idxs = random.sample(xrange(self.init_length, self,cur_size), n)
        return idxs

    def get_batch(self, n):
        # return a batch of n episodes to train 
        idxs = random.sample(xrange(self.cur_size), n)
        return [self.buffer[idx] for idx in idxs], None 
    
    def update_last_batch(self, delta):
        pass

