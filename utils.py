import numpy as np
import random

class ExperienceBuffer():
    def __init__(self, buffer_size = 200):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self, experience):
        if len(list(self.buffer)) + len(list(experience)) >= self.buffer_size:
            self.buffer[0:(len(list(experience))+len(list(self.buffer)))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return random.sample(self.buffer, size)
    
def get_f(m, offsets):
    f = np.zeros([len(m), m.shape[1], len(offsets)])
    for i, offset in enumerate(offsets):
        f[:-offset, :, i] = m[offset:, :] - m[:-offset, :]
        if i > 0:
            f[-offset:, :, i] = f[-offset:, :, i-1]
    return f

def one_hot(tensor, size):
    return np.eye(size)[tensor]