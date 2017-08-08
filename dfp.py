import sys
try:
    sys.path.remove('/home/dai/.local/lib/python3.6/site-packages')
except:
    pass

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DFP(nn.Module):
    def __init__(self, action_size, observation_shape,
                 num_measurements, num_offsets):
        super(DFP, self).__init__()
        
        self.observation_shape = observation_shape
        self.num_measurements = num_measurements
        self.action_size = action_size
        self.num_goals = num_measurements
        self.num_offsets = num_offsets
        
        self.h_o = nn.Linear(int(np.prod(observation_shape)), 128)
        self.h_m = nn.Linear(num_measurements, 64)
        self.h_g = nn.Linear(num_measurements, 64)
        self.h = nn.Linear(128 + 64 + 64, 256)
        
        # Calculate separate expectations and advantage stream
        self.h_expectation = nn.Linear(256, action_size * self.num_offsets * self.num_measurements)
        self.h_advantages = nn.Linear(256, action_size * self.num_offsets * self.num_measurements)
        

    def forward(self, observation, measurement, goals, temp):
        observation_flatten = observation.view(observation.size()[0], -1)
        h_o = F.elu(self.h_o(observation_flatten))
        h_m = F.elu(self.h_m(measurement))
        h_g = F.elu(self.h_g(goals))
        
        h = torch.cat([h_o, h_m, h_g], dim=1)
        h_ = F.elu(self.h(h))
        
        expectations = self.h_expectation(h_)
        advantages = self.h_advantages(h_)
        advantages = advantages - advantages.mean(1).repeat(1, advantages.size()[1])
        
        predictions = expectations + advantages
        predictions = predictions.view(-1, self.num_measurements, self.action_size, self.num_offsets)
        
        boltzman = F.softmax(predictions.sum(3) / temp)
        # boltzman = boltzman.squeeze(3)

        return boltzman, predictions

        '''
        b = torch.mm(goals, boltzman[0])
        c = b.sum(1)
        action = torch.multinomial(c, 1)
        return boltzman, predictions, action
        '''