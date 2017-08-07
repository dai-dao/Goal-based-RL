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
        
        # Training
        self.loss_fn = nn.MSELoss()

    def forward(self, observation, measurement, goals, temp):
        observation_flatten = observation.view(observation.size()[0], -1)
        h_o = F.elu(self.h_o(observation_flatten))
        h_m = F.elu(self.h_m(measurement))
        h_g = F.elu(self.h_g(goals))
        
        h = torch.cat([h_o, h_m, h_g], dim=1)
        h_ = F.elu(self.h(h))
        
        expectations = self.h_expectation(h_)
        advantages = self.h_advantages(h_)
        advantages = advantages - advantages.mean(1).expand_as(advantages)
        
        predictions = expectations + advantages
        predictions = predictions.view(-1, self.num_measurements, self.action_size, self.num_offsets)
        
        boltzman = F.softmax(predictions.mean(3) / temp)
        boltzman = boltzman.squeeze(3)

        b = torch.mm(goals, boltzman[0])
        c = b.sum(1)
        action = torch.multinomial(c, 1)

        return boltzman, predictions, action
    
    def compute_loss(self, observation, measurement, goals,
                     temp, action_onehot, target, optimizer):
        boltzman, predictions, _ = self(observation, measurement, goals, temp)
        action_resize = action_onehot.view(-1, 1, self.action_size, 1)
        action_resize = action_resize.repeat(1, 2, 1, 6)
        # Select the predictions relevant to the chosen actions
        pred_action = (predictions * action_resize).sum(2)
        
        loss = self.loss_fn(pred_action, target)
        entropy = -(boltzman * torch.log(boltzman + 1e-7)).sum()
        total_loss = loss # + entropy
        
        # Backward and optimize step
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 40.0)
        optimizer.step()

        return loss, entropy