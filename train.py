import sys
try:
    sys.path.remove('/home/dai/.local/lib/python3.6/site-packages')
except:
    pass

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from gridworld_goals import *
from utils import *
from dfp import DFP
from collections import namedtuple

Experience = namedtuple('Experience',
                        ('observation', 'action', 'measurement', 'goal', 'target'))

class Trainer():
    def __init__(self, rank, args, offsets, a_size, model):
        self.rank = rank
        self.exp_buff = ExperienceBuffer()
        self.env = gameEnv(partial=False, size=5)
        self.offsets = offsets
        self.a_size = a_size
        self.args = args
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

    def loss_fn(self, predict, target):
        return torch.pow(predict - target, 2).sum()

    def work(self):
        torch.manual_seed(self.args.seed + self.rank)
        self.model.train()

        episode_deliveries = []
        episode_lengths = []

        for episode_index in range(self.args.num_episodes):
            episode_buffer = []
            episode_frames = []
            d = False
            t = 0
            temp = 0.25 #How spread out we want our action distribution to be

            s, o_big, m, g, h = self.env.reset()
            current_goal = None

            while d is False:
                if m[1] <= .3:
                    current_goal = np.array([0.0,1.0])
                else:
                    current_goal = np.array([1.0,0.0])

                # Convert to Variable
                m = np.array(m)
                s_var = Variable(torch.from_numpy(s)).float().unsqueeze(0)
                m_var = Variable(torch.from_numpy(m)).float().unsqueeze(0)
                g_var = Variable(torch.from_numpy(current_goal)).float().unsqueeze(0)

                # Compute action probabilities
                boltzman, _ = self.model(s_var, m_var, g_var, temp)
                b = current_goal * boltzman.data.numpy()[0].T
                c = np.sum(b, 1)
                c /= c.sum()
                a = np.random.choice(c, p=c)
                a = np.argmax(c == a)

                # Add to episode buffer
                episode_buffer.append(Experience(s, a, m, current_goal, None))

                # Perform the action on the environment
                s1,s1_big,m1,g1,h1,d = self.env.step(a) 
                t += 1
                s = s1
                m = m1
                g = g1
                h = h1

                # End the episode after num steps
                if t > self.args.num_step:
                    d = True

            # Training statistics
            episode_deliveries.append(m[0])
            episode_lengths.append(t)

            # Update the network using experience buffer at the end
            # of the episode
            loss, entropy = self.train(episode_buffer)
            if loss != 0 and self.rank == 0 and episode_index % 50 == 0:
                print('Episode {} | Loss {} | Episode length {} | Deliveries {}'.
                        format(episode_index, loss, t, np.mean(episode_deliveries[-50:])))

            if episode_index % 1000 == 0 and self.rank == 0:
                torch.save(self.model.state_dict, 'models/model-weights.pth')

    def train(self, rollout):
        # rollout is a list of Experience
        batch = Experience(*zip(*rollout))
        measurements = np.array(batch.measurement)
        targets = np.array(get_f(measurements, self.offsets))

        new_rollout = []
        for index in range(len(rollout)):
            new_exp = Experience(batch.observation[index], batch.action[index],
                                 batch.measurement[index], batch.goal[index], targets[index])
            new_rollout.append(new_exp)
        self.exp_buff.add(new_rollout)

        # Get a batch of experiences from the buffer and 
        # use them to update the global network
        if len(self.exp_buff.buffer) > self.args.batch_size:
            exp_batch = self.exp_buff.sample(self.args.batch_size)
            batch = Experience(*zip(*exp_batch))

            observation_batch = np.array(batch.observation)
            measurement_batch = np.array(batch.measurement)
            temperature = 0.1
            action_batch = np.array(batch.action)
            target_batch= np.array(batch.target)
            goal_batch = np.array(batch.goal)

            # Convert action to one-hot vector
            action_onehot = one_hot(action_batch, self.a_size)

            # Convert to variables
            obs_var = Variable(torch.from_numpy(observation_batch)).float()
            mea_var = Variable(torch.from_numpy(measurement_batch)).float()
            goa_var = Variable(torch.from_numpy(goal_batch)).float()
            act_var = Variable(torch.from_numpy(action_onehot)).float()
            tar_var = Variable(torch.from_numpy(target_batch)).float()
            
            loss, entropy = self.compute_loss(obs_var, mea_var,
                                                    goa_var, temperature, 
                                                    act_var, tar_var)
            return loss.data.numpy()[0] / len(rollout), entropy / len(rollout)
        else:
            return 0, 0

    def compute_loss(self, observation, measurement, goals,
                     temp, action_onehot, target):
        boltzman, predictions = self.model(observation, measurement, goals, temp)
        action_resize = action_onehot.view(-1, 1, self.a_size, 1)
        action_resize = action_resize.repeat(1, 2, 1, 6)
        # Select the predictions relevant to the chosen actions
        pred_action = (predictions * action_resize).sum(2)
        
        loss = self.loss_fn(pred_action, target)
        entropy = -(boltzman * torch.log(boltzman + 1e-7)).sum()
        total_loss = loss + entropy

        # Backward and optimize step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 9999.0)
        self.optimizer.step()
        
        return loss, entropy

def train_worker(rank, args, offsets, a_size, model):
    trainer = Trainer(rank, args, offsets, a_size, model)
    trainer.work()