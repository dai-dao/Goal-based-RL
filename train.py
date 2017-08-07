import torch
from torch.autograd import Variable
import torch.optim as optim
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
                boltzman, predictions, action = self.model(s_var, m_var, g_var, temp)
                action = action.data.numpy()[0]
                # Add to episode buffer
                episode_buffer.append(Experience(s, action, m, current_goal, None))

                # Perform the action on the environment
                s, s1_big, m, g, h, d = self.env.step(action) 
                t += 1

                # End the episode after num steps
                if t > self.args.num_step:
                    d = True

            # Training statistics
            episode_deliveries.append(m[0])
            episode_lengths.append(t)

            # Update the network using experience buffer at the end
            # of the episode
            loss, entropy = self.train(episode_buffer)
            if loss != 0 and self.rank == 0 and episode_index % 300 == 0:
                print('Episode {} | Loss {} | Episode length {} | Deliveries {}'.
                        format(episode_index, loss, t, m[0]))

            if episode_index % 1000 == 0 and self.rank == 0:
                torch.save(self.model.state_dict, 'models/model-weights.pth')

    def train(self, rollout):
        # rollout is a list of Experience
        batch = Experience(*zip(*rollout))

        measurements = np.vstack(batch.measurement)
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
            
            loss, entropy = self.model.compute_loss(obs_var, mea_var,
                                                    goa_var, temperature, 
                                                    act_var, tar_var, self.optimizer)
            return loss.data.numpy()[0] / len(rollout), entropy / len(rollout)
        else:
            return 0, 0


