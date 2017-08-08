import sys
try:
    sys.path.remove('/home/dai/.local/lib/python3.6/site-packages')
except:
    pass

import argparse
import os

import torch.multiprocessing as mp
from train import train_worker
from dfp import DFP
from gridworld_goals import *
import torch

parser = argparse.ArgumentParser(description='Goal-based RL')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=1, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-step', type=int, default=100, metavar='NS',
                    help='number of forward steps in A3C (default: 100)')
parser.add_argument('--num-episodes', type=int, default=10000, 
                    help='Number of episodes to run training (default: 10000)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Batch size for training (default: 128)')

        
if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    env = gameEnv(partial=False, size=5)
    s, o_big, m, g, h = env.reset()
    offsets = [1, 2, 4, 8, 16, 32] # Set of temporal offsets
    a_size = 4                     # Number of available actions
    num_measurements = 2           # Number of measurements
    shared_model = DFP(a_size, s.shape, num_measurements, len(offsets))
    # This is important, make sure the model is shared among workers
    # Make sure each worker updates the same model
    # Make sure each worker has their own separate optimizers
    shared_model.share_memory()
    processes = []
    print('Number of processes', args.num_processes)

    train_worker(0, args, offsets, a_size, shared_model)
    '''
    for rank in range(0, args.num_processes):
        p = mp.Process(target=train_worker, args=(rank, args, offsets, a_size, shared_model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    '''
