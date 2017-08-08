
import sys
try:
    sys.path.remove('/home/dai/.local/lib/python3.6/site-packages')
except:
    pass

import torch
import torch.multiprocessing as mp

print("Torch module used", torch.__file__)


def wait(rank):
    print('This is rank', rank)
    while True:
        x = 1

if __name__ == '__main__':
    processes = []

    for rank in range(0, 4):
            p = mp.Process(target=wait, args=(rank, ))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()