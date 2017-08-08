import sys
try:
    sys.path.remove('/home/dai/.local/lib/python3.6/site-packages')
except:
    pass

import torch.multiprocessing as mp
from train import train

if __name__ == '__main__':
    print(mp.__file__)

    processes = []
    for rank in range(4):
        p = mp.Process(target=train, args=(rank, ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
