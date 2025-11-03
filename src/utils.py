import random, numpy as np, torch, os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum=0; self.count=0
    def update(self, val, n=1): self.sum += val*n; self.count += n
    @property
    def avg(self): return self.sum / max(1, self.count)

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
