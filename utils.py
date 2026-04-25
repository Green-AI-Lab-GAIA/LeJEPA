class CSVLogger(object):

    def __init__(self, fname, *argv):
        self.fname = fname
        self.types = []
        # -- print headers
        with open(self.fname, 'w') as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v)
                if i < len(argv):
                    f.write(f"{v},")
                else:
                    f.write(f"{v}\n")

    def log(self, vals):
        with open(self.fname, 'a') as f:
            for i, typ in enumerate(self.types, 1):
                if typ not in vals:
                    raise ValueError(f"Logger was initialized with type {typ} but no value was given.")
                val = vals[typ]
                if i < len(self.types):
                    if isinstance(val, int):
                        f.write(f"{val},")
                    else:
                        f.write(f"{val:.6f},")
                else:
                    if isinstance(val, int):
                        f.write(f"{val}\n")
                    else:
                        f.write(f"{val:.6f}\n")
            
class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import math
class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr