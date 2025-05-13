import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

class WarmUpSch:
    def __init__(
        self,
        opt,
        warmup_steps,
        base_lr,
        final_lr,
        patience = 20,
        factor = 0.5,
        min_lr = 1e-6,
        threshold = 1e-1
    ):
        self.opt = opt
        self.warmup_steps = warmup_steps
        self.current_steps = 0
        self.phase = 0

        self.base_lr = base_lr
        self.final_lr = final_lr

        self.sch = ReduceLROnPlateau(opt, 'min', factor, patience, threshold=threshold, min_lr=min_lr)
    
    def step_iter(self, metrics = None):
        self.current_steps += 1
        if self.current_steps <= self.warmup_steps:
            warmup_lr = self.base_lr + (self.final_lr - self.base_lr) * (self.current_steps / self.warmup_steps)
            for param_group in self.opt.param_groups:
                param_group['lr'] = warmup_lr
        else:
            self.phase = 1
    
    def step(self, metrics = None):
        if self.phase == 0:
            return
        self.sch.step(metrics)

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.opt.param_groups]