from jittor import nn


class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

        self.basic_lr = 1e-3
        self.lr_decay = 0.7
        self.decay_step = 2e4

    def step(self, step):
        lr_decay = self.lr_decay ** int(step / self.decay_step)
        lr_decay = max(lr_decay, 1e-2)
        self.optimizer.lr = lr_decay * self.basic_lr

