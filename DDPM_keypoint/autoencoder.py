import time

import torch
from torch.nn import Module

from .encoders import *
from .diffusion import *


class AutoEncoder(Module):

    def __init__(self):
        super().__init__()
        self.encoder = PointNetEncoder(zdim=256,input_dim=1)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=256, residual=True),
            var_sched = VarianceSchedule(
                num_steps=1000,
                beta_1=0.0001,
                beta_T=0.05,
                mode='linear'
            )
        )

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        code, _ = self.encoder(x)
        return code

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(num_points, code, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x,para):
        code = self.encode(para)
        loss = self.diffusion.get_loss(x, code)
        return loss

if __name__ == '__main__':
    torch.cuda.synchronize()
    start = time.time()
    model = AutoEncoder().cuda()
    x=torch.randn(4,13,1).cuda()
    y = torch.randn(4, 16, 3).cuda()
    code = model.encode(x)
    output = model.get_loss(y,x)
    torch.cuda.synchronize()
    end = time.time()
    print('infer_time:', end - start)