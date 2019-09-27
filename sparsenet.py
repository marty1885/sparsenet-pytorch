from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class KWinner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, density):
        assert(len(x.shape) == 2)
        assert(0.0 < density < 1.0)

        k = int(x.shape[1]*density)
        v, _ = x.topk(k, sorted=True)

        mask = (x >= v[:,k-1].view(-1, 1)) #a mask for the inhibited variable
        res = x * mask.float()

        ctx.save_for_backward(mask)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        res = grad_output * mask.float()
        return res, None, None, None

kwinner = KWinner.apply

class k_winners2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, density):
        batchSize = x.shape[0]
        boosted = x.detach()

        # Take the boosted version of the input x, find the top k winners.
        # Compute an output that only contains the values of x corresponding to the top k
        # boosted values. The rest of the elements in the output should be 0.
        k = int(density*x.shape[1]*x.shape[2]*x.shape[3])
        boosted = boosted.reshape((batchSize, -1))
        xr = x.reshape((batchSize, -1))
        res = torch.zeros_like(boosted)
        topk, indices = boosted.topk(k, dim=1, sorted=False)
        res.scatter_(1, indices, xr.gather(1, indices))
        res = res.reshape(x.shape)

        ctx.save_for_backward(indices)
        return res


    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we set the gradient to 1 for the winning units, and 0
        for the others.
        """
        batchSize = grad_output.shape[0]
        indices, = ctx.saved_tensors

        g = grad_output.reshape((batchSize, -1))
        grad_x = torch.zeros_like(g, requires_grad=False)
        grad_x.scatter_(1, indices, g.gather(1, indices))
        grad_x = grad_x.reshape(grad_output.shape)

        return grad_x, None, None, None
kwinner2d = k_winners2d.apply

class b_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, t = 3, alpha = 0.1):
        l = x < 0
        r = x > t
        mask = (l+r).sign()
        
        ctx.save_for_backward(mask, torch.FloatTensor([alpha]))
        return l.float()*x*alpha + (1-mask).float()*x + r.float()*((x-t)*alpha+t)

    @staticmethod
    def backward(ctx, grad_output):
        mask,alpha = ctx.saved_tensors
        mask = mask.float()
        alpha = alpha.item()
        res = mask*grad_output*alpha + (1-mask)*grad_output
        return res, None, None, None

brelu = b_relu.apply

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = brelu(self.conv1(x), 4)
        x = F.max_pool2d(x, 2, 2)
        x = kwinner2d(x, 0.04)
        x = brelu(self.conv2(x), 4)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        x = kwinner(x, 0.08)
        x = brelu(self.fc1(x), 4)
        x = kwinner(x, 0.04)
        x = self.fc2(x)
        return F.softmax(x)
