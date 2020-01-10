from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class KWinner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, density, factor=None):
        top_ignore = 0
        assert(len(x.shape) == 2)
        assert(0.0 < density < 1.0)

        batch_size = x.shape[0]

        xp = x if factor is None else x*factor
        k = int(x.shape[1]*density+top_ignore)
        v, _ = xp.topk(k, sorted=True, dim=1)
        mask = (xp >= v[:,k-1].reshape(batch_size, 1)).float() #a mask for the inhibited variable
        #mask2 = (xp <= v[:,top_ignore-1].reshape(batch_size, 1)).float()
        mask = mask#*mask2
        #stddev, mean = 0.1, 0
        res = x * mask # + (1-mask.float())*torch.cuda.FloatTensor(*x.shape).normal_(mean, stddev)

        ctx.save_for_backward(mask)
        return res, mask

    @staticmethod
    def backward(ctx, grad_output, _):
        mask, = ctx.saved_tensors
        res = grad_output * mask.float()
        #res = grad_output
        return res, None, None, None

kwinner = KWinner.apply

class k_winners2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, density):
        batch_size = x.shape[0]

        k = int(density*x.shape[1]*x.shape[2]*x.shape[3])

        xr = x.reshape((batch_size, -1))
        res = torch.zeros_like(x)
        v, indices = xr.topk(k, dim=1, sorted=True)
        mask = (xr >= v[:,k-1].view(-1, 1)) #a mask for the inhibited variable
        #av = xr[mask]
        #mean, stddev = xr.mean(), xr.std()
        res = xr * mask.float() #+ (1-mask.float())*torch.cuda.FloatTensor(*xr.shape).normal_(mean, stddev)
        res = res.reshape(x.shape)
        mask = mask.reshape(x.shape)

        ctx.save_for_backward(indices)
        return res, mask


    @staticmethod
    def backward(ctx, grad_output, _):
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

class KWinnerLayer(nn.Module):
    def __init__(self, input_shape, density, boost_factor = 0):
        super(KWinnerLayer, self).__init__()
        self.density = density
        self.boost_factor = boost_factor
        self.active_average = nn.Parameter(torch.ones(input_shape)*density, requires_grad=False)

    def forward(self, x):
        res = None
        if self.boost_factor != 0 and self.training:
            factor = torch.exp((self.density - self.active_average)*self.boost_factor)
            factor = factor.detach()
            res, mask = kwinner(x, self.density, factor)
        else:
            res, mask = kwinner(x, self.density)

        if self.boost_factor != 0 and self.training:
            self.active_average = nn.Parameter(0.999*self.active_average + mask.float().mean(0)*0.001)

        return res

class KWinnerLayer2D(nn.Module):
    def __init__(self, input_shape, channels, density, boost_factor = 0):
        super(KWinnerLayer2D, self).__init__()
        self.density = density
        self.boost_factor = boost_factor
        self.average_activation = nn.Parameter(torch.zeros([channels] + list(input_shape))
        	, requires_grad=False)

    def forward(self, x):
        res = None
        if self.boost_factor != 0:
            #Input boosting
            factor = torch.exp((self.density - self.average_activation)*self.boost_factor)
            factor = factor.detach()
            res, mask = kwinner2d(x*factor, self.density)
        else:
            res, mask = kwinner2d(x, self.density)

        if self.boost_factor != 0 and self.training:
            self.average_activation = nn.Parameter(self.average_activation*0.999+mask.float().mean(0)*0.001)

        return res

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

class BRelu(nn.Module):
    def __init__(self, t):
        super(BRelu, self).__init__()
        self.t = t

    def forward(self, x):
        return x#b_relu(x, self.t)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.conv3 = nn.Conv2d(16, 120, 4, 1)
        self.fc1 = nn.Linear(120, 64)
        self.fc2 = nn.Linear(64, 10)

        self.kw1 = KWinnerLayer(120, 0.06, 2.5)
        self.kw2 = KWinnerLayer(64, 0.08, 2.5)
        

    def forward(self, x):
        x = brelu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = brelu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = brelu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = self.kw1(x)
        x = self.fc1(x)
        x = self.kw2(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.conv3 = nn.Conv2d(16, 120, 4, 1)
        self.fc1 = nn.Linear(120, 64)
        self.fc2 = nn.Linear(64, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x), 4)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x), 4)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x), 4)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = brelu(x)
        x = self.fc2(x)
        x = brelu(x)
        return F.softmax(x, dim=1)
        
class LenetDropout(nn.Module):
    def __init__(self):
        super(LenetDropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.conv3 = nn.Conv2d(16, 120, 4, 1)
        self.fc1 = nn.Linear(120, 64)
        self.fc2 = nn.Linear(64, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x), 4)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x), 4)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x), 4)
        x = x.view(x.shape[0], -1)
        x = F.dropout(x, 0.2)
        x = self.fc1(x)
        x = brelu(x)
        x = F.dropout(x, 0.2)
        x = self.fc2(x)
        x = brelu(x)
        return F.softmax(x, dim=1)
