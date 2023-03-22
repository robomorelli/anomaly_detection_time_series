from torch import nn
import torch
import torch.nn.functional as F
import math

def conv_block(in_f, out_f, kernel_size =3, padding = 1, activation=nn.ReLU(), batch_norm=True,
               pool=True, pool_ks=2, pool_stride=2, pool_pad=0, *args, **kwargs):

    if batch_norm:
        if pool:
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size, padding=padding, *args, **kwargs),
                nn.BatchNorm2d(out_f),
                nn.MaxPool2d(pool_ks, pool_stride, pool_pad),
                activation)
        else:
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size, padding=padding, *args, **kwargs),
                nn.BatchNorm2d(out_f),
                activation)
    else:
        if pool:
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size, padding=padding, *args, **kwargs),
                nn.MaxPool2d(pool_ks, pool_stride, pool_pad),
                activation)
        else:
            return nn.Sequential(
                    nn.Conv2d(in_f, out_f, kernel_size, padding=padding, *args, **kwargs),
                    activation)

def deconv_block(in_f, out_f, kernel_size = 2, stride = 2, activation=nn.ReLU(), batch_norm=True, *args, **kwargs):

    if batch_norm:
        if activation:
            return nn.Sequential(
                    nn.ConvTranspose2d(in_f, out_f, kernel_size, stride, *args, **kwargs),
                    nn.BatchNorm2d(out_f),
                    activation)
        else:
            return nn.Sequential(
                    nn.ConvTranspose2d(in_f, out_f, kernel_size, stride, *args, **kwargs),
                    nn.BatchNorm2d(out_f))
    else:
        if activation:
            return nn.Sequential(
                    nn.ConvTranspose2d(in_f, out_f, kernel_size, stride *args, **kwargs),
                    activation)
        else:
            return nn.Sequential(
                    nn.ConvTranspose2d(in_f, out_f, kernel_size, stride *args, **kwargs))

def conv_block1D(in_f, out_f, kernel_size =3, padding = 1, stride=1, activation=nn.ReLU(), batch_norm=True,
               pool=True, pool_ks=2, pool_stride=2, pool_pad=0, *args, **kwargs):

    if batch_norm:
        if pool:
            return nn.Sequential(
                nn.Conv1d(in_f, out_f, kernel_size, padding=padding, stride=stride, *args, **kwargs),
                nn.BatchNorm1d(out_f),
                nn.MaxPool1d(pool_ks, pool_stride, pool_pad),
                activation)
        else:
            return nn.Sequential(
                nn.Conv1d(in_f, out_f, kernel_size, padding=padding, stride=stride, *args, **kwargs),
                nn.BatchNorm1d(out_f),
                activation)
    else:
        if pool:
            return nn.Sequential(
                nn.Conv1d(in_f, out_f, kernel_size, padding=padding, stride=stride, *args, **kwargs),
                nn.MaxPool1d(pool_ks, pool_stride, pool_pad),
                activation)
        else:
            return nn.Sequential(
                    nn.Conv1d(in_f, out_f, kernel_size, padding=padding, stride=stride, *args, **kwargs),
                    activation)

def deconv_block1D(in_f, out_f, kernel_size = 2, stride = 2, activation=nn.ReLU(), batch_norm=True, *args, **kwargs):

    if batch_norm:
        if activation:
            return nn.Sequential(
                    nn.ConvTranspose1d(in_f, out_f, kernel_size, stride, *args, **kwargs),
                    nn.BatchNorm1d(out_f),
                    activation)
        else:
            return nn.Sequential(
                    nn.ConvTranspose1d(in_f, out_f, kernel_size, stride, *args, **kwargs),
                    nn.BatchNorm1d(out_f))
    else:
        if activation:
            return nn.Sequential(
                    nn.ConvTranspose1d(in_f, out_f, kernel_size, stride, *args, **kwargs),
                    activation)
        else:
            return nn.Sequential(
                    nn.ConvTranspose1d(in_f, out_f, kernel_size, stride, *args, **kwargs))


clip_x_to0 = 1e-4

class InverseSquareRootLinearUnit(nn.Module):

    def __init__(self, min_value=5e-3):
        super(InverseSquareRootLinearUnit, self).__init__()
        self.min_value = min_value

    def forward(self, x):
        return 1. + self.min_value \
               + torch.where(torch.gt(x, 0), x, torch.div(x, torch.sqrt(1 + (x * x))))

class ClippedTanh(nn.Module):

    def __init__(self):
        super(ClippedTanh, self).__init__()

    def forward(self, x):
        return 0.5 * (1 + 0.999 * torch.tanh(x))

class ClippedTanh0(nn.Module):

    def __init__(self):
        super(ClippedTanh0, self).__init__()

    def forward(self, x):
        return 0*x

class SmashTo0(nn.Module):

    def __init__(self):
        super(SmashTo0, self).__init__()

    def forward(self, x):
        return 0*x

class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)

class SigmaPrior(nn.Module):

    def __init__(self, min_value=5e-3):
        super(SmashTo0, self).__init__()

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return 0*x

class LinConstr(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinConstr, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _max_norm(self, w):
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self._max_norm_val)
        return w * (desired / (self._eps + norm))

    def forward(self, x):
        x = F.linear(x, self.weight.clamp(min=-1.0*10**6, max=1.0*10**6), self.bias)
        return x

