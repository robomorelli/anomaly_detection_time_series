import torch.nn as nn

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
                nn.BatchNorm2d(out_f),
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

def conv_block1D(in_f, out_f, kernel_size =3, padding = 1, activation=nn.ReLU(), batch_norm=True,
               pool=True, pool_ks=2, pool_stride=2, pool_pad=0, *args, **kwargs):

    if batch_norm:
        if pool:
            return nn.Sequential(
                nn.Conv1d(in_f, out_f, kernel_size, padding=padding, *args, **kwargs),
                nn.BatchNorm1d(out_f),
                nn.MaxPool1d(pool_ks, pool_stride, pool_pad),
                activation)
        else:
            return nn.Sequential(
                nn.Conv1d(in_f, out_f, kernel_size, padding=padding, *args, **kwargs),
                nn.BatchNorm1d(out_f),
                activation)
    else:
        if pool:
            return nn.Sequential(
                nn.Conv1d(in_f, out_f, kernel_size, padding=padding, *args, **kwargs),
                nn.BatchNorm1d(out_f),
                nn.MaxPool1d(pool_ks, pool_stride, pool_pad),
                activation)
        else:
            return nn.Sequential(
                    nn.Conv1d(in_f, out_f, kernel_size, padding=padding, *args, **kwargs),
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
                    nn.ConvTranspose1d(in_f, out_f, kernel_size, stride *args, **kwargs),
                    activation)
        else:
            return nn.Sequential(
                    nn.ConvTranspose1d(in_f, out_f, kernel_size, stride *args, **kwargs))
