import os
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('..')
from utils.opt import EarlyStopping
from utils.layers import conv_block1D, deconv_block1D

class Encoder(nn.Module):
    def __init__(self, in_channel=1, kernel_size=3, padding=1, filter_num_list=None, latent_dim=10,
                  length=16, activation=nn.ReLU(), stride=1, pool=True, bn=True, dilation=1, flattened=True):
        super(Encoder, self).__init__()

        self.nn_enc = nn.Sequential()

        if filter_num_list is None:
            self.filter_num_list = [1, 32, 64]

        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.filter_num_list = filter_num_list
        self.latent_dim = latent_dim
        self.l = length
        self.act = activation
        self.stride = stride
        self.pool = pool
        self.bn = bn
        self.dilation = dilation
        self.padding=padding
        self.flattened=flattened

        for i, num in enumerate(self.filter_num_list):
            if i + 2 == len(self.filter_num_list):
                self.nn_enc.add_module('enc_lay_{}'.format(i), conv_block1D(num, self.filter_num_list[i + 1],
                                                                          self.kernel_size, activation=self.act, stride=self.stride,
                                                                            pool=self.pool,  batch_norm=self.bn, padding=self.padding))
                break
            self.nn_enc.add_module('enc_lay_{}'.format(i), conv_block1D(num, self.filter_num_list[i+1],
                                                                  self.kernel_size, activation=self.act, stride=self.stride,
                                                                        pool=self.pool, batch_norm=self.bn, padding=self.padding))

        self.flattened_size, self.l_enc = self._get_final_flattened_size()

        if self.flattened:
            self.encoder_layer = nn.Linear(self.flattened_size, self.latent_dim)

        self.init_kaiming_normal()

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, self.in_channel, self.l)
            )
            x = self.nn_enc(x)
            _, c, w = x.size()
        return c * w, w

    def forward(self, x):
        enc = self.nn_enc(x)
        if self.flattened:
            enc = enc.view(-1, self.flattened_size)
            enc = self.encoder_layer(enc)
        return enc


class Decoder(nn.Module):
    def __init__(self, in_channel=1, kernel_size=3, filter_num_list=None, latent_dim=10, flattened_size=None,
                 length=16, activation=nn.ReLU(), bn=True, flattened=True):
        super(Decoder, self).__init__()

        self.nn_dec = nn.Sequential()

        if filter_num_list is None:
            self.filter_num_list = [32, 64]
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.filter_num_list = filter_num_list
        self.latent_dim = latent_dim
        self.flattened_size = flattened_size
        self.length = length
        self.filter_num_list = self.filter_num_list[::-1]
        self.act = activation
        self.bn = bn
        self.flattened=flattened

        if self.flattened:
            self.reshape = nn.Linear(self.latent_dim, self.flattened_size)

        for i, num in enumerate(self.filter_num_list):
            if i + 2 == len(self.filter_num_list):
                self.nn_dec.add_module('dec_lay_{}'.format(i), deconv_block1D(num, self.filter_num_list[i+1], activation=self.act, batch_norm = self.bn))
                break
            self.nn_dec.add_module('dec_lay_{}'.format(i), deconv_block1D(num, self.filter_num_list[i + 1], activation=self.act, batch_norm = self.bn))

        self.decoder_layer = nn.Conv1d(self.filter_num_list[i+1], self.in_channel, kernel_size=1)
        self.init_kaiming_normal()

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.flattened:
            x = self.reshape(x)
            x = x.view((-1, self.filter_num_list[0], self.length))
        dec = self.nn_dec(x)
        dec = self.decoder_layer(dec)

        #dec1 = self.nn_dec[0](x)
        #dec2 = self.nn_dec[1](dec1)
        #dec3 = self.nn_dec[2](dec2)

        return dec

# define the NN architecture
class CONV_AE1D(nn.Module):
    def __init__(self, in_channel=16,  length=16, kernel_size=3, filter_num=2,
                 latent_dim=50, n_layers=1, activation = nn.ReLU(), stride=1, dilation=1,
                 pool=True, bn=True,
                 increasing=False, flattened=True):
        super(CONV_AE1D, self).__init__()

        self.in_channel = in_channel
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.act = activation
        self.l = length  # sequence length (to transpose respect to the df format)
        self.stride = stride
        self.pool = pool
        self.bn = bn
        self.increasing = increasing
        self.dilation=dilation
        self.flattened=flattened

        self.padding = int((self.dilation * (self.kernel_size - 1) / 2))

        if not self.pool:
            self.stride = 2

        if self.increasing:
            self.filter_num_list = [int(self.filter_num * ((ix + 1) * 2)) for ix in range(self.n_layers)]
        else:
            self.filter_num_list = [int(self.filter_num / ((ix + 1) * 2)) for ix in range(self.n_layers)]

        self.filter_num_list = [self.in_channel] + [self.filter_num] + self.filter_num_list

        self.encoder = Encoder(self.in_channel, kernel_size=self.kernel_size, padding=self.padding, filter_num_list=self.filter_num_list,
                               latent_dim=self.latent_dim, length=self.l, activation=self.act, stride=self.stride, pool=self.pool,
                               bn=self.bn, flattened=self.flattened)
        self.flattened_size = self.encoder.flattened_size
        self.decoder = Decoder(self.in_channel, kernel_size=self.kernel_size, filter_num_list=self.filter_num_list,
                               latent_dim=self.latent_dim, flattened_size=self.flattened_size,
                               length=self.encoder.l_enc, activation=self.act,
                               bn=self.bn, flattened=self.flattened)

        print(self)
    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)
        return out

def train_conv_ae1D(param_conf, train_iter, test_iter, model, criterion, optimizer,scheduler, device,
           out_dir, model_name, epochs=100, es_patience=10):
    """
    Training function.

    Args:
        train_iter: (DataLoader): train data iterator
        test_iter: (DataLoader): test data iterator
        model: model
        criterion: loss to use
        optimizer: optimizer to use
        config:
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    early_stopping = EarlyStopping(patience=es_patience)

    val_loss = 10 ** 16
    for epoch in tqdm(range(epochs), unit='epoch'):
        print('epoch', epoch)
        train_loss = 0.0
        train_steps = 0
        for i, batch in tqdm(enumerate(train_iter), total=len(train_iter), unit="batch"):
            model.train()
            optimizer.zero_grad()

            # y.requires_grad_(True)
            y_o = model(batch[0].to(device))
            loss = criterion(y_o.to(device), batch[1].to(device))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            train_loss += loss.item()
            train_steps += 1
            # if (i + 1) % config['gradient_accumulation_steps'] == 0:
            optimizer.step()

            if i % 10 == 0:
                print("Loss:")
                print(loss.item())

        print('train loss at the end of epoch is ', train_loss/train_steps)

        model.eval()
        val_steps = 0
        temp_val_loss = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_iter), total=len(test_iter), desc="Evaluating"):

                y_o = model(batch[0].to(device))
                loss = criterion(y_o.to(device), batch[1].to(device)).item()
                temp_val_loss += loss
                val_steps += 1

            early_stopping(temp_val_loss)
            if early_stopping.early_stop:
                break

            temp_val_loss= temp_val_loss / val_steps
            print('eval loss {}'.format(temp_val_loss))
            scheduler.step(temp_val_loss)
            if temp_val_loss < val_loss:
                print('val_loss improved from {} to {}, saving model  {} to {}' \
                      .format(val_loss, temp_val_loss, model_name, out_dir))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'param_conf': param_conf,
                }, out_dir + '/{}.pth'.format(model_name))
                val_loss = temp_val_loss
