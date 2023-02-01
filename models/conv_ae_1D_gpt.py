import os
import torch.nn as nn
from models.utils import conv_block1D, deconv_block1D
import torch
from tqdm import tqdm

import torch
import torch.nn as nn


class CONV_AE1DGPT(nn.Module):
    def __init__(self, in_channel=16,  filter_num= 32, kernel_size=3, activation = nn.ReLU()):
        super(CONV_AE1DGPT, self).__init__()

        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.act = activation

        self.filter_num=filter_num
        self.filter_num_halved = int(filter_num/2)

        self.encoder = nn.Sequential(
            nn.Conv1d(16, self.filter_num, kernel_size=self.kernel_size, stride=2, padding=1),
            self.act,
            nn.Conv1d(self.filter_num, self.filter_num_halved, kernel_size=self.kernel_size, stride=2, padding=1),
            self.act
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.filter_num_halved, self.filter_num, kernel_size=self.kernel_size, stride=2, padding=1, output_padding=1),
            self.act,
            nn.ConvTranspose1d(self.filter_num, 16, kernel_size=self.kernel_size, stride=2, padding=1, output_padding=1),
            self.act
        )

        print(self)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_conv_ae1Dgpt(param_conf, train_iter, test_iter, model, criterion, optimizer,scheduler, device,
           out_dir, model_name, epochs=100):
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

            temp_val_loss= temp_val_loss / val_steps
            print('eval loss {}'.format(temp_val_loss))
            scheduler.step()
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


'''
  (act): ELU(alpha=1.0)
  (encoder): Sequential(
    (0): Conv1d(16, 64, kernel_size=(3,), stride=(2,), padding=(1,))
    (1): ELU(alpha=1.0)
    (2): Conv1d(64, 32, kernel_size=(3,), stride=(2,), padding=(1,))
    (3): ELU(alpha=1.0)
  )
  (decoder): Sequential(
    (0): ConvTranspose1d(32, 64, kernel_size=(3,), stride=(2,), padding=(1,), output_padding=(1,))
    (1): ELU(alpha=1.0)
    (2): ConvTranspose1d(64, 16, kernel_size=(3,), stride=(2,), padding=(1,), output_padding=(1,))
    (3): ELU(alpha=1.0)
  )
)'''

'''
CONV_AE1D(
  (act): ELU(alpha=1.0)
  (encoder): Encoder(
    (nn_enc): Sequential(
      (enc_lay_0): Sequential(
        (0): Conv1d(16, 64, kernel_size=(3,), stride=(1,), padding=(1,))
        (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): ELU(alpha=1.0)
      )
      (enc_lay_1): Sequential(
        (0): Conv1d(64, 32, kernel_size=(3,), stride=(1,), padding=(1,))
        (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): ELU(alpha=1.0)
      )
    )
    (act): ELU(alpha=1.0)
  )
  (decoder): Decoder(
    (nn_dec): Sequential(
      (dec_lay_0): Sequential(
        (0): ConvTranspose1d(32, 64, kernel_size=(2,), stride=(2,))
        (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ELU(alpha=1.0)
      )
      (dec_lay_1): Sequential(
        (0): ConvTranspose1d(64, 16, kernel_size=(2,), stride=(2,))
        (1): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ELU(alpha=1.0)
      )
    )
    (act): ELU(alpha=1.0)
    (reshape): Linear(in_features=50, out_features=128, bias=True)
  )
)


'''