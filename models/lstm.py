import os
import torch
import torch.nn as nn
import torch
from tqdm import tqdm
import sys
sys.path.append('..')
from utils.opt import EarlyStopping
import matplotlib.pyplot as plt
torch.manual_seed(0)

####################
# LSTM Autoencoder #
####################
# code inspired by  https://github.com/shobrook/sequitur/blob/master/sequitur/autoencoders/rae.py
# annotation sourced by  https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
class Encoder(nn.Module):
    def __init__(self, seq_in, seq_out, n_features, output_size,
                 embedding_size, n_layers_1=2, n_layers_2=1):
        super().__init__()

        # self.seq_len = seq_in
        self.n_features = n_features  # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size  # the number of features in the embedded points of the inputs' number of features
        self.n_layers_1 = n_layers_1
        self.n_layers_2 = n_layers_2
        self.hidden_size = (2 * embedding_size)  # The number of features in the hidden state h
        self.seq_out = seq_out
        self.output_size = output_size

        self.LSTMenc = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=n_layers_1,
            batch_first=True
        )
        self.LSTM1 = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=embedding_size,
            num_layers=n_layers_2,
            batch_first=True
        )

        self.out = nn.Linear(embedding_size, self.output_size)
        self.apply(self.weight_init)

        print(self)
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        # x is the output and so the size is > (batch, seq_len, hidden_size)
        # x, (_,_) = self.LSTM1(x)
        x, (hidden_state, cell_state) = self.LSTMenc(x)
        x, (hidden_state, cell_state) = self.LSTM1(x) ### to switch to x because it needs repeated sequence
        #  ht[:,0,:] == x[0, -1, :] (or also x[0,-1,:]==hidden_state[-1,0,:]) with x ([32, 10, 64]) and h_state ([1, 32, 64])
        #last_lstm_layer_hidden_state = hidden_state[-1, :, :] #take only the last layer
        #we need hidden state only here because is like our encoding of the time series
        out = self.out(x)
        return out

class LSTM(nn.Module):
    def __init__(self, seq_in = 7, seq_out= 7, n_features = 8, output_size=16,
                 embedding_dim=64, n_layers_1=2, n_layers_2=2):
        super().__init__()

        self.seq_in = seq_in
        self.seq_out = seq_out
        self.output_size = output_size
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.n_layers_1 = n_layers_1
        self.n_layers_2 = n_layers_2

        self.encoder = Encoder(self.seq_in, self.seq_out, self.n_features,self.output_size,
                               self.embedding_dim, self.n_layers_1, self.n_layers_2)

    def forward(self, x):
        torch.manual_seed(0)
        y = self.encoder(x)
        return y


def train_lstm(param_conf, train_iter, test_iter, model, criterion, optimizer, scheduler,
                  device,out_dir, model_name, epochs=100, es_patience=10):
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

    early_stopping = EarlyStopping(patience=es_patience)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    val_loss = 10 ** 16
    val_losses = []
    train_losses = []
    for epoch in tqdm(range(epochs), unit='epoch'):
        train_loss = 0.0
        train_steps = 0
        for i, batch in tqdm(enumerate(train_iter), total=len(train_iter), unit="batch"):
            model.train()
            optimizer.zero_grad()

            # y.requires_grad_(True)
            out = model(batch[0].to(device))
            loss = criterion(out.to(device), batch[1].to(device))
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
        train_losses.append(train_loss/train_steps)

        model.eval()
        val_steps = 0
        temp_val_loss = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_iter), total=len(test_iter), desc="Evaluating"):

                out = model(batch[0].to(device))
                loss = criterion(out.to(device), batch[1].to(device)).item()
                temp_val_loss += loss
                val_steps += 1

            temp_val_loss= temp_val_loss / val_steps
            scheduler.step(temp_val_loss)
            print('eval loss {}'.format(temp_val_loss))
            
            val_losses.append(temp_val_loss)       
            epochs = [x for x in range(len(train_losses))]
            
      
            fig = plt.figure(figsize=(4,3))

            plt.plot(epochs, train_losses, marker='.',label = "train mse loss")
            plt.plot(epochs, val_losses,marker='.', label = "val mse loss")
            plt.xlabel('epochs', fontsize=18)
            plt.ylabel('mse value', fontsize=18)
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.legend(fontsize=14)
            plt.show()

            early_stopping(temp_val_loss)
            if early_stopping.early_stop:
                break

            if temp_val_loss < val_loss:
                print('val_loss improved from {} to {}, saving model  {} to {}' \
                      .format(val_loss, temp_val_loss, model_name, out_dir))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'param_conf': param_conf,
                }, out_dir + '/{}.pth'.format(model_name))
                #torch.save(model, out_dir + '{}.pth'.format(model_name))
                val_loss = temp_val_loss
