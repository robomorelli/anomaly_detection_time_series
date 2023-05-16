import numpy as np
import random
import os, errno
import sys
from tqdm import tqdm
import sys
sys.path.append('..')
from utils.opt import EarlyStopping

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers=1, n_cells=1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_cells = n_cells


        if self.n_cells == 1:
            # define LSTM layer
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size*2,
                                num_layers=num_layers, batch_first=True)
            self.lstm1 = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True)

    def forward(self, x_input):
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''
        #h0, c0 = (torch.zeros(self.num_layers, x_input.shape[0], self.hidden_size, requires_grad=True).to(x_input.device),
        # torch.zeros(self.num_layers, x_input.shape[0], self.hidden_size, requires_grad=True).to(x_input.device))

        if self.n_cells == 1:
            lstm_out, self.hidden = self.lstm(x_input)

        else:
            lstm_out, _ = self.lstm(x_input)
            lstm_out, self.hidden = self.lstm1(lstm_out, (h0,c0))

        return lstm_out, self.hidden

    def init_hidden(self, batch_size):
        
        #initialize hidden state
        #: param batch_size:    x_input.shape[1]
        #: return:              zeroed hidden state and cell state

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))



class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, num_layers=1, n_cells=1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_cells = n_cells

        if self.n_cells == 1:

            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, input_size)

        else:

            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True)
            self.lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size*2,
                                num_layers=num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size*2, input_size)

    def forward(self, x_input, encoder_hidden_states):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''

        if self.n_cells == 1:
            lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
            output = self.linear(lstm_out)
        else:
            lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
            lstm_out, _ = self.lstm1(lstm_out)
            output = self.linear(lstm_out)

        return output, self.hidden


class ENC_DEC_LSTM(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, seq_in, seq_out, input_size = 16,
                 hidden_size=32, n_layers=1, n_cells=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.n_cells = n_cells

        self.encoder = lstm_encoder(self.input_size,
                               self.hidden_size, self.n_layers, self.n_cells)
        self.decoder = lstm_decoder(self.input_size,
                               self.hidden_size, self.n_layers, self.n_cells)
    def forward(self, x):

        # outputs tensor
        outputs = torch.zeros(x.shape[0], self.seq_out, self.input_size)
        # initialize hidden state
        encoder_output, encoder_hidden = self.encoder(x)

        decoder_input = x[:, -1, :].unsqueeze(1)  # shape: (batch_size, input_size)
        decoder_hidden = encoder_hidden

        for t in range(self.seq_out):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:,t,:] = decoder_output.squeeze(1)
            decoder_input = decoder_output

        return outputs



def train_enc_dec_lstm(param_conf, train_iter, test_iter, model, criterion, optimizer, scheduler,
                  device,out_dir, model_name, epochs=100, es_patience=10):
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

            # initialize hidden state
            #encoder_hidden = model.encoder.init_hidden(batch[0].shape[0])

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

        print('train loss at the end of epoch is ', train_loss / train_steps)
        train_losses.append(train_loss / train_steps)

        model.eval()
        val_steps = 0
        temp_val_loss = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_iter), total=len(test_iter), desc="Evaluating"):
                out = model(batch[0].to(device))
                loss = criterion(out.to(device), batch[1].to(device)).item()
                temp_val_loss += loss
                val_steps += 1

            temp_val_loss = temp_val_loss / val_steps
            scheduler.step(temp_val_loss)
            print('eval loss {}'.format(temp_val_loss))

            val_losses.append(temp_val_loss)

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
                val_loss = temp_val_loss

''' 
    def predict(self, input_tensor, target_len):

        
        #: param input_tensor:      input data (seq_len, input_size); PyTorch tensor
        #: param target_len:        number of target values to predict
        #: return np_outputs:       np.array containing predicted values; prediction done recursively
        

        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(1)  # add in batch size of 1
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[2])

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden

        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output

        np_outputs = outputs.detach().numpy()

        return
    '''