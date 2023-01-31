import os 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
sys.path.append('..')
from dataset.sentinel import *
from config import *


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def unrolling_batches_conv(num_batch, test_iter, sequence_length, cols, shift=None):
    
    if shift==None:
        shift = np.random.randint(len(test_iter.dataset.df_data)-100)
    x = np.zeros((num_batch, sequence_length, len(cols)))

    x = np.zeros((num_batch, sequence_length, len(cols)))
    for i in range(num_batch):

        x[i,:,:] = test_iter.dataset.df_data.iloc[shift + i*sequence_length:shift + (i+1)*sequence_length,:]\
        .values

    x = torch.from_numpy(x).float()
    x = torch.unsqueeze(x, 1)
    
    return x, shift

def read_dataset_conv(dataset_name, scaled,columns_subset, dataset_subset, cols, 
                 train_val_split, sequence_length, out_window,  predict=False,
                 batch_size=4, transform=None):
    
    target=None
    xdf = pd.read_pickle(dataset_name)

    if columns_subset:
        cols = cols[:columns_subset]
    dataRaw = xdf[cols].dropna()

    if dataset_subset:
        dataRaw = dataRaw.iloc[:dataset_subset, :]

    df = dataRaw.copy()
    x = df.values

    if scaled:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        dfNorm = pd.DataFrame(x_scaled, columns=df.columns)
    else:
        dfNorm = pd.DataFrame(x, columns=df.columns)

    X_train, X_test, y_train, y_test = train_test_split(dfNorm, dfNorm, \
                                                        train_size=train_val_split\
                                                        , shuffle=False)
    df_train = pd.DataFrame(X_train, columns=dfNorm.columns)
    df_test = pd.DataFrame(X_test, columns=dfNorm.columns)

    

    train_dataset = Dataset_seq(df_train, target = target, sequence_length = sequence_length,
                                out_window = out_window, prediction=predict,\
                                transform=transform)
    train_iter = DataLoader(dataset=train_dataset,\
                            batch_size=batch_size, shuffle=True)

    test_dataset = Dataset_seq(df_test, target = target, \
                               sequence_length = sequence_length,
                                out_window = out_window, prediction=predict,\
                               transform=transform)
    test_iter = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return df_train, df_test, train_iter, test_iter

def show_results(x, yo, cols, model_name, params_conf, par_nums, shift, 
                 num_batch, architecture='conv_ae', save=False):
    
    path = './figure_results/evaluation/{}/{}/shift_{}/'.format(architecture, model_name, shift, num_batch)
    for i in range(x.shape[2]):
        fig, ax = plt.subplots(1,1, figsize=(8,4))

        ax.plot(yo.flatten(0,1)[:,i].to("cpu").detach().numpy(), 
                   linestyle='--', label='reconstr', color='black')
        ax.plot(x.flatten(0,1)[:,i].to("cpu").detach().numpy(), label ='original')
        x_i = x.flatten(0,1)[:,i]
        y_o = yo.flatten(0,1)[:,i]

        loss_u = torch.nn.L1Loss(reduction='none')(y_o, x_i)
        loss_u = moving_average(loss_u.to("cpu").detach().numpy(), params_conf['sequence_length'])

        ax.plot(loss_u, label ='mean abs err')
        ax.set_xlabel('time steps')
        ax.set_ylabel('{}'.format(cols[i]))
        ax.set_title('Input Vs Recon. Par num {} s_l {}'\
                        .format(par_nums, 
                                params_conf['sequence_length'], 
                                #params_conf['latent_dim'], 
                                #params_conf['n_layers'],
                                #params_conf['filter_num'],
                                ))
        ax.legend()
        if save:
            os.makedirs(path, exist_ok=True)
            plt.savefig(path + '{}_shift_{}_batch_{}.png'.format(cols[i], num_batch))
