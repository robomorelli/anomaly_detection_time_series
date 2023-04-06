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

def unrolling_batches_conv(num_batch, test_iter, sequence_length, cols, shift=None, arch='conv_ae'):
    
    if shift==None:
        shift = np.random.randint(len(test_iter.dataset.df_data)-100)
    x = np.zeros((num_batch, sequence_length, len(cols)))

    x = np.zeros((num_batch, sequence_length, len(cols)))
    for i in range(num_batch):

        x[i,:,:] = test_iter.dataset.df_data.iloc[shift + i*sequence_length:shift + (i+1)*sequence_length,:]\
        .values
    
    x = torch.from_numpy(x).float()
    if arch=='conv_ae':
        x = torch.unsqueeze(x, 1)
    elif arch=='conv_ae1D':
        x = torch.squeeze(x, 1)
        x = x.permute((0, 2, 1)) 
    
    return x, shift

def read_dataset_conv(dataset_name, scaled,columns_subset, dataset_subset, cols, 
                 train_val_split, sequence_length, out_window,  predict=False,
                 batch_size=4, transform=None, test=False):
    
    target=None
    xdf = pd.read_pickle(dataset_name)

    if columns_subset:
        cols = cols[:columns_subset]
    dataRaw = xdf[cols].dropna()

    if dataset_subset and not test:
        
        dataRaw = dataRaw.iloc[:dataset_subset, :]
        
        dfTrainVal = dataRaw .copy()
        xTrainVal = dfTrainVal.values
        
        if scaled:
            scaler = StandardScaler()
            x_train_val_scaled = scaler.fit_transform(xTrainVal)
            dfNorm = pd.DataFrame(x_train_val_scaled, columns=dataRaw.columns)

        else:
            df_train_val = pd.DataFrame(xTrainVal, columns=df.columns)
            dfNorm = pd.DataFrame(xTrainVal, columns=dataRaw.columns)
  
        X_train, X_test, y_train, y_test = train_test_split(dfNorm, dfNorm, \
                                                            train_size=train_val_split\
                                                            , shuffle=False)

        df_train = pd.DataFrame(X_train, columns=dfNorm.columns)
        df_test = pd.DataFrame(X_test, columns=dfNorm.columns)


        train_dataset = Dataset_seq(df_train, target = target, sequence_length = sequence_length,
                                    out_window = out_window, prediction=predict,\
                                    transform=transform)
        train_iter = DataLoader(dataset=train_dataset,\
                                batch_size=batch_size, shuffle=False)

        test_dataset = Dataset_seq(df_test, target = target, \
                                   sequence_length = sequence_length,
                                    out_window = out_window, prediction=predict,\
                                   transform=transform)
        test_iter = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        test_iter_loss = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        
        return df_train, df_test, train_iter, test_iter, test_iter_loss
        
        
    elif dataset_subset and test:
        dataRawTest = dataRaw.iloc[dataset_subset:, :]
        dataRawTrainVal = dataRaw.iloc[:dataset_subset, :]

        dfTrainVal = dataRawTrainVal.copy()
        xTrainVal = dfTrainVal.values

        dfTest = dataRawTest.copy()
        x_unscaled_test = dfTest.values

        if scaled:
            scaler = StandardScaler()
            x_train_val_scaled = scaler.fit_transform(xTrainVal)
            x_test = scaler.fit_transform(x_unscaled_test)
            df_train_val = pd.DataFrame(x_train_val_scaled, columns=dataRaw.columns)
            df_test = pd.DataFrame(x_test, columns=dataRaw.columns)

        else:
            df_test = pd.DataFrame(x_unscaled_test, columns=dataRaw.columns)
            df_train_val = pd.DataFrame(xTrainVal, columns=dataRaw.columns)


        train_dataset = Dataset_seq(df_train_val, target = target, sequence_length = sequence_length,
                                    out_window = out_window, prediction=predict,\
                                    transform=transform)
        train_iter = DataLoader(dataset=train_dataset,\
                                batch_size=batch_size, shuffle=False)

        test_dataset = Dataset_seq(df_test, target = target, \
                                   sequence_length = sequence_length,
                                    out_window = out_window, prediction=predict,\
                                   transform=transform)
        test_iter = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        test_iter_loss = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        
        #dataset_size = len(df_test)
        #idxs = np.arange(0, dataset_size, sequence_length)
        
        #test_sampler = SubsetRandomSampler(idxs)
        
        #test_iter_loss = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler)

        return df_train_val, df_test, train_iter, test_iter, test_iter_loss



def show_results(x, yo, cols, model_name, params_conf, par_nums, shift, 
                 num_batch, arch='conv_ae2D', save=False):
    
    path = './figure_results/evaluation/{}/{}/shift_{}/'.format(arch, model_name, shift, num_batch)
    
    if arch=='conv_ae':
        for i in range(x.shape[2]):
            fig, ax = plt.subplots(1,1, figsize=(8,4))

            ax.plot(yo.flatten(0,1)[:,i].to("cpu").detach().numpy(), 
                       linestyle='--', label='reconstr', color='black')
            ax.plot(x.flatten(0,1)[:,i].to("cpu").detach().numpy(), label ='original')
            x_i = x.flatten(0,1)[:,i]
            y_o = yo.flatten(0,1)[:,i]

            #loss_u = torch.nn.L1Loss(reduction='none')(y_o, x_i)
            #loss_u = moving_average(loss_u.to("cpu").detach().numpy(), params_conf['sequence_length'])
            #ax.plot(loss_u, label ='mean abs err')

            ax.set_xlabel('time steps')
            ax.set_ylabel('{}'.format(cols[i]))
            ax.set_title('Input Vs Recon. Par num {} s_l {}'\
                            .format(par_nums, 
                                    params_conf['sequence_length'], 
                                    ))
            ax.legend()
            if save:
                os.makedirs(path, exist_ok=True)
                plt.savefig(path + '{}_shift_{}_batch_{}.png'.format(cols[i], num_batch))

    elif arch=='conv_ae1D':
        for i in range(x.shape[1]):
            fig, ax = plt.subplots(1,1, figsize=(8,4))
            ax.plot(yo[:,i,:].flatten().to("cpu").detach().numpy(), 
                       linestyle='--', label='reconstr', color='black')
            ax.plot(x[:,i,:].flatten().to("cpu").detach().numpy(), label ='original')
            ax.set_xlabel('time steps')
            ax.set_ylabel('{}'.format(cols[i]))
            ax.set_title('Input Vs Recon. Par num {} s_l {}'\
                            .format(par_nums, params_conf['sequence_length']))

            ax.legend()
            if save:
                os.makedirs(path, exist_ok=True)
                plt.savefig(path + '{}_shift_{}_batch_{}.png'.format(cols[i], num_batch))
