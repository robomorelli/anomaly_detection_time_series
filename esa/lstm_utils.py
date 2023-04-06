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

def read_dataset(dataset_name, scaled,columns_subset, dataset_subset, cols, 
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
            df_train_val = pd.DataFrame(xTrainVal, columns=df.columns)


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
        test_iter_loss = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        return df_train_val, df_test, train_iter, test_iter, test_iter_loss


def unrolling_batches(num_batch, test_iter, sequence_length, cols, shift=None):
    if shift==None:
        shift = np.random.randint(len(test_iter.dataset.df_data)-100)
    x = np.zeros((num_batch, sequence_length, len(cols)))

    for i in range(num_batch):
        x[i,:,:] = test_iter.dataset.df_data.iloc[shift + i*sequence_length:shift + (i+1)*sequence_length,:]\
        .values

    x = torch.from_numpy(x).float()
    print('random shift', shift)
    return x, shift


def show_results(x, yo, cols, model_name, params_conf, par_nums, shift, 
                 num_batch, architecture='lstm_ae', save=False):
    
    if architecture == 'lstm':
        yo.flatten(0,1)[params_conf['sequence_length']:,:]=yo.flatten(0,1)[:-params_conf['sequence_length'],:].clone()
        yo.flatten(0,1)[:params_conf['sequence_length'],:]=0
    
    path = './figure_results/evaluation/{}/{}/shift_{}/'.format(architecture, model_name, shift, num_batch)
    for i in range(x.shape[2]):
        fig, ax = plt.subplots(1,1, figsize=(8,4))

        ax.plot(yo.flatten(0,1)[:,i].to("cpu").detach().numpy(), 
                   linestyle='--', label='reconstr', color='black')
        ax.plot(x.flatten(0,1)[:,i].to("cpu").detach().numpy(), label ='original')
        
        if architecture=='lstm':
            ax.plot(x.flatten(0,1)[:params_conf['sequence_length'],i].to("cpu").detach().numpy()
            , color = 'orange')
            ax.plot(yo.flatten(0,1)[:params_conf['sequence_length'],i].to("cpu").detach().numpy()
            , label ='warm-up', color = 'orange')
        
        x_i = x.flatten(0,1)[:,i]
        y_o = yo.flatten(0,1)[:,i]

        loss_u = torch.nn.L1Loss(reduction='none')(y_o, x_i)
        loss_u = moving_average(loss_u.to("cpu").detach().numpy(), params_conf['sequence_length'])

        ax.plot(loss_u, label ='mean abs err')
        ax.set_xlabel('time steps')
        ax.set_ylabel('{}'.format(cols[i]))
        if architecture=='lstm_ae':
            try:
                ax.set_title('Input Vs Recon. Par num {} s_l {}'\
                                .format(par_nums, params_conf['sequence_length'], 
                                        #params_conf['latent_dim'], 
                                        #params_conf['n_layers_1'],params_conf['n_layers_2'],
                                        #params_conf['embedding_dim'],
                                        ))
            except:
                ax.set_title('Input Vs Recon. Par num {} s_l'\
                .format(par_nums, params_conf['sequence_length'], 
                        #params_conf['latent_dim'], 
                        #params_conf['n_layers'],params_conf['n_layers'],
                        #params_conf['embedding_dim'],
                        ))
        else:
            try:
                ax.set_title('Input Vs Recon. Par num {} s_l {}'\
                                .format(par_nums, params_conf['sequence_length']))
                                        #params_conf['n_layers'],params_conf['n_layers'],
                                        #params_conf['embedding_dim'],))    
            except:
                ax.set_title('Input Vs Recon. Par num {} s_l {}'\
                                .format(par_nums, params_conf['sequence_length']))
                                         #params_conf['n_layers_1'],params_conf['n_layers_2'],
                                        #params_conf['embedding_dim']))     
        ax.legend()
        if save:
            os.makedirs(path, exist_ok=True)
            plt.savefig(path + '{}_shift_{}_batch_{}.png'.format(cols[i], num_batch))
            

def unrolling_batches_with_anomalies(num_batch, test_iter, sequence_length, cols
                                     , sigma=5, feats=[0], batch = [2]):
    
    shift = np.random.randint(len(test_iter.dataset.df_data)-1000)
    x = np.zeros((num_batch, sequence_length, len(cols)))
    for i in range(num_batch):
        x[i,:,:] = test_iter.dataset.df_data.iloc[shift + i*sequence_length:shift + (i+1)*sequence_length,:]\
        .values

    xa = x.copy()
    for i in feats:
        for j in batch:
            xa[j,:,i] = xa[j,:,i] + sigma*np.std(xa[:,:,i])

    x = torch.from_numpy(x).float()
    xa = torch.from_numpy(xa).float()
    
    print('random shift', shift)
    return x, xa, shift


def show_results_anomalies(model_name, params_conf, par_nums, shift, feats = [0], save=False):
    
    path = './figure_results/{}/evaluation/{}/feats_{}/shift_{}/'.format(architecture, model_name, feats[0], shift)
    for i in range(x.shape[2]):
        fig, ax = plt.subplots(2,1, figsize=(12,16))

        ax[0].plot(yo.flatten(0,1)[:,i].to("cpu").detach().numpy(), 
                   linestyle='--', label='reconstr', color='black')
        ax[0].plot(x.flatten(0,1)[:,i].to("cpu").detach().numpy(), label ='original')
        x_i = x.flatten(0,1)[:,i]
        y_o = yo.flatten(0,1)[:,i]

        loss_u = torch.nn.L1Loss(reduction='none')(y_o, x_i)
        loss_u = moving_average(loss_u.to("cpu").detach().numpy(), seq_len)

        ax[0].plot(loss_u, label ='mean abs err')
        ax[0].set_xlabel('time steps')
        ax[0].set_ylabel('{}'.format(cols[i]))
        ax[0].set_title('Input Vs Recon. Par num {} s_l {} l_s {} n_l {}-{} emb_dim {}'\
                        .format(par_nums, params_conf['sequence_length'], 
                                params_conf['latent_dim'], 
                                params_conf['n_layers_1'],params_conf['n_layers_2'],
                                params_conf['embedding_dim'],
                                ))

        ax[0].legend()


        ax[1].plot(ya.flatten(0,1)[:,i].to("cpu").detach().numpy(), 
                   linestyle='--', label='reconstr', color='black')
        ax[1].plot(xa.flatten(0,1)[:,i].to("cpu").detach().numpy(), label ='original')

        x_i = xa.flatten(0,1)[:,i]
        y_o = ya.flatten(0,1)[:,i]

        loss_u = torch.nn.L1Loss(reduction='none')(y_o, x_i)
        loss_u = moving_average(loss_u.to("cpu").detach().numpy(), seq_len)
        filtered = np.concatenate((loss_u[0:(batch[0])*seq_len], loss_u[(batch[0]+1)*seq_len:]))


        ax[1].plot(loss_u, label ='mean abs err')
        ax[1].set_xlabel('time steps')
        ax[1].set_ylabel('{}'.format(cols[i]))
        ax[1].set_title('Injected anom. on {}'\
                        .format(cols[feats[0]]))

        ax[1].axvspan((batch[0])*seq_len, (batch[0]+1)*seq_len, alpha=0.2, color='red')
        ax[1].axhspan(np.mean(loss_u)-3*np.std(loss_u),
                      np.mean(loss_u)+3*np.std(loss_u), alpha=0.2, color='red', label='3 sigma interval')

        ax[1].axhspan(np.mean(filtered)-3*np.std(filtered),
                      np.mean(filtered)+3*np.std(filtered)
                      , alpha=0.2, color='green', label='3 sigma interval without anomaly')
        ax[1].legend()

        if save:
            os.makedirs(path, exist_ok=True)
            plt.savefig(path + '{}_shift_{}_{}_sigma_{}_batch_{}.png'.format(cols[i],shift,sigma,i, num_batch))
