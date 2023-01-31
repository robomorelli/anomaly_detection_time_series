import os 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import sys
from torch.utils.data import DataLoader
sys.path.append('..')
from dataset.sentinel import *
from config import *

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def exp_name_folders(path =esa_exploration, architecture='lstm_ae'):
    
    folder_list = os.listdir(path+ '{}/'.format(architecture))
    exclude_results = ['cell1_2', 'first_checkpoint', 'pr_1', 'pr_2', 'pr_2_grace']
    folder_list = [x for x in folder_list if x not in exclude_results]
    try:
        name_mapping = {'pr_6k':'6k', 'pr_12k':'12k', 'pr_13k':'13k', 'pr_20k':'20k'}
        exp_name = [name_mapping[x] for x in folder_list]
    except:
        exp_name = folder_list
        
    exp_dict = {}
    for i, en in enumerate(exp_name): 
        exp_dict[str(i)]=en
    
    return exp_dict

def select_by_rank(rank, model_results_path, summary):
    
    model_checkpoints = os.listdir(os.path.join(model_results_path,summary.iloc[rank]['name']))
    model_checkpoints = [x for x in model_checkpoints if x.startswith('checkpoint')]
    last_checkpoint = compute_last_checkpoint(model_checkpoints)
    model_name_path = os.path.join(summary.iloc[rank]['name'], last_checkpoint, 'model.pt')
    
    return model_name_path

def find_models(files, model_results_path):
    model_list = [x for x in files if x != 'summary.csv']
    if len([x for x in files if x == 'summary.csv']) > 0:
        summary_path = os.path.join(model_results_path + '/summary.csv')
        summary = pd.read_csv(summary_path).sort_values(by='val_loss')
        print('summary path:', summary_path)
    return model_list, summary

def compute_last_checkpoint(model_checkpoints):
    nums = [int(x.split('_')[1]) for x in model_checkpoints]
    nums.sort()
    last = str(nums[-1])

    if len(last) == 1:
        suffix = '00000' + str(last)
    elif len(last) == 2:
        suffix = '0000' + str(last)
    elif len(last) == 3:
        suffix = '000' + str(last)
    else:
        suffix = '00' + str(last)
    return 'checkpoint_' + suffix


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)