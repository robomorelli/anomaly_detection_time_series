import os 
import numpy as np
from config import *

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def exp_name_folders(architecture='lstm_ae'):
    
    folder_list = os.listdir(model_results+ '{}/'.format(architecture))
    exclude_results = ['cell1_2', 'first_checkpoint', 'pr_1', 'pr_2', 'pr_2_grace']
    folder_list = [x for x in folder_list if x not in exclude_results]
    name_mapping = {'pr_6k':'6k', 'pr_12k':'12k', 'pr_13k':'13k', 'pr_20k':'20k'}
    exp_name = [name_mapping[x] for x in folder_list]
    
    return exp_name