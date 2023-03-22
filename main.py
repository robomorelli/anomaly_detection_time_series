import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from dataset.sentinel import *
from config import *
import argparse
from omegaconf import OmegaConf
from omegaconf import DictConfig
import click
import yaml
import json
import types
from utils.training import start_train, dataset_preprocessing, dataset_split, get_name, get_transform

def load_object(dct):
    return types.SimpleNamespace(**dct)

#@click.command()
#@click.option('-c', '--config-name', type=click.Path())
def main(config_name):

    cfg = OmegaConf.load('configuration/{}.yaml'.format(args.config_name))
    cfg.dataset.out_window = cfg.dataset.sequence_length
    cfg.dataset.data_path = os.path.join(data_path, 'dataset_{}'.format(cfg.dataset.sampling_rate))

    model_name = get_name(cfg)
    transform = get_transform(cfg)

    try:
        sm = str(torch.cuda.get_device_capability())
        sm = ''.join((sm.strip('()').split(',')[0], sm.strip('()').split(',')[1])).replace(' ', '')
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda and sm in torch.cuda.get_arch_list() else "cpu")
    except:
        device='cpu'

    if cfg.dataset.scaled:
        scaler = StandardScaler()

    dataset_path = os.path.join(cfg.dataset.data_path, cfg.dataset.name)
    df_processed = dataset_preprocessing(dataset_path, cfg, scaler)
    X_train, X_test, y_train, y_test = dataset_split(df_processed, cfg)

    df_train = pd.DataFrame(X_train, columns=cfg.dataset.columns)
    df_test = pd.DataFrame(X_test, columns=cfg.dataset.columns)

    if cfg.dataset.target != None:
        cfg.dataset.n_features = len(cfg.dataset.columns) - len(cfg.dataset.target)
    else:
        cfg.dataset.n_features = len(cfg.dataset.columns)
        cfg.dataset.target = cfg.dataset.columns

    if 'vae' in cfg.model.architecture:
        cfg.model.Nf_binomial = cfg.dataset.n_features - cfg.model.Nf_lognorm

    # TODO: make get param dict function
    act_dict = {'relu': nn.ReLU(), 'elu': nn.ELU()}
    cfg_dict = cfg.__dict__['_content']
    param_conf = {}
    for k1 in list(cfg_dict.keys()):
        for k2 in list(cfg_dict[k1].keys()):
            param_conf[k2] = cfg_dict[k1][k2]
    try:
        act = cfg.model.activation.lower()
        param_conf['activation'] = act_dict[act]
    except:
        if 'lstm' in cfg.model.architecture:
            pass
        else:
            print('attention, activation is not present in the model')

    train_dataset = Dataset_seq(df_train, target =cfg.dataset.target, sequence_length = cfg.dataset.sequence_length,
                                out_window = cfg.dataset.out_window, prediction=cfg.dataset.predict, forecast_all=cfg.dataset.forecast_all,
                                transform=transform)
    train_iter = DataLoader(dataset=train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=12)

    test_dataset = Dataset_seq(df_test, target = cfg.dataset.target, sequence_length = cfg.dataset.sequence_length, forecast_all=cfg.dataset.forecast_all,
                                out_window = cfg.dataset.out_window, prediction=cfg.dataset.predict, transform=transform)
    test_iter = DataLoader(dataset=test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, num_workers=12)

    checkpoint_path = os.path.join(model_results, cfg.model.architecture)
    start_train(cfg, param_conf, train_iter, test_iter, device, checkpoint_path, model_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Define parameters for crop.')
    parser.add_argument('--config_name', nargs="?", default='lstm_vae',
                        help='the folder including the images to crop')
    args = parser.parse_args()
    main(args)