#from colorama import init as colorama_init
#from colorama import Fore
#from colorama import Style

W  = '\033[0m'  # white (normal)
R  = '\033[31m' # red
G  = '\033[32m' # green
O  = '\033[33m' # orange
B  = '\033[34m' # blue
P  = '\033[35m' # purple

def launch_train_advice(cfg, model_list, config_file):
    if len(model_list) == 0:
        print('no model available, to train a model {} switch train_model to True in the next cell {}'.format(R, W))
        print('')
        print('check the configuration file in {} to set the hyperparameters of the model'.format(config_file))
        print('')
        print('actual dataset configuration is {}'.format(cfg.dataset))
        print('')
        print('actual architecture configuration is {}'.format(cfg.model))
        print('')
        print('actual opt configuration is {}'.format(cfg.opt))
    else:
        print('to train a model {} switch train_model to True {} in the next cell'.format(R, W))
        print('to train a new model from scratch set train_model to True in the next cell')
        print('')
        print('check the configuration file in {} to set the hyperparameters of the model'.format(config_file))
        print('')
        print('actual dataset configuration is {}'.format(cfg.dataset))
        print('')
        print('actual architecture configuration is {}'.format(cfg.model))
        print('')
        print('actual opt configuration is {}'.format(cfg.opt))