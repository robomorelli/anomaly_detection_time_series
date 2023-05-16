from models.lstm_ae import *
from models.lstm import *
from models.enc_dec_lstm import *
from models.lstm_vae1cell import *
from models.lstm_vae import *
from models.conv_ae import *
from models.conv_ae_1D import *
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from torchvision.transforms import transforms as T, Lambda

def get_name(cfg):

    now = datetime.now()
    print("now =", now)
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string = dt_string.replace('/', '_').replace(' ', '_')

    if 'lstm' in cfg.model.architecture:
        if 'vae' in cfg.model.architecture:
            if '1cell' in cfg.model.architecture:
                model_name = '{}_sl_{}_emb_{}_layers_{}_recon_loss_{}_{}'.format(cfg.model.architecture, cfg.dataset.sequence_length,
                                                                                cfg.model.embedding_dim,
                                                                                cfg.model.n_layers_1,
                                                                                cfg.model.recon_loss, dt_string)
            else:
                model_name = '{}_sl_{}_emb_{}_layers_{}_{}_recon_loss_{}_{}'.format(cfg.model.architecture,cfg.dataset.sequence_length,
                                                                                     cfg.model.embedding_dim,
                                                                                     cfg.model.n_layers_1, cfg.model.n_layers_2,
                                                                                     cfg.model.recon_loss, dt_string)
        elif 'enc_dec' in cfg.model.architecture:
            model_name = '{}_sl_{}_hs_{}_n_cells_{}_layers_{}_out_w_{}_{}'.format(cfg.model.architecture,
                                                                                cfg.dataset.sequence_length,
                                                                                cfg.model.hidden_size,
                                                                                cfg.model.n_cells,
                                                                                cfg.model.n_layers, cfg.dataset.out_window, dt_string)
        else:
            model_name = '{}_sl_{}_emb_{}_layers_{}_{}_out_w_{}_{}'.format(cfg.model.architecture,
                                                                                cfg.dataset.sequence_length,
                                                                                cfg.model.embedding_dim,
                                                                                cfg.model.n_layers_1,
                                                                                cfg.model.n_layers_2, cfg.dataset.out_window, dt_string)
    else:
        model_name = '{}_sl_{}_filter_n_{}_kernel_size_{}_{}'.format(cfg.model.architecture,
                                                                     cfg.dataset.sequence_length,
                                                                     cfg.model.filter_num, cfg.model.kernel_size,
                                                                     dt_string)
    return model_name

def get_transform(cfg):

    if cfg.model.architecture == "conv_ae":
        transform = T.Compose([
                               T.ToTensor(),
                               ])
    elif cfg.model.architecture == 'conv_ae1D':
        transform = T.Compose([
            T.ToTensor(),
            Lambda(lambda x: x.permute((0, 2, 1))),
            Lambda(lambda x: x.squeeze(0))
        ])
    else:
        transform = None
    return transform
def dataset_preprocessing(dataset_path, cfg, scaler=None):

    xdf = pd.read_pickle(dataset_path)

    if cfg.dataset.columns_subset:
        cfg.dataset.columns = cfg.dataset.columns[:cfg.dataset.columns_subset]
    dataRaw = xdf[cfg.dataset.columns].dropna()

    if cfg.dataset.dataset_subset:
        dataRaw = dataRaw.iloc[:cfg.dataset.dataset_subset, :]

    df = dataRaw.copy()
    x = df.values

    if scaler is not None:
        x_scaled = scaler.fit_transform(x)
        df_processed = pd.DataFrame(x_scaled, columns=df.columns)
    else:
        df_processed = pd.DataFrame(x, columns=df.columns)

    return df_processed

def dataset_split(df, cfg):

    X_train, X_test, y_train, y_test = train_test_split(df, df, train_size=cfg.dataset.train_val_split,
                                                            shuffle=False)
    return X_train, X_test, y_train, y_test

def start_train(cfg, param_conf, train_iter, test_iter, device, checkpoint_path, model_name):

    if cfg.model.architecture == "lstm_ae":
        model = LSTM_AE(seq_in=cfg.dataset.sequence_length, seq_out= cfg.dataset.sequence_length, n_features=len(cfg.dataset.columns),
                        output_size=len(cfg.dataset.target), embedding_dim=cfg.model.embedding_dim, latent_dim=cfg.model.latent_dim,
                        n_layers_1=cfg.model.n_layers_1, n_layers_2=cfg.model.n_layers_2, no_latent=cfg.model.no_latent).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=cfg.opt.lr_patience,
                                                               threshold=0.0001,
                                                               threshold_mode='rel', cooldown=0, min_lr=9e-8,
                                                               verbose=True)

        train_lstm_ae(param_conf, train_iter, test_iter, model, criterion, optimizer, scheduler, device,
              out_dir =checkpoint_path , model_name= model_name, epochs = cfg.opt.epochs, es_patience=cfg.opt.es_patience)

    if cfg.model.architecture == "lstm":
        model = LSTM(seq_in=cfg.dataset.sequence_length, seq_out= cfg.dataset.sequence_length, n_features=len(cfg.dataset.columns),
                        output_size=len(cfg.dataset.target), embedding_dim=cfg.model.embedding_dim,
                        n_layers_1=cfg.model.n_layers_1, n_layers_2=cfg.model.n_layers_2).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=cfg.opt.lr_patience,
                                                               threshold=0.0001,
                                                               threshold_mode='rel', cooldown=0, min_lr=9e-8,
                                                               verbose=True)

        train_lstm(param_conf, train_iter, test_iter, model, criterion, optimizer, scheduler, device,
              out_dir =checkpoint_path , model_name= model_name, epochs = cfg.opt.epochs, es_patience=cfg.opt.es_patience)

    if cfg.model.architecture == "enc_dec_lstm":

        model = ENC_DEC_LSTM(seq_in=cfg.dataset.sequence_length, seq_out= cfg.dataset.out_window, input_size=len(cfg.dataset.columns),
                        hidden_size=cfg.model.hidden_size,
                        n_layers=cfg.model.n_layers, n_cells=cfg.model.n_cells).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=cfg.opt.lr_patience,
                                                               threshold=0.0001,
                                                               threshold_mode='rel', cooldown=0, min_lr=9e-8,
                                                               verbose=True)

        train_enc_dec_lstm(param_conf, train_iter, test_iter, model, criterion, optimizer, scheduler, device,
              out_dir =checkpoint_path , model_name= model_name, epochs = cfg.opt.epochs, es_patience=cfg.opt.es_patience)

    elif cfg.model.architecture == "lstm_vae1cell":
        model = LSTM_VAE1cell(seq_in=cfg.dataset.sequence_length, seq_out= cfg.dataset.out_window, no_features=len(cfg.dataset.columns),
                        output_size=len(cfg.dataset.target), embedding_dim=cfg.model.embedding_dim, latent_dim=cfg.model.latent_dim,
                        n_layers_1=cfg.model.n_layers_1,
                        Nf_lognorm=cfg.model.Nf_lognorm, Nf_binomial=cfg.model.Nf_binomial,
                         kld_type=cfg.model.kld, recon_loss_type=cfg.model.recon_loss, batch_size=cfg.dataset.batch_size).to(device)
        criterion = None
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=cfg.opt.lr_patience,
                                                               threshold=0.0001,
                                                               threshold_mode='rel', cooldown=0, min_lr=9e-8,
                                                               verbose=True)
        train_lstm_vae1cell(param_conf, len(cfg.dataset.columns), train_iter, test_iter, model, criterion, optimizer, scheduler,
                  device, out_dir =checkpoint_path , model_name= model_name, epochs = cfg.opt.epochs, Nf_lognorm=cfg.model.Nf_lognorm, Nf_binomial=cfg.model.Nf_binomial
                       , kld_factor = cfg.opt.kld_factor, recon_loss_type=cfg.model.recon_loss, kld_type=cfg.model.kld, es_patience=cfg.opt.es_patience)

    elif cfg.model.architecture == "lstm_vae":
        model = LSTM_VAE(seq_in=cfg.dataset.sequence_length, seq_out= cfg.dataset.out_window, no_features=len(cfg.dataset.columns),
                        output_size=len(cfg.dataset.target), embedding_dim=cfg.model.embedding_dim, latent_dim=cfg.model.latent_dim,
                         Nf_lognorm=cfg.model.Nf_lognorm, Nf_binomial=cfg.model.Nf_binomial,
                         n_layers_1=cfg.model.n_layers_1, n_layers_2=cfg.model.n_layers_2, recon_loss_type=cfg.model.recon_loss,
                         kld_type=cfg.model.kld).to(device)
        criterion = None
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=cfg.opt.lr_patience,
                                                               threshold=0.0001,
                                                               threshold_mode='rel', cooldown=0, min_lr=9e-8,
                                                               verbose=True)
        train_lstm_vae(param_conf, len(cfg.dataset.columns), train_iter, test_iter, model, criterion, optimizer, scheduler,
                  device, out_dir =checkpoint_path , model_name= model_name, epochs = cfg.opt.epochs,
                       Nf_lognorm=cfg.model.Nf_lognorm, Nf_binomial=cfg.model.Nf_binomial,
                       recon_loss_type=cfg.model.recon_loss, kld_type=cfg.model.kld, kld_factor = 1, es_patience=cfg.opt.es_patience)

    elif cfg.model.architecture == "conv_ae":
        model = CONV_AE(in_channel=1,  heigth=cfg.dataset.sequence_length, width=len(cfg.dataset.columns),
                        kernel_size=cfg.model.kernel_size, filter_num=cfg.model.filter_num,
                        latent_dim=cfg.model.latent_dim, n_layers=cfg.model.n_layers, activation = param_conf['activation'],
                        increasing=cfg.model.increasing, flattened=cfg.model.flattened).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=cfg.opt.lr_patience,
                                                               threshold=0.0001,
                                                               threshold_mode='rel', cooldown=0, min_lr=9e-8,
                                                               verbose=True)
        criterion = nn.MSELoss()

        train_conv_ae(param_conf, train_iter, test_iter, model, criterion, optimizer, scheduler, device,
              out_dir =checkpoint_path , model_name= model_name, epochs = cfg.opt.epochs, es_patience=cfg.opt.es_patience)

    elif cfg.model.architecture == "conv_ae1D":
        model = CONV_AE1D(in_channel=cfg.dataset.n_features, length=cfg.dataset.sequence_length,
                          kernel_size=cfg.model.kernel_size, filter_num=cfg.model.filter_num, stride=cfg.model.stride,pool=cfg.model.pool,
                          latent_dim=cfg.model.latent_dim, n_layers=cfg.model.n_layers, activation=param_conf['activation'], bn=cfg.model.bn
                          ,increasing=cfg.model.increasing, flattened=cfg.model.flattened).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=cfg.opt.lr_patience,
                                                               threshold=0.0001,
                                                               threshold_mode='rel', cooldown=0, min_lr=9e-8,
                                                               verbose=True)
        criterion = nn.MSELoss()
        train_conv_ae1D(param_conf, train_iter, test_iter, model, criterion, optimizer, scheduler, device,
                        out_dir=checkpoint_path, model_name=model_name, epochs=cfg.opt.epochs, es_patience=cfg.opt.es_patience)
