import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from dataset.sentinel import *
from models.lstm_ae import *
from models.lstm import *
from models.lstm_vae import *
from models.lstm_vae_vanilla import *
from models.conv_ae import *
from models.conv_ae_1D import *
from config import *
import argparse
from torchvision.transforms import transforms as T, Lambda
import platform

def main(args1, args2):
    xdf = pd.read_pickle(os.path.join(args2.data_path, args2.dataset))

    if platform.node() == 'leonard':
        device = 'cpu'
    else:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

    if args1.columns_subset:
        args1.columns = args1.columns[:args1.columns_subset]
    dataRaw = xdf[args1.columns].dropna()

    if args1.dataset_subset:
        dataRaw = dataRaw.iloc[:args1.dataset_subset, :]

    df = dataRaw.copy()
    x = df.values

    if args1.scaled:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        dfNorm = pd.DataFrame(x_scaled, columns=df.columns)
    else:
        dfNorm = pd.DataFrame(x, columns=df.columns)

    # If shuffle is True the sequences are taken from different time and when testing are merged with noisy effect:
    # two contigous sequences (maybe distant in time) are linked with the line of matplotlib making a noiysy effect
    if args1.shuffle:
        X_train, X_test, y_train, y_test = train_test_split(dfNorm, dfNorm, train_size=args1.train_val_split, shuffle=True,
                                                            random_state=123)
    else:
        X_train, X_test, y_train, y_test = train_test_split(dfNorm, dfNorm, train_size=args1.train_val_split, shuffle=False)

    df_train = pd.DataFrame(X_train, columns=dfNorm.columns)
    df_test = pd.DataFrame(X_test, columns=dfNorm.columns)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if args1.architecture == "conv_ae":
        transform = T.Compose([
                               T.ToTensor(),
                               ])
    elif args1.architecture == 'conv_ae1D':
        transform = T.Compose([
            T.ToTensor(),
            Lambda(lambda x: x.permute((0, 2, 1))),
            Lambda(lambda x: x.squeeze(0))

        ])

    else:
        transform = None

    param_conf = args1.__dict__
    param_conf.update(args2.__dict__)

    train_dataset = Dataset_seq(df_train, target = args1.target, sequence_length = args1.sequence_length,
                                out_window = args1.out_window, prediction=args2.predict, forecast_all=args2.forecast_all,
                                transform=transform)
    train_iter = DataLoader(dataset=train_dataset, batch_size=args1.batch_size, shuffle=True)

    #######################################################
    # Check the index sampled with shuffle false or true:
    # list(train_iter._index_sampler.sampler.__iter__())
    #######################################################
    test_dataset = Dataset_seq(df_test, target = args1.target, sequence_length = args1.sequence_length, forecast_all=args2.forecast_all,
                                out_window = args1.out_window, prediction=args2.predict, transform=transform)
    test_iter = DataLoader(dataset=test_dataset, batch_size=args1.batch_size, shuffle=False)

    if 'conv' not in args1.architecture:
        if args1.scaled:
            if not args1.shuffle:
                torch.save(train_iter, './dataloader/train_dataloader_{}_ft_{}_{}.pth'.format(len(args1.columns), args1.sampling_rate, args1.sequence_length))
                torch.save(test_iter, './dataloader/test_dataloader_{}_ft_{}_{}.pth'.format(len(args1.columns), args1.sampling_rate, args1.sequence_length))
            else:
                torch.save(train_iter, './dataloader/train_dataloader_{}_ft_{}_{}_shuffle.pth'.format(len(args1.columns), args1.sampling_rate, args1.sequence_length))
                torch.save(test_iter, './dataloader/test_dataloader_{}_ft_{}_{}_shuffle.pth'.format(len(args1.columns), args1.sampling_rate, args1.sequence_length))
        else:
            if not args1.shuffle:
                torch.save(train_iter, './dataloader/train_dataloader_not_scaled_{}_ft_{}_{}.pth'.format(len(args1.columns), args1.sampling_rate, args1.sequence_length))
                torch.save(test_iter, './dataloader/test_dataloader_not_scaled_{}_ft_{}_{}.pth'.format(len(args1.columns), args1.sampling_rate, args1.sequence_length))
            else:
                torch.save(train_iter, './dataloader/train_dataloader_not_scaled_{}_ft_{}_{}_shuffle.pth'.format(len(args1.columns), args1.sampling_rate, args1.sequence_length))
                torch.save(test_iter, './dataloader/test_dataloader_not_scaled_{}_ft_{}_{}_shuffle.pth'.format(len(args1.columns), args1.sampling_rate, args1.sequence_length))

    if args1.target != None:
        n_features = len(args1.columns) - len(args1.target)
    else:
        n_features = len(args1.columns)
        target = args1.columns

    param_conf.update({'n_features':n_features,
                       'output_size':len(target)})

    checkpoint_path = os.path.join(args1.model_path, args1.architecture)

    if args1.architecture == "lstm_ae":
        model = LSTM_AE(seq_in=args1.sequence_length, seq_out= args1.out_window, n_features=n_features,
                        output_size=len(target), embedding_dim=args1.embedding_dim, latent_dim=args1.latent_dim,
                        n_layers_1=args1.n_layers_1, n_layers_2=args1.n_layers_2, no_latent=args1.no_latent).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args1.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

        train_lstm_ae(param_conf, train_iter, test_iter, model, criterion, optimizer, scheduler, device,
              out_dir =checkpoint_path , model_name= args2.model_name, epochs = args1.epochs)

    if args1.architecture == "lstm":
        model = LSTM(seq_in=args1.sequence_length, seq_out= args1.out_window, n_features=n_features,
                        output_size=len(target), embedding_dim=args1.embedding_dim,
                        n_layers_1=args1.n_layers_1, n_layers_2=args1.n_layers_2).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args1.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

        train_lstm(param_conf, train_iter, test_iter, model, criterion, optimizer, scheduler, device,
              out_dir =checkpoint_path , model_name= args2.model_name, patience=args1.patience, epochs = args1.epochs)


    elif args1.architecture == "lstm_vae":
        model = LSTM_VAE(seq_in=args1.sequence_length, seq_out= args1.out_window, no_features=n_features,
                        output_size=len(target), embedding_dim=args1.embedding_dim, latent_dim=args1.latent_dim,
                        Nf_lognorm=n_features, Nf_binomial=args1.N_binomial, n_layers=args1.n_layers).to(device)
        criterion = None
        optimizer = torch.optim.Adam(model.parameters(), lr=args1.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
        train_lstm_vae(param_conf, n_features, train_iter, test_iter, model, criterion, optimizer, scheduler,
                  device, out_dir =checkpoint_path , model_name= args2.model_name, epochs = args1.epochs,
                       Nf_lognorm=None, Nf_binomial=None, kld_factor = 1)

    elif args1.architecture == "lstm_vae_vanilla":
        model = LSTM_VAEV(seq_in=args1.sequence_length, seq_out= args1.out_window, no_features=n_features,
                        output_size=len(target), embedding_dim=args1.embedding_dim, latent_dim=args1.latent_dim,
                         n_layers=args1.n_layers).to(device)
        #criterion = nn.MSELoss()
        criterion = None
        optimizer = torch.optim.Adam(model.parameters(), lr=args1.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

        train_lstm_vae_vanilla(param_conf, n_features, train_iter, test_iter, model, criterion, optimizer, scheduler,
                  device, out_dir =checkpoint_path , model_name= args2.model_name, epochs = args1.epochs,
                Nf_lognorm=None, Nf_binomial=None, kld_factor = 0.01)

    elif args1.architecture == "conv_ae":
        model = CONV_AE(in_channel=1,  heigth=args1.sequence_length, width=len(args1.columns),
                        kernel_size=args1.kernel_size, filter_num=args1.filter_num,
                 latent_dim=args1.latent_dim, n_layers=args1.n_layers, activation = args1.activation).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args1.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.001, gamma=0.5)
        criterion = nn.MSELoss()

        train_conv_ae(param_conf, train_iter, test_iter, model, criterion, optimizer, scheduler, device,
              out_dir =checkpoint_path , model_name= args2.model_name, epochs = args1.epochs)

    elif args1.architecture == "conv_ae1D":
        model = CONV_AE1D(in_channel=len(df_train.columns), length=args1.sequence_length,
                        kernel_size=args1.kernel_size, filter_num=args1.filter_num,
                 latent_dim=args1.latent_dim, n_layers=args1.n_layers, activation = args1.activation).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args1.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
        criterion = nn.MSELoss()
        train_conv_ae1D(param_conf, train_iter, test_iter, model, criterion, optimizer, scheduler, device,
              out_dir =checkpoint_path , model_name= args2.model_name, epochs = args1.epochs)



if __name__ == '__main__':

    parser1 = argparse.ArgumentParser()
    parser1.add_argument("--architecture", default='conv_ae', help="[lstm, lstm_ae, lstm_vae,"
                                                                " lstm_vae_vanilla, conv_ae, conv_ae1D,"
                                                                " conv_vae]")
    parser1.add_argument("--columns", default=columns, help="columns imported from config, [columns, columns_third_wheel]")
    parser1.add_argument("--model_path", default=model_results, help="where to save model")
    parser1.add_argument("--train_val_split", default=0.80, help="a number to specify how many feats to take from columns")
    parser1.add_argument('--shuffle', action='store_const', const=False, default=False, help='')
    parser1.add_argument("--columns_subset", default=0, help="a number to specify how many feats to take from columns")
    parser1.add_argument("--dataset_subset", default=500000, help="number of row to use from all the dataset")
    parser1.add_argument("--batch_size", default=500, help="batch size")

    parser1.add_argument("--epochs", default=50, help="ns")
    parser1.add_argument("--patience", default=5, help="ns")
    parser1.add_argument("--lr", default=0.0009, help="nus")
    parser1.add_argument("--out_window", default=7, help="sequence lenght of the output")
    parser1.add_argument("--sequence_length", default=16, help="sequence_length")

    # conv architecture
    parser1.add_argument("--n_layers", default=2, help="")
    parser1.add_argument("--kernel_size", default=3, help="")
    parser1.add_argument("--filter_num", default=32, help="")
    parser1.add_argument("--activation", default=nn.ReLU(), help="")

    # lstm architecture
    parser1.add_argument("--embedding_dim", default=32, help="s")
    parser1.add_argument("--n_layers_1", default=2, help="")
    parser1.add_argument("--n_layers_2", default=2, help="")
    parser1.add_argument("--no_latent",  action='store_const', const=False, default=False)
    parser1.add_argument("--latent_dim", default=50, help="")

    parser1.add_argument("--N_binomial", default=1, help="number of epochs")
    parser1.add_argument("--target", default=None, help="columns name of the target if none >>> autoencoder mode")
    parser1.add_argument("--sampling_rate", type=str, default="4s", help="[2s, 4s]")
    parser1.add_argument('--scaled', action='store_const', const=False, default=True,
                        help='')
    parser1.add_argument('--forecast_all', action='store_true',
                        help='')
    args1 = parser1.parse_args()

    parser2 = argparse.ArgumentParser()
    parser2.add_argument("--data_path", type=str, default=f'./data/FIORIRE/dataset_{args1.sampling_rate}/')
    parser2.add_argument("--dataset", default=f'all_2016-2018_clean_std_{args1.sampling_rate}.pkl', help="ae") #all_2016-2018_clean_std_{args1.sampling_rate}.pkl
    parser2.add_argument('--predict', action='store_true',
                        help='')
    parser2.add_argument('--forecast', action='store_true',
                        help='')
    parser2.add_argument('--forecast_all', action='store_true',
                        help='')


    n_features = args1.columns_subset if args1.columns_subset != 0 else len(columns)

    if args1.scaled and 'lstm' in args1.architecture:
        parser2.add_argument("--model_name", default='{}_sl_{}_emb_{}_layers_{}_{}_sc'.format(args1.architecture
                                                                                 , args1.sequence_length
                                                                                 , args1.embedding_dim
                                                                                 ,args1.n_layers_1
                                                                                 ,args1.n_layers_2),
                             help="ae")
    elif 'lstm' in args1.architecture:
        parser2.add_argument("--model_name", default='{}_sl_{}_emb_{}'.format(args1.architecture, args1.sequence_length, args1.embedding_dim), help="ae")

    else:
        parser2.add_argument("--model_name", default='{}_sl_{}_fn_{}_ks_{}'.format(args1.architecture, args1.sequence_length,
                                                                                   args1.filter_num, args1.kernel_size), help="ae")

    args2 = parser2.parse_args()

    args1.out_window = args1.sequence_length
    main(args1, args2)
    # Recurrent ISSUE:
    # When size in the loss between yo and x does not match the reason is the input size that when reduced with conv layer end witha  dimension
    # that is not equal to input size >>> sequence length X features have to be 16X16
