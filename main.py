import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from dataset.dataset_custom import *
from models.ae import *
from models.vae import *
from config import *
import argparse

def main(args1, args2):
    xdf = pd.read_pickle(os.path.join(args2.data_path, args2.dataset))

    if args1.columns_subset:
        args1.columns = args1.columns[:args1.columns_subset]
    dataRaw = xdf[args1.columns].dropna()

    if args1.dataset_subset:
        dataRaw = dataRaw.iloc[:args1.dataset_subset, :]

    df = dataRaw.copy()
    x = df.values

    param_conf = args1.__dict__
    param_conf.update(args2.__dict__)

    if args1.scaled:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        dfNorm = pd.DataFrame(x_scaled, columns=df.columns)
    else:
        dfNorm = pd.DataFrame(x, columns=df.columns)

    X_train, X_test, y_train, y_test = train_test_split(dfNorm, dfNorm, train_size=args1.train_val_split, shuffle=False)
    df_train = pd.DataFrame(X_train, columns=dfNorm.columns)
    df_test = pd.DataFrame(X_test, columns=dfNorm.columns)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    train_dataset = Dataset_seq(df_train, target = args1.target, sequence_length = args1.sequence_length,
                                out_window = args1.out_window, prediction=args1.predict)
    train_iter = DataLoader(dataset=train_dataset, batch_size=args1.batch_size, shuffle=True)

    test_dataset = Dataset_seq(df_test, target = args1.target, sequence_length = args1.sequence_length,
                                out_window = args1.out_window, prediction=args1.predict)
    test_iter = DataLoader(dataset=test_dataset, batch_size=args1.batch_size, shuffle=True)

    if args1.scaled:
        torch.save(train_iter, './dataloader/train_dataloader_{}_ft_{}.pth'.format(len(args1.columns), args1.sampling_rate))
        torch.save(train_iter, './dataloader/test_dataloader_{}_ft_{}.pth'.format(len(args1.columns), args1.sampling_rate))
    else:
        torch.save(train_iter, './dataloader/train_dataloader_not_scaled_{}_ft_{}.pth'.format(len(args1.columns), args1.sampling_rate))
        torch.save(train_iter, './dataloader/test_dataloader_not_scaled_{}_ft_{}.pth'.format(len(args1.columns), args1.sampling_rate))

    if args1.target != None:
        n_features = len(args1.columns) - len(args1.target)
    else:
        n_features = len(args1.columns)
        target = args1.columns

    param_conf.update({'n_features':n_features,
                       'output_size':len(target)})

    if args1.architecture == "ae":
        model = LSTM_AE(seq_in=args1.sequence_length, seq_out= args1.out_window, n_features=n_features,
                        output_size=len(target), embedding_dim=args1.embedding_dim, latent_dim=args1.latent_dim,
                        n_layers=args1.n_layers).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args1.lr)
        train_ae(param_conf, train_iter, test_iter, model, criterion, optimizer, device,
              out_dir =args1.model_path, model_name= args2.model_name, epochs = args1.epochs)
    else:
        model = LSTM_VAE(seq_in=args1.sequence_length, seq_out= args1.out_window, no_features=n_features,
                        output_size=len(target), embedding_dim=args1.embedding_dim, latent_dim=args1.latent_dim,
                        Nf_lognorm=n_features, Nf_binomial=args1.N_binomial, n_layers=args1.n_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args1.lr)
        train_vae(n_features, train_iter, test_iter, model, optimizer,
                  device, args1.model_path, model_name= args2.model_name,  Nf_lognorm=None, Nf_binomial=None, epochs=args1.epochs)


if __name__ == '__main__':

    parser1 = argparse.ArgumentParser()
    parser1.add_argument("--architecture", default='ae', help="ae or vae")
    parser1.add_argument("--columns", default=columns, help="columns imported from config")
    parser1.add_argument("--model_path", default=model_results, help="where to save model")
    parser1.add_argument("--train_val_split", default=0.80, help="a number to specify how many feats to take from columns")
    parser1.add_argument("--columns_subset", default=0, help="a number to specify how many feats to take from columns")
    parser1.add_argument("--dataset_subset", default=100000, help="number of row to use from all the dataset")
    parser1.add_argument("--batch_size", default=10, help="batch sizet")
    parser1.add_argument("--sequence_length", default=5, help="sequence_lenght")
    parser1.add_argument("--out_window", default=5, help="sequence lenght of the output")
    parser1.add_argument("--epochs", default=100, help="number of epochs")
    parser1.add_argument("--lr", default=0.003, help="number of epochs")
    parser1.add_argument("--n_layers", default=1, help="number of epochs")
    parser1.add_argument("--embedding_size", default=10, help="number of epochs")
    parser1.add_argument("--embedding_dim", default=64, help="number of epochs")
    parser1.add_argument("--latent_dim", default=5, help="number of epochs")
    parser1.add_argument("--N_binomial", default=1, help="number of epochs")
    parser1.add_argument("--target", default=None, help="columns name of the target if none >>> autoencoder mode")
    parser1.add_argument('--predict', action='store_const', const=False, default=False,
                        help='')
    parser1.add_argument('--scaled', action='store_const', const=False, default=True,
                        help='')
    parser1.add_argument("--sampling_rate", type=str, default="2s", help="[2s, 4s]")
    args1 = parser1.parse_args()

    parser2 = argparse.ArgumentParser()
    parser2.add_argument("--data_path", type=str, default=f'./data/FIORIRE/dataset_{args1.sampling_rate}/')
    parser2.add_argument("--dataset", default=f'all_2016-2018_clean_{args1.sampling_rate}.pkl', help="ae") #all_2016-2018_clean_std_{args1.sampling_rate}.pkl
    if args1.scaled:
        parser2.add_argument("--model_name", default='ae_{}_ft_{}_sc'.format(len(args1.columns), args1.sampling_rate),
                             help="ae")
    else:
        parser2.add_argument("--model_name", default='ae_{}_ft_{}'.format(len(args1.columns), args1.sampling_rate), help="ae")
    args2 = parser2.parse_args()


    main(args1, args2)
