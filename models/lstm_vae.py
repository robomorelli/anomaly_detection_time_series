import os
import torch.nn as nn
import torch
from tqdm import tqdm
import sys
sys.path.append('..')
from utils.opt import EarlyStopping
from utils.layers import InverseSquareRootLinearUnit, ClippedTanh
from utils.losses import KL_loss, KL_loss_forVAE_custom, loss_function

torch.manual_seed(0)

class Encoder_vae(nn.Module):
    def __init__(self, seq_in, no_features, embedding_size, latent_dim
                 , n_layers_1=1, n_layers_2=1, kld_type='vanilla'):
        super().__init__()

        # self.seq_len = seq_in
        self.no_features = no_features  # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size  # the number of features in the embedded points of the inputs' number of features
        self.n_layers_1 = n_layers_1
        self.n_layers_2 = n_layers_2
        self.hidden_size = (2 * embedding_size)  # The number of features in the hidden state h
        self.latent_dim = latent_dim

        ''' 
        self.act2 = InverseSquareRootLinearUnit()
        self.ClippedTanh0 = ClippedTanh0()
        '''

        self.LSTMenc = nn.LSTM(
            input_size=no_features,
            hidden_size=self.hidden_size,
            num_layers=n_layers_1,
            batch_first=True
        )
        self.LSTM1 = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=embedding_size,
            num_layers=n_layers_2,
            batch_first=True
        )


        self.mu = nn.Linear(embedding_size, latent_dim)  # nn.Linear takes as input_size the last dimension of the tensor
        self.logvar = nn.Linear(embedding_size, latent_dim)
        #self.sigma = nn.Linear(embedding_size, latent_dim)

        ''' 
        self.h1_prior = nn.Linear(no_features, 1)
        self.h1_prior.weight.data.fill_(0)
        self.h1_prior.bias.data.fill_(1)

        self.mu_prior = nn.Linear(1,  latent_dim)  # nn.Linear takes as input_size the last dimension of the tensor
        self.mu_prior.weight.data.fill_(0)
        self.mu_prior.bias.data.fill_(0)

        self.sigma_prior_preActivation = nn.Linear(1, latent_dim)
        self.sigma_prior_preActivation.weight.data.fill_(0)
        self.sigma_prior_preActivation.bias.data.fill_(1)
        '''

        def _init_weights(self, module):
            if isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std =1.0)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.Linear):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)

    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        # x is the output and so the size is > (batch, seq_len, hidden_size)

        #fixed_input = self.ClippedTanh0(x)
        #fixed_input = fixed_input[:, -1, :]

        x, (hidden_state, cell_state) = self.LSTMenc(x)
        x, (hidden_state, cell_state) = self.LSTM1(x)
        last_lstm_layer_hidden_state = hidden_state[-1, :, :]

        # use the hidden state with size (batch, hidde_size=embedding_size) as input for the fc layer to output mu and sigma
        # fc layer input (input_size, output_size) but if you have (batch, seq_len, input_size) it takes the same operation batch by bath and for
        # each sequence

        mu = self.mu(last_lstm_layer_hidden_state)
        logvar = self.logvar(last_lstm_layer_hidden_state)
        #sigma = self.act2(self.sigma(last_lstm_layer_hidden_state))


        ########### TO IMPLEMENT???###############
        ########### TO IMPLEMENT???###############
        ########### TO IMPLEMENT???###############
        #hidden_2 = Dense(intermediate_dim, activation=act_fun, name='Encoder_h2')
        #aux = hidden_2(aux)
        ########### TO IMPLEMENT???###############
        ########### TO IMPLEMENT???###############
        ########### TO IMPLEMENT???###############


        '''
        with torch.no_grad():
            h1_prior = self.h1_prior(fixed_input)
        '''

        #x, (hidden_state, cell_state) = self.LSTMenc_prior(h1_prior)
        #x, (hidden_state, cell_state) = self.LSTM1_enc_prior(x)
        #last_lstm_layer_hidden_state = hidden_state[-1, :, :]

        ''' 
        mu_prior = self.mu_prior(h1_prior)

        sigma_prior_preActivation = self.sigma_prior_preActivation(h1_prior)
        sigma_prior = self.act2(sigma_prior_preActivation)
        '''

        return mu, logvar # mu, sigma, mu_prior, sigma_prior


# (2) Decoder
class Decoder_vae(nn.Module):
    def __init__(self, seq_out, embedding_size, output_size, latent_dim, n_layers_1=1, n_layers_2=1,
                 recon_loss_type='custom'):
        super().__init__()

        self.seq_len = seq_out
        self.embedding_size = embedding_size
        self.hidden_size = (2 * embedding_size)
        self.n_layers_1 = n_layers_1
        self.n_layers_2 = n_layers_2
        self.output_size = output_size
        self.latent_dim = latent_dim
        self.Nf_lognorm = output_size
        self.recon_loss_type = recon_loss_type

        self.act2 = InverseSquareRootLinearUnit()
        self.act3 = ClippedTanh()

        self.LSTMdec = nn.LSTM(
            input_size=latent_dim,
            hidden_size=embedding_size,
            num_layers=n_layers_1,
            batch_first=True
        )
        self.LSTM1 = nn.LSTM(
            input_size=embedding_size,
            hidden_size=self.hidden_size,
            num_layers=n_layers_2,
            batch_first=True
        )

        #clipper = WeightClipper()
        #self.LSTMdec.apply(clipper)

        #self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.par1 = nn.Linear(self.hidden_size, output_size)
        if self.recon_loss_type == 'custom':
            self.par2 = nn.Linear(self.hidden_size, self.Nf_lognorm)
            self.par3 = nn.Linear(self.hidden_size, self.Nf_lognorm)


    def forward(self, z):

        z = z.unsqueeze(1).repeat(1, self.seq_len, 1) #x[0,0,:]==x[0,1,:] ## we nedd to repeat to have an output of secquences (how is our target)
        x, (hidden_state, cell_state) = self.LSTMdec(z)
        x, (_, _) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))    #fc layer input (input_size, output_size) but if you have (batch, seq_len, input_size) it takes the same operation batch by bath and for
        # each sequence
        # we use the output to target a regression o a label
        # out = self.fc(x) #it needs ([32, n, 64]) because in the next operation needs to output a sequence of n
                        #if you don't reshape with sequence lenght in dimension 1 we don'n have out.size = [batch,n , n_features)
                        #also for this we need: x = x.unsqueeze(1).repeat(1, self.seq_len, 1) >>> lstm output >>> reshape

        if self.recon_loss_type == 'custom':
            par2 = self.par2(x)
            par3 = self.par3(x)
            return self.par1(x), self.act2(par2), self.act3(par3)
        else:
            return self.par1(x)


class LSTM_VAE(nn.Module):
    def __init__(self, seq_in, seq_out, no_features, output_size
                 , embedding_dim, latent_dim, Nf_lognorm, Nf_binomial=0, n_layers_1=1
                 ,n_layers_2=1,  kld_type='vanilla', recon_loss_type='custom'):
        super().__init__()

        self.seq_in = seq_in
        self.seq_out = seq_out
        self.no_features = no_features
        self.embedding_dim = embedding_dim
        self.hidden_size = 2*embedding_dim
        self.n_layers_1 = n_layers_1
        self.n_layers_2 = n_layers_2
        self.output_size = output_size
        self.latent_dim = latent_dim
        self.Nf_lognorm = Nf_lognorm
        self.Nf_binomial = Nf_binomial
        self.kld_type = kld_type
        self.recon_loss_type = recon_loss_type

        self.encoder = Encoder_vae(self.seq_in, self.no_features, self.embedding_dim, self.latent_dim, self.n_layers_1,
                                   self.n_layers_2, kld_type=self.kld_type)
        self.decoder = Decoder_vae(self.seq_out, self.embedding_dim, self.output_size, self.latent_dim, self.n_layers_1,
                                   self.n_layers_2, self.recon_loss_type)

        self.apply(self.weight_init)
        print(self)
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def sample(self, mu, dispersion):
        if self.kld_type=='custom':
            std = dispersion
            eps = torch.randn_like(std)
            return mu + eps * std
        elif self.kld_type=='vanilla':
            std = torch.exp(0.5 * dispersion)
            eps = torch.randn_like(std)
            return mu + eps * std

    def forward(self, x):
        torch.manual_seed(0)
        if self.kld_type == 'vanilla':
            mu, logvar = self.encoder(x)
            z = self.sample(mu, logvar)
            pars = self.decoder(z)
            return x, mu, logvar, pars

        elif self.kld_type == 'custom':
            mu, sigma, mu_prior, sigma_prior = self.encoder(x)
            z = self.sample(mu, sigma)
            pars = self.decoder(z)
            return x, mu, sigma, mu_prior, sigma_prior, pars

    def encode(self, x):
        self.eval()
        mu, sigma = self.encoder(x)
        return mu, sigma

    def decode(self, x):
        self.eval()
        decoded = self.decoder(x)
        squeezed_decoded = decoded.squeeze()
        return squeezed_decoded


def train_lstm_vae(param_conf, no_features, train_iter, test_iter, model, criterion, optimizer, scheduler,
          device, out_dir, model_name,  Nf_lognorm=0, Nf_binomial=0, epochs=100,
                   kld_factor = 1, kld_type='vanilla', recon_loss_type='custom', es_patience=10):

    """
    Training function.

    Args:
        train_iter: (DataLoader): train data iterator
        test_iter: (DataLoader): test data iterator
        model: model
        criterion: loss to use
        optimizer: optimizer to use
        config:
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if Nf_lognorm == 0:
        Nf_lognorm = no_features
        Nf_binomial = 0

    early_stopping = EarlyStopping(patience=es_patience)

    train_loss = 0.0
    val_loss = 10 ** 16
    for epoch in tqdm(range(epochs), unit='epoch'):
        train_loss = 0.0
        for i, batch in tqdm(enumerate(train_iter), total=len(train_iter), unit="batch"):
            model.train()
            optimizer.zero_grad()

            if kld_type == 'custom':
                if recon_loss_type == 'custom':
                    x, mu, sigma, mu_prior, sigma_prior, pars = model(batch[0].to(device))
                    recon_loss = loss_function(x, pars, Nf_lognorm,
                                               Nf_binomial).mean()
                    mse_loss = nn.MSELoss()(x, pars[0])
                else:
                    x, mu, sigma, mu_prior, sigma_prior, y = model(batch[0].to(device))
                    recon_loss = nn.MSELoss()(x, y)

                KLD = KL_loss_forVAE_custom(mu, sigma, mu_prior, sigma_prior).mean()

            if kld_type == 'vanilla':
                if recon_loss_type == 'custom':
                    x, mu, log_var, pars = model(batch[0].to(device))
                    recon_loss = loss_function(x, pars, Nf_lognorm,
                                               Nf_binomial).mean()
                    mse_loss = nn.MSELoss()(x, pars[0])

                else:
                    x, mu, log_var, y = model(batch[0].to(device))
                    recon_loss = nn.MSELoss()(x, y)
                KLD = KL_loss(mu, log_var)


            loss = recon_loss + kld_factor * KLD  # the sum of KL is added to the mean of MSE
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if i % 10 == 0:
                print("Loss:")
                if recon_loss_type == 'custom':
                    print("loss {}: recon loss {} and kld {} mse_loss {}".format(loss, recon_loss, KLD, mse_loss))
                else:
                    print("loss {}: recon loss {} and kld {} ".format(loss, recon_loss, KLD))
                print(loss.item())

        model.eval()
        val_steps = 0
        temp_val_loss = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_iter), total=len(test_iter), desc="Evaluating"):

                if kld_type == 'custom':
                    if recon_loss_type == 'custom':
                        x, mu, sigma, mu_prior, sigma_prior, pars = model(batch[0].to(device))
                        recon_loss = loss_function(x, pars, Nf_lognorm,
                                                   Nf_binomial).mean()
                        mse_loss = nn.MSELoss()(x, pars[0])
                    else:
                        x, mu, sigma, mu_prior, sigma_prior, y = model(batch[0].to(device))
                        recon_loss = nn.MSELoss()(x, y)

                    KLD = KL_loss_forVAE_custom(mu, sigma, mu_prior, sigma_prior).mean()

                if kld_type == 'vanilla':
                    if recon_loss_type == 'custom':
                        x, mu, log_var, pars = model(batch[0].to(device))
                        recon_loss = loss_function(x, pars, Nf_lognorm,
                                                   Nf_binomial).mean()
                        mse_loss = nn.MSELoss()(x, pars[0])

                    else:
                        x, mu, log_var, y = model(batch[0].to(device))
                        recon_loss = nn.MSELoss()(x, y)

                    KLD = KL_loss(mu, log_var)

                temp_val_loss += recon_loss + kld_factor * KLD
                val_steps += 1

            temp_val_loss = temp_val_loss / val_steps
            scheduler.step(temp_val_loss)

            if i % 10 == 0:
                print("Loss:")
                if recon_loss_type == 'custom':
                    print("loss {}: recon loss {} and kld {} mse_loss {}".format(temp_val_loss, recon_loss, KLD, mse_loss))
                else:
                    print("loss {}: recon loss {} and kld {} ".format(loss, recon_loss, KLD))
                print(loss.item())

            early_stopping(temp_val_loss)
            if early_stopping.early_stop:
                break

            if temp_val_loss < val_loss:
                print('val_loss improved from {} to {}, saving model  {} to {}' \
                      .format(val_loss, temp_val_loss, model_name, out_dir))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'param_conf': param_conf,
                }, out_dir + '/{}.pth'.format(model_name))
                #torch.save(model, out_dir + '{}.pth'.format(model_name))
                val_loss = temp_val_loss

            print('eval loss {}'.format(val_loss))
