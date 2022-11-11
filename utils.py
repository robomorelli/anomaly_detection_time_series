from torch import nn
import torch

clip_x_to0 = 1e-4

class InverseSquareRootLinearUnit(nn.Module):

    def __init__(self, min_value=5e-3):
        super(InverseSquareRootLinearUnit, self).__init__()
        self.min_value = min_value

    def forward(self, x):
        return 1. + self.min_value \
               + torch.where(torch.gt(x, 0), x, torch.div(x, torch.sqrt(1 + (x * x))))

class ClippedTanh(nn.Module):

    def __init__(self):
        super(ClippedTanh, self).__init__()

    def forward(self, x):
        return 0.5 * (1 + 0.999 * torch.tanh(x))

class ClippedTanh0(nn.Module):

    def __init__(self):
        super(ClippedTanh0, self).__init__()

    def forward(self, x):
        return 0*x

class SmashTo0(nn.Module):

    def __init__(self):
        super(SmashTo0, self).__init__()

    def forward(self, x):
        return 0*x

class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)

class SigmaPrior(nn.Module):

    def __init__(self, min_value=5e-3):
        super(SmashTo0, self).__init__()

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return 0*x


def KL_loss_forVAE(mu, sigma):
    mu_prior = torch.tensor(0)
    sigma_prior = torch.tensor(1)
    kl_loss = torch.mul(torch.mul(sigma, sigma), torch.mul(sigma_prior,sigma_prior))
    div = torch.div(mu_prior - mu, sigma_prior)
    kl_loss += torch.mul(div, div)
    kl_loss += torch.log(torch.div(sigma_prior, sigma)) -1
    return 0.5 * torch.sum(kl_loss, axis=-1)

def KL_loss_forVAE_2(mu, sigma, mu_prior, sigma_prior):

    kl_loss = torch.mul(torch.mul(sigma, sigma), torch.mul(sigma_prior, sigma_prior))
    div = torch.div(mu_prior - mu, sigma_prior)
    kl_loss += torch.mul(div, div)
    kl_loss += torch.log(torch.div(sigma_prior, sigma)) - 1

    #kl_loss = K.tf.multiply(K.square(sigma), K.square(sigma_prior))
    #kl_loss += K.square(K.tf.divide(mu_prior - mu, sigma_prior))
    #kl_loss += K.log(K.tf.divide(sigma_prior, sigma)) -1
    return 0.5 * torch.sum(kl_loss, axis=-1)

#def KL_loss(mean, sigma):
#    log_std = torch.log(sigma)
#    kl_loss = 0.5 * torch.sum(1 + log_std - mean ** 2 - torch.exp(log_std), dim=1)
#    return kl_loss

def KL_loss(mu, log_var):
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl = kl.sum(-1)     # to go from multi-dimensional z to single dimensional z : (batch_size x latent_size) ---> (batch_size)
                            # i.e Z = [ [z1_1, z1_2 , ...., z1_lt] ] ------> z = [ z1]
                            #         [ [z2_1, z2_2, ....., z2_lt] ]             [ z2]
                            #                   .                                [ . ]
                            #                   .                                [ . ]
                            #         [[zn_1, zn_2, ....., zn_lt] ]              [ zn]

                            #        lt=latent_size
    kl = kl.mean()

    return kl

def loss_function(x, pars, Nf_lognorm, Nf_binomial):
    recon_loss = RecoProb_forVAE(x, pars[0], pars[1], pars[2], Nf_lognorm, Nf_binomial)
    return recon_loss

def RecoProb_forVAE(x, par1, par2, par3, Nf_lognorm, Nf_binomial):

    N = 0
    nll_loss = 0

    #Log-Normal distributed variables
    mu = par1[:,:,:Nf_lognorm]
    sigma = par2[:,:,:Nf_lognorm]
    fraction = par3[:,:,:Nf_lognorm]

    x_clipped = torch.clamp(x[:,:,:Nf_lognorm], clip_x_to0, 1e8)
    single_NLL = torch.where(torch.le(x[:,:,:Nf_lognorm], clip_x_to0),
                            -torch.log(fraction),
                                -torch.log(1-fraction)
                                + torch.log(sigma)
                                + torch.log(x_clipped)
                                + 0.5*torch.mul(torch.div(torch.log(x_clipped) - mu, sigma),
                                                  torch.div(torch.log(x_clipped) - mu, sigma)))
    nll_loss += torch.sum(single_NLL, axis=-1)
    N += Nf_lognorm

    if Nf_binomial != 0:
        #Binomial distributed variables
        p = 0.5*(1+0.98*torch.tanh(par1[:, :,N: N+Nf_binomial]))
        single_NLL = -torch.where(torch.eq(x[:,:, N: N+Nf_binomial],1), torch.log(p), torch.log(1-p))
        nll_loss += torch.sum(single_NLL, axis=-1)
        N += Nf_binomial

    return nll_loss


def KL_mvn(mu, var):
    return (mu.shape[1] + torch.sum(torch.log(var), dim=1) - torch.sum(mu**2, dim=1) - torch.sum(var, dim=1)) / 2.0