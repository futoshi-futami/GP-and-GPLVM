# Implements auto-encoding variational Bayes.
import sys; sys.path.append("./")
from copy import deepcopy
import sys; sys.path.insert(0, "../Theano"); sys.path.insert(0, "../../Theano")
import theano; import theano.tensor as T; import theano.sandbox.linalg as sT
import numpy as np
import pickle
from utils_pg import *

#入力ＸからのＮＮによる変換によって隠れ変数Ｚを推定します。後ろ向きには2層のＮＮ、前向きにはＧＰＬＶＭで推定します。
#XはN×Ｄ,隠れ層はN×Qの行列

class BackConstrained(object):
    def __init__(self, in_size, out_size, hidden_size, latent_size, optimizer = "adadelta"):
        self.prefix = "BC_"
        self.X = T.matrix("X")
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.optimizer = optimizer

        self.define_layers()
        
    def define_layers(self):
        self.params = []
        
        layer_id = "1"
        self.W_xh = init_weights((self.in_size, self.hidden_size), self.prefix + "W_xh" + layer_id)
        self.b_xh = init_bias(self.hidden_size, self.prefix + "b_xh" + layer_id)

        layer_id = "2"#uは平均用、sigmaは分散用の重み
        self.W_hu = init_weights((self.hidden_size, self.latent_size), self.prefix + "W_hu" + layer_id)
        self.b_hu = init_bias(self.latent_size, self.prefix + "b_hu" + layer_id)
        self.W_hsigma = init_weights((self.hidden_size, self.latent_size), self.prefix + "W_hsigma" + layer_id)
        self.b_hsigma = init_bias(self.latent_size, self.prefix + "b_hsigma" + layer_id)
        
        
        #後ろ向き用のパラメータ（今回は使いません）
        layer_id = "3"
        self.W_zh = init_weights((self.latent_size, self.hidden_size), self.prefix + "W_zh" + layer_id)
        self.b_zh = init_bias(self.hidden_size, self.prefix + "b_zh" + layer_id)
 
        self.params += [self.W_xh, self.b_xh, self.W_hu, self.b_hu, self.W_hsigma, self.b_hsigma, \
                        self.W_zh, self.b_zh]

        layer_id = "4"
        if self.continuous:
            self.W_hyu = init_weights((self.hidden_size, self.out_size), self.prefix + "W_hyu" + layer_id)
            self.b_hyu = init_bias(self.out_size, self.prefix + "b_hyu" + layer_id)
            self.W_hysigma = init_weights((self.hidden_size, self.out_size), self.prefix + "W_hysigma" + layer_id)
            self.b_hysigma = init_bias(self.out_size, self.prefix + "b_hysigma" + layer_id)
            self.params += [self.W_hyu, self.b_hyu, self.W_hysigma, self.b_hysigma]
        else:
            self.W_hy = init_weights((self.hidden_size, self.out_size), self.prefix + "W_hy" + layer_id)
            self.b_hy = init_bias(self.out_size, self.prefix + "b_hy" + layer_id)
            self.params += [self.W_hy, self.b_hy]

        # encoder
        h_enc = T.nnet.relu(T.dot(self.X, self.W_xh) + self.b_xh)
        
        self.mu = T.dot(h_enc, self.W_hu) + self.b_hu
        log_var = T.dot(h_enc, self.W_hsigma) + self.b_hsigma
        self.var = T.exp(log_var)
        self.sigma = T.sqrt(self.var)

        srng = T.shared_randomstreams.RandomStreams(234)
        eps = srng.normal(self.mu.shape)
        self.z = self.mu + self.sigma * eps

        # decoder
        h_dec = T.nnet.relu(T.dot(self.z, self.W_zh) + self.b_zh)
        if self.continuous:
            self.reconstruct = T.dot(h_dec, self.W_hyu) + self.b_hyu
            self.log_var_dec = T.dot(h_dec, self.W_hysigma) + self.b_hysigma
            self.var_dec = T.exp(self.log_var_dec)
        else:
            self.reconstruct = T.nnet.sigmoid(T.dot(h_dec, self.W_hy) + self.b_hy)

    def multivariate_bernoulli(self, y_pred, y_true):
        return T.sum(y_true * T.log(y_pred) + (1 - y_true) * T.log(1 - y_pred), axis=1)
   
    def log_mvn(self, y_pred, y_true):
        p = y_true.shape[1]
        return T.sum(-0.5 * p * np.log(2 * np.pi) - 0.5 * self.log_var_dec - 0.5 * ((y_true - y_pred)**2 / self.var_dec), axis=1)

    def kld(self, mu, var):
        return 0.5 * T.sum(1 + T.log(var) - mu**2 - var, axis=1)









from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.optimizers import adam
from data import load_mnist, save_images

def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def unpack_gaussian_params(params):
    # Params of a diagonal Gaussian.
    D = np.shape(params)[-1] // 2
    mean, log_std = params[:, :D], params[:, D:]
    return mean, log_std

def sample_diag_gaussian(mean, log_std, rs):
    return rs.randn(*mean.shape) * np.exp(log_std) + mean

def bernoulli_log_density(targets, unnormalized_logprobs):
    # unnormalized_logprobs are in R
    # Targets must be -1 or 1
    label_probabilities = -np.logaddexp(0, -unnormalized_logprobs*targets)
    return np.sum(label_probabilities, axis=-1)   # Sum across pixels.

def relu(x):    return np.maximum(0, x)
def sigmoid(x): return 0.5 * (np.tanh(x) + 1)

def init_net_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a (weights, biases) tuples for all layers."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def batch_normalize(activations):
    mbmean = np.mean(activations, axis=0, keepdims=True)
    return (activations - mbmean) / (np.std(activations, axis=0, keepdims=True) + 1)

def neural_net_predict(params, inputs):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       Applies batch normalization to every layer but the last."""
    for W, b in params[:-1]:
        outputs = batch_normalize(np.dot(inputs, W) + b)  # linear transformation
        inputs = relu(outputs)                            # nonlinear transformation
    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb
    return outputs

def nn_predict_gaussian(params, inputs):
    # Returns means and diagonal variances
    return unpack_gaussian_params(neural_net_predict(params, inputs))

def vae_lower_bound(gen_params, rec_params, data, rs):
    # We use a simple Monte Carlo estimate of the KL
    # divergence from the prior.
    q_means, q_log_stds = nn_predict_gaussian(rec_params, data)
    return q_means, q_log_stds


if __name__ == '__main__':
    # Model hyper-parameters
    latent_dim = 10
    data_dim = 784  # How many pixels in each image (28x28).
    gen_layer_sizes = [latent_dim, 300, 200, data_dim]
    rec_layer_sizes = [data_dim, 200, 300, latent_dim * 2]

    # Training parameters
    param_scale = 0.01
    batch_size = 200
    num_epochs = 15
    step_size = 0.001

    print("Loading training data...")
    N, train_images, _, test_images, _ = load_mnist()
    on = train_images > 0.5
    train_images = train_images * 0 - 1
    train_images[on] = 1.0

    init_gen_params = init_net_params(param_scale, gen_layer_sizes)
    init_rec_params = init_net_params(param_scale, rec_layer_sizes)
    combined_init_params = (init_gen_params, init_rec_params)

    num_batches = int(np.ceil(len(train_images) / batch_size))
    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    seed = npr.RandomState(0)
    def objective(combined_params, iter):
        data_idx = batch_indices(iter)
        gen_params, rec_params = combined_params
        return -vae_lower_bound(gen_params, rec_params, train_images[data_idx], seed) / data_dim

    # Get gradients of objective using autograd.
    objective_grad = grad(objective)

    print("     Epoch     |    Objective  |       Fake probability | Real Probability  ")
    def print_perf(combined_params, iter, grad):
        if iter % 10 == 0:
            gen_params, rec_params = combined_params
            bound = np.mean(objective(combined_params, iter))
            print("{:15}|{:20}".format(iter//num_batches, bound))

            fake_data = generate_from_prior(gen_params, 20, latent_dim, seed)
            save_images(fake_data, 'vae_samples.png', vmin=0, vmax=1)

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, combined_init_params, step_size=step_size,
                            num_iters=num_epochs * num_batches, callback=print_perf)
