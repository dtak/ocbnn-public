
"""
Class templates for OC-BNNs (without inference).
"""

import numpy as np
import torch
import logging
import json
import os

from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions.kl import kl_divergence
from tensorboardX import SummaryWriter


class BayesianNeuralNetwork:
	""" Base class for performing Bayesian inference over a multi-layered perceptron (MLP). """

	def __init__(self, uid="bnn-0", configfile="config.json"):
		""" Instantiates the BNN. """
		self.uid = uid # unique ID
		with open(configfile, 'r') as rf:
			self.config = json.load(rf)
		if self.config["use_tensorboard"]:
			logging.info(f'Logging results using TensorBoard.')
			self.writer = SummaryWriter(f'runs/instance_{self.uid}')
		if not os.path.isdir('history'):
			os.mkdir('history')
		with open(f'history/{self.uid}.json', 'w') as wf:
			json.dump(self.config, wf)
		logging.info(f'[{self.uid}] BNN instantiated.')

	def _unpack_layers(self, weights):
		""" Helper function for forward pass. Code taken from PyTorch. """
		num_weight_samples = weights.shape[0]
		for m, n in self.layer_shapes:
			yield weights[:, :m*n].reshape((num_weight_samples, m, n)), weights[:, m*n:m*n+n].reshape((num_weight_samples, 1, n))
			weights = weights[:,(m+1)*n:]

	def _nonlinearity(self, x):
		""" Activation function. """
		if self.config["activation"] == "rbf":
			return torch.exp(-x.pow(2))
		return x

	def forward(self, X, weights=None):
		""" Forward pass of BNN. Code taken from PyTorch. """
		if weights is None:
			weights = self.weights

		if weights.ndimension() == 1:
			weights = weights.unsqueeze(0)

		num_weight_samples = weights.shape[0]
		X = X.expand(num_weight_samples, *X.shape)
		for W, b in self._unpack_layers(weights):
			outputs = torch.einsum('mnd,mdo->mno', [X, W]) + b
			X = self._nonlinearity(outputs)

		outputs = outputs.squeeze(dim=0)
		return outputs

	def log_weight_prior(self, weights=None):
		""" Computes the "standard" prior, i.e. Gaussian log-probability of BNN weights. """
		if weights is None:
			weights = self.weights
		return self.weight_dist.log_prob(weights).sum()

	def log_prior(self):
		""" Computes the log-prior term. """
		prior = self.log_weight_prior()
		if self.config["prior_type"] == "oc_positive":
			prior += self.log_positive_prior()
		elif self.config["prior_type"] == 'oc_negative':
			prior += self.log_negative_prior()
		return prior

	def _sample_from_hypercube(self, region, nsamples, mode='even'):
		""" Generate `nsamples` points from the hypercube `region`.

			Arguments:
				mode: 'uniform' for uniform sampling, 'even' for evenly-spaced sampling
			Returns:
				Tensor of shape (nsamples, self.Xdim)
		"""
		samples = torch.tensor([])
		for d in range(self.Xdim):
			lower = min(self.X_train[:,d])
			if region[2*d] > -np.inf:
				lower = region[2*d]
			upper = max(self.X_train[:,d])
			if region[2*d+1] < np.inf:
				upper = region[2*d+1]
			if mode == 'uniform':
				unif = Uniform(torch.tensor([lower]), torch.tensor([upper])).sample(torch.Size([nsamples])).squeeze()
				samples = torch.cat((samples, unif.unsqueeze(0)), 0)
			elif mode == 'even':
				samples = torch.cat((samples, torch.linspace(lower, upper, nsamples).unsqueeze(0)), 0)
		return torch.t(samples)

	def log_posterior(self, batch_indices=None):
		""" Computes the log-posterior term. """
		return self.log_prior() + self.log_likelihood(batch_indices=batch_indices)


class BNNRegressor(BayesianNeuralNetwork):
	""" Base class for BNN for regression. """

	def __init__(self, uid="bnn-0", configfile="config.json"):
		""" Instantiates the BNN. """
		super().__init__(uid, configfile)

	def load(self, **kwargs):
		""" Loads dataset. """

		# Define train/test sets.
		self.__dict__.update(kwargs)
		self.Xdim = self.X_train.shape[1]
		self.Ydim = self.Y_train.shape[1]
		self.N_train = self.Y_train.shape[0]
		if hasattr(self, 'Y_test'):
			self.N_test = self.Y_test.shape[0]

		# Initialize all weights.
		self.layer_sizes = [self.Xdim] + self.config["architecture"] + [self.Ydim]
		self.layer_shapes = list(zip(self.layer_sizes[:-1], self.layer_sizes[1:]))
		self.nweights = sum((m + 1) * n for m, n in self.layer_shapes)
		self.weight_dist = Normal(0, self.config["sigma_w"])
		self.noise_dist = Normal(0, self.config["sigma_noise"])
		self.weights = Variable(self.weight_dist.sample(torch.Size([self.nweights])), requires_grad=True)

		logging.info(f'[{self.uid}] Dataset <{self.dataset_name}> loaded and BNN weights initialized.')

	def add_positive_constraint(self, domain, function):
		""" Define a constrained region as a function (positive constraint).

			Arguments:
				`domain` should be a (x1_lower, x1_upper, x2_lower, x2_upper, ...) tuple, i.e. length is Xdim * 2,
					representing the lower and upper bounds of each input dimension.
				`function` is a lambda function f: Tensor of (n_samples, self.Xdim) --> Tensor of (n_samples, 1),
					computing the constrained output for points lying in the domain. f does not have to be well-defined
					for points outside the constrained region.
		"""
		if not hasattr(self, 'pconstraints'):
			self.pconstraints = []
		self.pconstraints.append((domain, function))

		if not self.config["ocp_bimodal"]:
			self.cr_xsamples = torch.tensor([])
			self.cr_ysamples = torch.tensor([])
			for dd, ff in self.pconstraints:
				xsamp = self._sample_from_hypercube(dd, self.config["ocp_nsamples"])
				self.cr_xsamples = torch.cat((self.cr_xsamples, xsamp), 0)
				self.cr_ysamples = torch.cat((self.cr_ysamples, ff(xsamp)), 0)
		else:
			self.cr_xsamples = torch.tensor([])
			self.cr_up_ysamples = torch.tensor([])
			self.cr_down_ysamples = torch.tensor([])
			for i, (dd, ff) in enumerate(self.pconstraints):
				if i % 2 == 0:
					xsamp = self._sample_from_hypercube(dd, 2 * self.config["ocp_nsamples"])
					self.cr_xsamples = torch.cat((self.cr_xsamples, xsamp), 0)
					self.cr_down_ysamples = torch.cat((self.cr_down_ysamples, ff(xsamp)), 0)
				else:
					self.cr_up_ysamples = torch.cat((self.cr_up_ysamples, ff(xsamp)), 0)

		logging.info(f'[{self.uid}] Defined positive constrained region.')

	def log_positive_prior(self):
		""" Computes log-positive-prior term. """
		nn_mean = self.forward(X=self.cr_xsamples)
		if self.config["ocp_bimodal"]:
			up = Normal(self.cr_up_ysamples, self.config["ocp_sigma_y"] * torch.ones(len(self.cr_xsamples))).log_prob(nn_mean)
			down = Normal(self.cr_down_ysamples, self.config["ocp_sigma_y"] * torch.ones(len(self.cr_xsamples))).log_prob(nn_mean)
			up += torch.log(torch.tensor([0.5]))
			down += torch.log(torch.tensor([0.5]))
			return torch.logsumexp(torch.stack((up, down), 0), dim=0).sum()
		return Normal(self.cr_ysamples, self.config["ocp_sigma_y"] * torch.ones(len(self.cr_xsamples))).log_prob(nn_mean).sum()

	def add_negative_constraint(self, domain, constraint_funcs):
		""" Define a constrained region as one that has to satisfy the constraint inequality: g(x, y) < 0 (negative constraint). """
		if not hasattr(self, 'nconstraints'):
			self.nconstraints = []
		self.nconstraints.append((constraint_funcs, domain))
		logging.info(f'[{self.uid}] Defined negative constrained region.')

	def _constraint_classifier(self, z):
		""" Sigmoidal constraint classfier. """
		return 0.25 * (torch.tanh(-self.config["ocn_tau"][0]*z) + 1) * (torch.tanh(-self.config["ocn_tau"][1]*z) + 1)

	def log_negative_prior(self):
		""" Computes log-negative-prior term. """
		penalty = torch.zeros(self.config["ocn_nsamples"])
		for cflist, sr in self.nconstraints:
			cr_xsamples = self._sample_from_hypercube(sr, self.config["ocn_nsamples"], mode='uniform')
			nn_mean = self.forward(X=cr_xsamples)
			mult = torch.ones(nn_mean.shape[0])
			for cf in cflist:
				mult *= self._constraint_classifier(cf(cr_xsamples, nn_mean))
			penalty += mult
		violation = penalty.sum() / (self.config["ocn_nsamples"] * len(self.nconstraints))
		log_prob = -self.config["ocn_gamma"] * violation # p(w) = exp(- gamma * constraint_viol)
		return log_prob

	def log_likelihood(self, batch_indices=None):
		""" Computes log-likelihood term. """
		if batch_indices is None:
			batch = self.X_train
			target = self.Y_train
			multiplier = 1
		else:
			batch = self.X_train[batch_indices]
			target = self.Y_train[batch_indices]
			multiplier = (self.N_train / len(batch_indices))
		means = self.forward(X=batch)
		if self.Ydim == 1:
			return multiplier * self.noise_dist.log_prob(means - target).sum()
		return multiplier * MVN(means, self.config["sigma_noise"] * torch.eye(self.Ydim)).log_prob(target).sum()

	def evaluate(self):
		""" Evaluate BNN on train/test sets. """
		logging.info(f'Train RMSE: {self.train_rmse():.3f}')
		logging.info(f'Test RMSE: {self.test_rmse():.3f}')
		logging.info(f'Test NLL: {self.test_neg_loglik():.3f}')
	

class BNNClassifier(BayesianNeuralNetwork):
	""" Base class for BNN for classification. Output dimension must be > 1. """

	def __init__(self, uid="bnn-0", configfile="config.json"):
		""" Instantiates the BNN. """
		super().__init__(uid, configfile)

	def load(self, **kwargs):
		""" Loads dataset. """

		# Define train/test sets.
		self.__dict__.update(kwargs)
		self.Xdim = self.X_train.shape[1]
		self.Ydim = torch.max(self.Y_train).item() + 1
		self.N_train = self.Y_train.shape[0]
		if hasattr(self, 'Y_test'):
			self.N_test = self.Y_test.shape[0]

		# Initialize all weights.
		self.layer_sizes = [self.Xdim] + self.config["architecture"] + [self.Ydim]
		self.layer_shapes = list(zip(self.layer_sizes[:-1], self.layer_sizes[1:]))
		self.nweights = sum((m + 1) * n for m, n in self.layer_shapes)
		self.weight_dist = Normal(0, self.config["sigma_w"])
		self.noise_dist = Normal(0, self.config["sigma_noise"])
		self.weights = Variable(self.weight_dist.sample(torch.Size([self.nweights])), requires_grad=True)

		logging.info(f'[{self.uid}] Dataset <{self.dataset_name}> loaded and BNN weights initialized.')

	def add_positive_constraint(self, domain, forbidden_classes):
		""" Define a constrained region as: ((x1_lower, x1_upper, x2_lower, x2_upper, ...), [y_k1, y_k2, ...])
			where y_k* are the forbidden classes.
		"""
		if not hasattr(self, 'pconstraints'):
			self.pconstraints = []
		if not forbidden_classes or len(forbidden_classes) >= self.Ydim:
			logging.error("Number of forbidden classes must be between 0 and total number of classes (exclusive).")
			raise Exception
		self.pconstraints.append((domain, forbidden_classes))
		logging.info(f'[{self.uid}] Defined positive constrained region.')

	def log_positive_prior(self):
		""" Computes log-positive-prior term. """
		logprob = torch.tensor(0.)
		for i, (dom, fclasses) in enumerate(self.pconstraints):
			xsamp = self._sample_from_hypercube(dom, self.config["ocp_nsamples"])
			nn_mean = self.forward(X=xsamp)
			nn_probs = torch.nn.Softmax(dim=1)(nn_mean)
			dirprobs = self.config["ocp_gamma"] * torch.tensor(fclasses).bincount(minlength=self.Ydim).type(torch.float)
			dirprobs = self.config["ocp_gamma"] - (dirprobs * (1 - self.config["ocp_alpha"]))
			logprob += Dirichlet(dirprobs).log_prob(nn_probs).sum()
		return logprob

	def log_likelihood(self, batch_indices=None):
		""" Computes log P(Y|X,W). Tensorized. """
		if batch_indices is None:
			batch = self.X_train
			target = self.Y_train
			multiplier = 1
		else:
			batch = self.X_train[batch_indices]
			target = self.Y_train[batch_indices]
			multiplier = (self.N_train / len(batch_indices))
		means = self.forward(X=batch)
		probs = torch.nn.LogSoftmax(dim=1)(means)
		return multiplier * probs.gather(1, target.view(-1, 1)).sum()

	def evaluate(self, pred_type="max", beta=1.0):
		""" Evaluate BNN on train/test sets. """
		accuracy, precision, recall, fscore = self.train_metrics(beta=beta)
		logging.info(f'Train Accuracy: {accuracy:.3f}')
		if self.Ydim == 2:
			logging.info(f'Train Precision: {precision:.3f}')
			logging.info(f'Train Recall: {recall:.3f}')
			logging.info(f'Train F({beta}) Score: {fscore:.3f}')

		accuracy, precision, recall, fscore = self.test_metrics(beta=beta)
		logging.info(f'Test Accuracy: {accuracy:.3f}')
		if self.Ydim == 2:
			logging.info(f'Test Precision: {precision:.3f}')
			logging.info(f'Test Recall: {recall:.3f}')
			logging.info(f'Test F({beta}) Score: {fscore:.3f}')
