
"""
Base class template for OC-BNNs. 

Supports the baseline isotropic Gaussian prior, as well as our output-constrained priors (COCP and AOCP).

Contains mixin classes for:
	- Regression (generic BNN supports multi-dimensional output, but OC-BNNs only support 1D output)
	- Classification
	- Binary Classification (used only with the AOCP, see paper for details)
"""

import numpy as np
import torch
import logging
import yaml
import math
import os

from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal as MVN


class BayesianNeuralNetwork:
	""" Base class for performing Bayesian inference over a multi-layered perceptron (MLP). """

	def __init__(self, uid="bnn-0", configfile="config.yaml"):
		""" Instantiates the BNN. """
		self.uid = uid
		with open(configfile, 'r') as rf:
			config = yaml.load(rf, Loader=yaml.FullLoader)
			self.__dict__.update(config)
		if not os.path.isdir('history'):
			os.mkdir('history')
		# Saves a copy of the configs in the `history/` folder.
		with open(f'history/{self.uid}.yaml', 'w') as wf:
			yaml.dump(config, wf)

		self.all_bayes_samples = []
		self.dconstraints = dict()
		self.pconstraints = dict()
		self.aocp_mean = None
		self.aocp_std = None

		logging.info(f'[{self.uid}] BNN instantiated. Configs saved as `history/{self.uid}.yaml`.')

	def load(self, **kwargs):
		""" Loads dataset. """

		# Define train/test sets.
		self.__dict__.update(kwargs)
		self.Xdim = self.X_train.shape[1]
		self.Ydim = self._Ydim		
		self.N_train = self.Y_train.shape[0]
		if hasattr(self, 'Y_test'):
			self.N_test = self.Y_test.shape[0]
			test_stats = f" and {self.N_test} test points"
		else:
			test_stats = ""

		# Initialize weights.
		self.layer_sizes = [self.Xdim] + self.architecture + [self.Ydim]
		self.layer_shapes = list(zip(self.layer_sizes[:-1], self.layer_sizes[1:]))
		self.nweights = sum((m + 1) * n for m, n in self.layer_shapes)
		self.weights = Variable(Normal(0, self.sigma_w).sample(torch.Size([self.nweights])), requires_grad=True)

		logging.info(f'[{self.uid}] Dataset <{self.dataset_name}> loaded: {self.N_train} training points{test_stats}.')

	def _unpack_layers(self, weights):
		""" Helper function for forward pass. Implementation from PyTorch. """
		num_weight_samples = weights.shape[0]
		for m, n in self.layer_shapes:
			yield weights[:, :m*n].reshape((num_weight_samples, m, n)), weights[:, m*n:m*n+n].reshape((num_weight_samples, 1, n))
			weights = weights[:,(m+1)*n:]

	def _nonlinearity(self, x):
		""" Activation function.
			Implement custom functions below with additional control statements.  
		"""
		if self.activation == "rbf":
			return torch.exp(-x.pow(2))
		elif self.activation == "relu":
			return torch.max(x, torch.zeros_like(x))
		return x

	def forward(self, X, weights=None):
		""" Forward pass of BNN. Implementation from PyTorch. """
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

	def log_prior(self):
		""" Computes the prior.
			Automatically decides which output-constrained prior to use, depending on existing constraints.  
		"""
		if self.use_ocbnn:
			if self.aocp_mean is not None:
				return self.isotropic_gaussian_prior(mean=self.aocp_mean, std=self.aocp_std)
			logp = self.isotropic_gaussian_prior()
			if "positive_gaussian_cocp" in self.dconstraints:
				logp += self.positive_gaussian_cocp()
			if "negative_exponential_cocp" in self.dconstraints:
				logp += self.negative_exponential_cocp()
			if "positive_dirichlet_cocp" in self.dconstraints:
				logp += self.positive_dirichlet_cocp()
			return logp
		return self.isotropic_gaussian_prior()

	def log_posterior(self, batch_indices=None):
		""" Computes the posterior. """
		return self.log_prior() + self.log_likelihood(batch_indices=batch_indices)

	def isotropic_gaussian_prior(self, weights=None, mean=0, std=None):
		""" Isotropic Gaussian prior. """
		if weights is None:
			weights = self.weights
		if std is None:
			std = self.sigma_w
		return Normal(mean, std).log_prob(weights).sum()

	def _sample_from_hypercube(self, region, nsamples, mode='uniform'):
		""" Generate `nsamples` points from the hypercube `region`.

			Args:
				mode: 'uniform' for uniform sampling, 'even' for evenly-spaced sampling
			Returns:
				Tensor of shape (nsamples, self.Xdim)
		"""
		samples = torch.tensor([])
		for d in range(self.Xdim):
			lower = self.X_train_min[d]
			if region[2*d] > -np.inf:
				lower = region[2*d]
			upper = self.X_train_max[d]
			if region[2*d+1] < np.inf:
				upper = region[2*d+1]
			lower, upper = float(lower), float(upper)
			if mode == 'uniform':
				unif = Uniform(lower, upper).sample(torch.Size([nsamples])).unsqueeze(0)
				samples = torch.cat((samples, unif), 0)
			elif mode == 'even':
				samples = torch.cat((samples, torch.linspace(lower, upper, nsamples).unsqueeze(0)), 0)
		return torch.t(samples)

	def load_bayes_samples(self, filename, descriptor="sample"):
		""" Load prior/posterior samples from file and appends to `self.all_bayes_samples`. """
		self.all_bayes_samples.append((torch.load(filename), descriptor))
		logging.info(f"[{self.uid}] Posterior Sample #{len(self.all_bayes_samples)}: Loaded from `{filename}`.")

	def learn_gaussian_aocp(self, verbose=True):
		""" Amortized output-constrained prior.
			Optimizes the constrained objective directly to learn AOCP parameters.
			AOCP parameters are also saved to file.
		"""
		# Initialize VP parameters.
		init_means = self.aocp_init_mean + torch.randn(self.nweights, 1)
		init_log_stds = self.aocp_init_std * torch.ones(self.nweights, 1)
		self._aocp_params = Variable(torch.cat([init_means, init_log_stds], dim=1), requires_grad=True)
		optimizer = torch.optim.Adagrad([self._aocp_params], lr=self.aocp_init_lr)

		# Optimize VP.
		for epoch in range(1, self.aocp_nepochs + 1):
			optimizer.zero_grad()
			loss = self.gaussian_aocp_objective()
			loss.backward()
			optimizer.step()
			if verbose and epoch % 10 == 0:
				logging.info(f'[{self.uid}] Epoch {epoch}: {loss.item():.2f}')

		self._aocp_params.detach_()
		torch.save(self._aocp_params, f"history/{self.uid}_aocp.pt")
		self.aocp_mean = self._aocp_params[:,0]
		self.aocp_std = torch.logsumexp(torch.stack((self._aocp_params[:,1], torch.zeros(self.nweights))), 0) / self.aocp_std_multiplier
		logging.info(f'[{self.uid}] AOCP parameters learnt. Also saved as `history/{self.uid}_aocp.pt`.')

	def load_gaussian_aocp_parameters(self, filename):
		""" Load AOCP parameters. """
		aocp_params = torch.load(filename)
		self.aocp_mean = aocp_params[:,0]
		self.aocp_std = torch.logsumexp(torch.stack((aocp_params[:,1], torch.zeros(self.nweights))), 0) / self.aocp_std_multiplier
		logging.info(f'[{self.uid}] Loaded AOCP parameters from `{filename}`. This replaces any previously loaded or learnt AOCP parameters.')

	def update_config(self, **kwargs):
		""" Helper function for updating configs. """
		self.__dict__.update(**kwargs)
		logging.info(f'[{self.uid}] Configs updated.')

	def clear_all_samples(self):
		""" Empty self.all_bayes_samples. """
		self.all_bayes_samples = []
		logging.info(f'[{self.uid}] All posterior samples cleared from memory.')

	def debug_mode(self):
		""" Debug mode: collect a few samples only to ensure code/pipeline is working. """
		self.old_infer_nsamples = self.infer_nsamples
		self.old_hmc_nburnin = self.hmc_nburnin
		self.old_bbb_epochs = self.bbb_epochs
		self.old_svgd_epochs = self.svgd_epochs
		self.old_sgld_nburnin = self.sgld_nburnin

		self.infer_nsamples = 3
		self.hmc_nburnin = 5
		self.bbb_epochs = 100
		self.svgd_epochs = 10
		self.sgld_nburnin = 5
		self.aocp_nepochs = 1

		logging.info(f'[{self.uid}] Debug mode activated.')

	def switch_off_debug_mode(self):
		""" Deactivate debug mode. """
		if not hasattr(self, 'old_infer_nsamples'):
			logging.error("You cannot switch off debug mode without having turned it on first!")
			raise Exception

		self.infer_nsamples = self.old_infer_nsamples
		self.hmc_nburnin = self.old_hmc_nburnin
		self.bbb_epochs = self.old_bbb_epochs
		self.svgd_epochs = self.old_svgd_epochs
		self.sgld_nburnin = self.old_sgld_nburnin

		del self.old_infer_nsamples
		del self.old_hmc_nburnin
		del self.old_bbb_epochs
		del self.old_svgd_epochs
		del self.old_sgld_nburnin

		logging.info(f'[{self.uid}] Debug mode turned off.')


class RegressorMixin:
	""" Regression-specific functionality. """

	@property
	def _Ydim(self):
		return self.Y_train.shape[1]

	def log_likelihood(self, batch_indices=None):
		""" Computes the likelihood. """
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
			return multiplier * Normal(0, self.sigma_noise).log_prob(means - target).sum()
		return multiplier * MVN(means, self.sigma_noise * torch.eye(self.Ydim)).log_prob(target).sum()

	def predict(self, bayes_samples, domain):
		""" Generate BNN's prediction over `domain` for each sample.

			Args: 
				`bayes_samples` is a (M, self.nweights) tensor, representing M samples of the prior/posterior.
				`domain` is a (K, self.Xdim) tensor, representing K input points passed to the BNN.

			Returns:
				Tensor of (M, K, self.Ydim) of outputs across all posterior samples and all input points.
		"""
		return torch.tensor(np.apply_along_axis(lambda w: self.forward(X=domain, weights=torch.tensor(w)).numpy(), 1, bayes_samples))

	def add_deterministic_constraint(self, constrained_domain, interval_func, prior_type):
		""" Specify a (positive or negative) deterministic constraint. 
			Implemented for 1D output only.

			Args:
				`constrained_domain` is a (x1_lower, x1_upper, x2_lower, x2_upper, ...) tuple, i.e. length is Xdim * 2,
					representing the lower and upper bounds of each input dimension of the constrained input region.
					Values of +/- np.inf will be replaced with the corresponding bounds of the training set.

				`interval_func` is a function f:
					f takes as input a tensor X of shape (n_samples, self.Xdim) sampled from `constrained_domain`.
					f(X) evaluates to a list of length `n_samples`, each containing a sublist of permitted output ranges for X[i].

				`prior_type` is the COCP/AOCP that we want to use for this constraint. Supported options for regression are:
					- 'positive_gaussian_cocp'
					- 'negative_exponential_cocp'
					- 'gaussian_aocp'

				E.g. Negative constraint: constrained_domain=(4,5) and interval_func=lambda x: [[(-np.inf, 1 + x), (3 + x, np.inf)] for x in X]
						--> forbidden region is a parallelogram with vertices (4, 5), (4, 7), (5, 6), (5, 8)
						--> prior_type='negative_exponential_cocp' should be used

				E.g. Positive constraint: constrained_domain=(4,5) and interval_func=lambda x: [[(2, 2), (3, 3)] for _ in X]
						--> For x between 4 and 5, the only permitted values are 2 and 3
						--> prior_type='positive_gaussian_cocp' should be used (bimodal Gaussian)
		"""
		if prior_type not in self.dconstraints:
			self.dconstraints[prior_type] = []
		self.dconstraints[prior_type].append((constrained_domain, interval_func))

		# Create samples for inference.
		if prior_type == "positive_gaussian_cocp":
			total_nsamples = sum([len(ifunc(self._sample_from_hypercube(dom, 1))) * self.ocp_nsamples for dom, ifunc in self.dconstraints['positive_gaussian_cocp']])
			self._cr_pos_xsamples = torch.zeros(total_nsamples, self.Xdim)
			self._cr_pos_ysamples = torch.zeros(total_nsamples, self.Ydim)
			self._cr_ylens = []
			index = 0
			for dom, ifunc in self.dconstraints['positive_gaussian_cocp']:
				xsamp = self._sample_from_hypercube(dom, self.ocp_nsamples)
				intervals = ifunc(xsamp)
				self._cr_ylens.append(len(intervals[0]))
				sub_nsamples = len(intervals[0]) * self.ocp_nsamples
				self._cr_pos_xsamples[index:index+sub_nsamples,:] = xsamp.repeat(len(intervals[0]), 1)
				for i in range(len(intervals[0])):
					self._cr_pos_ysamples[index:index+self.ocp_nsamples,:] = torch.tensor([sublist[i][0] for sublist in intervals]).unsqueeze(dim=1)
					index += self.ocp_nsamples
		elif prior_type == "negative_exponential_cocp":
			total_nsamples = self.ocp_nsamples * len(self.dconstraints['negative_exponential_cocp'])
			self._cr_neg_xsamples = torch.zeros(total_nsamples, self.Xdim)
			index = 0
			for dom, _ in self.dconstraints['negative_exponential_cocp']:
				xsamp = self._sample_from_hypercube(dom, self.ocp_nsamples)
				self._cr_neg_xsamples[index:index+self.ocp_nsamples,:] = xsamp
				index += self.ocp_nsamples
		elif prior_type == "gaussian_aocp":
			total_nsamples = self.ocp_nsamples * len(self.dconstraints['gaussian_aocp'])
			self._cr_aocp_xsamples = torch.zeros(total_nsamples, self.Xdim)
			index = 0
			for dom, ifunc in self.dconstraints['gaussian_aocp']:
				xsamp = self._sample_from_hypercube(dom, self.ocp_nsamples)
				self._cr_aocp_xsamples[index:index+self.ocp_nsamples,:] = xsamp
				index += self.ocp_nsamples

		logging.info(f'[{self.uid}] Defined constrained region for `{prior_type}`.')

	def positive_gaussian_cocp(self):
		""" Conditional output-constrained prior: mixture of Gaussian.
			Assume uniform mixing weights for each mixture.
			Assume isotropic Gaussian. 
		"""
		nn_mean = self.forward(X=self._cr_pos_xsamples)
		index = 0
		log_prob = torch.tensor(0.0)
		for i, (dom, ifunc) in enumerate(self.dconstraints['positive_gaussian_cocp']):
			sub_nsamples = self._cr_ylens[i] * self.ocp_nsamples
			dist = MVN(self._cr_pos_ysamples[index:index+sub_nsamples,:], self.cocp_gaussian_sigma_c * torch.eye(self.Ydim)).log_prob(nn_mean[index:index+sub_nsamples,:])
			dist += torch.log(torch.tensor(1/self._cr_ylens[i]))
			log_prob += torch.logsumexp(torch.stack(dist.split(self.ocp_nsamples), dim=0), dim=0).sum()
			index += sub_nsamples
		return log_prob

	def _constraint_classifier(self, z):
		""" Sigmoidal constraint classfier. """
		return 0.25 * (torch.tanh(-self.cocp_expo_tau[0]*z) + 1) * (torch.tanh(-self.cocp_expo_tau[1]*z) + 1)

	def negative_exponential_cocp(self):
		""" Conditional output-constrained prior: Exponential. """
		penalty = torch.zeros(self.ocp_nsamples)
		nn_mean = self.forward(X=self._cr_neg_xsamples)
		index = 0
		for _, ifunc in self.dconstraints['negative_exponential_cocp']:
			i, j = index, index + self.ocp_nsamples
			nn_mean_subset = nn_mean[i:j,:]
			mult = torch.ones(nn_mean_subset.shape[0])
			intervals = ifunc(self._cr_neg_xsamples[i:j,:])
			marker = torch.tensor([-np.inf for _ in range(self.ocp_nsamples)])
			for k in range(len(intervals[0])):
				lbs = torch.tensor([sublist[k][0] for sublist in intervals])
				ubs = torch.tensor([sublist[k][1] for sublist in intervals])
				if all(marker < lbs):
					penalty += self._constraint_classifier(marker - nn_mean_subset[:,0]) * self._constraint_classifier(nn_mean_subset[:,0] - lbs)
				marker = ubs
			else:
				infs = torch.tensor([np.inf for _ in range(self.ocp_nsamples)])
				if all(marker < infs):
					penalty += self._constraint_classifier(marker - nn_mean_subset[:,0]) * self._constraint_classifier(nn_mean_subset[:,0] - infs)
			index += self.ocp_nsamples
		violation = penalty.sum() / (self.ocp_nsamples * len(self.dconstraints['negative_exponential_cocp']))
		log_prob = -self.cocp_expo_gamma * violation # p(w) = exp(- gamma * constraint_viol)
		return log_prob
	
	def gaussian_aocp_objective(self):
		""" Gaussian AOCP optimization objective. """
		pvals = torch.tensor(0.)
		index = 0
		for _, ifunc in self.dconstraints['gaussian_aocp']:
			i, j = index, index + self.ocp_nsamples
			xsamples = self._cr_aocp_xsamples[i:j,:]
			intervals = ifunc(xsamples)
			for k, x in enumerate(xsamples):
				# Compute approximate prior predictive.
				_mean = Variable(self._aocp_params[:,0].clone().data, requires_grad=True) 
				pp_mean = self.forward(x.unsqueeze(dim=1), weights=_mean).squeeze()
				pp_mean.backward(retain_graph=False)
				g = _mean.grad
				A = (torch.logsumexp(torch.stack((self._aocp_params[:,1], torch.zeros(self.nweights))), 0) ** 2) * torch.eye(self.nweights) 
				pp_std = (self.sigma_noise + (g @ A @ g)) ** 0.5
				pp_mean = self.forward(x.unsqueeze(dim=1), weights=self._aocp_params[:,0]).squeeze()

				# Calculate fraction of prior predictive violating the constraint.
				permitted = torch.tensor(0.)
				for lb, ub in intervals[k]:
					if ub == np.inf:
						permitted += 1.0 - Normal(pp_mean, pp_std).cdf(lb)
					elif lb == -np.inf:
						permitted += Normal(pp_mean, pp_std).cdf(ub)
					else:
						permitted += Normal(pp_mean, pp_std).cdf(ub) - Normal(pp_mean, pp_std).cdf(lb)
				pvals += 1 - permitted
			index += self.ocp_nsamples
		return pvals / len(self._cr_aocp_xsamples)


class ClassifierMixin:
	""" Classification-specific functionality. Output dimension must be > 1. 
		Number of output nodes = number of classes. Softmax applied to derive output probabilities.
		Note: Classes are 0-indexed.
	"""

	@property
	def _Ydim(self):
		return torch.max(self.Y_train).int().item() + 1

	def log_likelihood(self, batch_indices=None):
		""" Computes the likelihood. """
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

	def predict(self, bayes_samples, domain, return_probs=False):
		""" Generate BNN's prediction over `domain` for each sample.

			Args: 
				`bayes_samples` is a (M, self.nweights) tensor, representing M samples of the prior/posterior.
				`domain` is a (K, self.Xdim) tensor, representing K input points passed to the BNN.
				`return_probs` is a Boolean: true if we want to return the raw softmax probabilities instead of class predictions.

			Returns:
				Tensor of (M, K) of output classes (integer) across all posterior samples and all input points.
		"""
		probs = torch.tensor(np.apply_along_axis(lambda w: self.forward(X=domain, weights=torch.tensor(w)).numpy(), 1, bayes_samples)) # M x K x self.Ydim
		softprobs = torch.nn.Softmax(dim=2)(probs)
		if return_probs:
			return softprobs
		return softprobs.argmax(dim=2)

	def add_deterministic_constraint(self, constrained_domain, forbidden_classes, prior_type="positive_dirichlet_cocp"):
		""" Specify a (positive or negative) deterministic constraint. 

			Args:
				`constrained_domain` is a (x1_lower, x1_upper, x2_lower, x2_upper, ...) tuple, i.e. length is Xdim * 2,
					representing the lower and upper bounds of each input dimension of the constrained input region.
					Values of +/- np.inf will be replaced with the corresponding bounds of the training set.

				`forbidden_classes` is an integer list of FORBIDDEN classes for ALL points in `constrained_domain`.

				`prior_type` is the COCP/AOCP that we want to use for this constraint. Supported options for classification are:
					- 'positive_dirichlet_cocp'

				E.g. constrained_domain=(4,5,6,7) and forbidden_classes=[2,3,4]
						--> For the input rectangle bounded by (4,6), (5,6), (4,7), (5,7), output classes 2, 3, and 4 are forbidden.
		"""
		if not forbidden_classes or len(forbidden_classes) >= self.Ydim:
			logging.error("Number of forbidden classes must be between 0 and total number of classes (exclusive).")
			raise Exception

		if prior_type not in self.dconstraints:
			self.dconstraints[prior_type] = []
		self.dconstraints[prior_type].append((constrained_domain, forbidden_classes))
		
		# Prepare samples for inference.
		total_nsamples = self.ocp_nsamples * len(self.dconstraints[prior_type])
		self._cr_xsamples = torch.zeros(total_nsamples, self.Xdim)
		index = 0
		for domain, _ in self.dconstraints[prior_type]:
			xsamp = self._sample_from_hypercube(domain, self.ocp_nsamples)
			self._cr_xsamples[index:index+self.ocp_nsamples,:] = xsamp
			index += self.ocp_nsamples

		logging.info(f'[{self.uid}] Defined constrained region for `{prior_type}`.')

	def positive_dirichlet_cocp(self):
		""" Conditional output-constrained prior: Dirichlet. """
		log_prob = torch.tensor(0.)
		nn_mean = self.forward(X=self._cr_xsamples)
		index = 0
		for _, fclasses in self.dconstraints["positive_dirichlet_cocp"]:
			i, j = index, index + self.ocp_nsamples
			nn_mean_subset = nn_mean[i:j,:]
			nn_probs = torch.nn.Softmax(dim=1)(nn_mean_subset)
			dirprobs = self.cocp_dirichlet_gamma * torch.tensor(fclasses).bincount(minlength=self.Ydim).type(torch.float)
			dirprobs = self.cocp_dirichlet_gamma - (dirprobs * (1 - self.cocp_dirichlet_alpha))
			log_prob += Dirichlet(dirprobs).log_prob(nn_probs).sum()
			index += self.ocp_nsamples
		return log_prob


class BinaryClassifierMixin:
	""" Alternative setup for BINARY classification, necessary to learn AOCP.
		There is only a single node. Sigmoid applied to that node represents p(Y=1).
		Note: Classes are 0-indexed, i.e. labels are 0 or 1. 
	"""

	@property
	def _Ydim(self):
		return 1

	def _sigmoid(self, batch):
		""" Sigmoid fuction. """
		return torch.exp(batch) / (torch.exp(batch) + 1.0)

	def log_likelihood(self, batch_indices=None):
		""" Computes the likelihood. """
		if batch_indices is None:
			batch = self.X_train
			target = self.Y_train
			multiplier = 1
		else:
			batch = self.X_train[batch_indices]
			target = self.Y_train[batch_indices]
			multiplier = (self.N_train / len(batch_indices))
		target = target.type(torch.float32)
		means = self.forward(X=batch)
		probs = self._sigmoid(means).squeeze()
		terms = target * torch.log(probs) + (1.0 - target) * torch.log(1 - probs)
		return multiplier * terms.sum()

	def predict(self, bayes_samples, domain, return_probs=False):
		""" Generate BNN's prediction over `domain` for each sample.

			Args: 
				`bayes_samples` is a (M, self.nweights) tensor, representing M samples of the prior/posterior.
				`domain` is a (K, self.Xdim) tensor, representing K input points passed to the BNN.
				`return_probs` is a Boolean: true if we want to return the raw sigmoid probability p(Y=1) instead of class predictions.

			Returns:
				Boolean tensor of (M, K) of output classes (integer) across all posterior samples and all input points.
		"""
		probs = torch.tensor(np.apply_along_axis(lambda w: self._sigmoid(self.forward(X=domain, 
			weights=torch.tensor(w)).squeeze(dim=1)).numpy(), 1, bayes_samples))
		if return_probs:
			return probs
		return probs >= 0.5

	def add_deterministic_constraint(self, constrained_domain, desired_class, prior_type="gaussian_aocp"):
		""" Specify a (positive or negative) deterministic constraint. 

			Args:
				`constrained_domain` is a (x1_lower, x1_upper, x2_lower, x2_upper, ...) tuple, i.e. length is Xdim * 2,
					representing the lower and upper bounds of each input dimension of the constrained input region.
					Values of +/- np.inf will be replaced with the corresponding bounds of the training set.

				`desired_class` is 0 or 1, for the desired output of ALL points in `constrained_domain`.
					--> For binary classification, this fully specifies a deterministic constraint!

				`prior_type` is the COCP/AOCP that we want to use for this constraint. Supported options for binary classification are:
					- 'gaussian_aocp' 

				E.g. constrained_domain=(4,5,6,7) and desired_class=1
						--> For the input rectangle bounded by (4,6), (5,6), (4,7), (5,7), the output should be 1
		"""
		if prior_type not in self.dconstraints:
			self.dconstraints[prior_type] = []
		self.dconstraints[prior_type].append((constrained_domain, desired_class))
		
		total_nsamples = self.ocp_nsamples * len(self.dconstraints[prior_type])
		self._cr_det_xsamples = torch.zeros(total_nsamples, self.Xdim)
		index = 0
		for domain, _ in self.dconstraints[prior_type]:
			xsamp = self._sample_from_hypercube(domain, self.ocp_nsamples)
			self._cr_det_xsamples[index:index+self.ocp_nsamples,:] = xsamp
			index += self.ocp_nsamples

		logging.info(f'[{self.uid}] Defined constrained region for `{prior_type}`.')

	def add_probabilistic_constraint(self, constrained_domain, prob_func, prior_type="gaussian_aocp"):
		""" Specify a probabilistic constraint. 

			Args:
				`constrained_domain` is a (x1_lower, x1_upper, x2_lower, x2_upper, ...) tuple, i.e. length is Xdim * 2,
					representing the lower and upper bounds of each input dimension of the constrained input region.
					Values of +/- np.inf will be replaced with the corresponding bounds of the training set.

				`prob_func` is a function f:
					f takes as input a tensor X of shape (n_samples, self.Xdim) sampled from `constrained_domain`.
					f(X) evaluates to the (n_samples, ) tensor of desired output probability p(Y=1|X[i]) for each X[i].

				`prior_type` is the COCP/AOCP that we want to use for this constraint. Supported options for binary classification are:
					- 'gaussian_aocp' 

				E.g. constrained_domain=(1,2,0,1) and prob_func=lambda x: x[:, 1]
						--> For the input rectangle bounded by (1,0), (2,0), (1,1), (1,1), the output should be X_2
		"""
		if prior_type not in self.pconstraints:
			self.pconstraints[prior_type] = []
		self.pconstraints[prior_type].append((constrained_domain, prob_func))
		
		total_nsamples = self.ocp_nsamples * len(self.pconstraints[prior_type])
		self._cr_prob_xsamples = torch.zeros(total_nsamples, self.Xdim)
		index = 0
		for domain, _ in self.pconstraints[prior_type]:
			xsamp = self._sample_from_hypercube(domain, self.ocp_nsamples)
			self._cr_prob_xsamples[index:index+self.ocp_nsamples,:] = xsamp
			index += self.ocp_nsamples

		logging.info(f'[{self.uid}] Defined constrained region for `{prior_type}`.')

	def gaussian_aocp_objective(self):
		""" Gaussian AOCP optimization objective. """
		pvals = torch.zeros(1)
		index = 0
		denom = 0
		if 'gaussian_aocp' in self.dconstraints:
			for _, dclass in self.dconstraints['gaussian_aocp']:
				i, j = index, index + self.ocp_nsamples
				xsamples = self._cr_det_xsamples[i:j,:]
				for k, x in enumerate(xsamples):
					# Compute approximate prior predictive.
					_mean = Variable(self._aocp_params[:,0].clone().data, requires_grad=True) 
					pp_mean = self.forward(x.unsqueeze(dim=0), weights=_mean).squeeze()
					pp_mean.backward(retain_graph=False)
					b = _mean.grad
					A = (torch.logsumexp(torch.stack((self._aocp_params[:,1], torch.zeros(self.nweights))), 0) ** 2) * torch.eye(self.nweights)
					sigma2 = b @ A @ b
					kappa = (1 + (math.pi * sigma2 / 8)) ** (-0.5)
					p1 = self._sigmoid(kappa * (b @ self._aocp_params[:,0]))

					# Calculate fraction of prior predictive violating the constraint.
					pvals += abs(dclass - p1)
			denom += len(self._cr_det_xsamples)
		index = 0
		if 'gaussian_aocp' in self.pconstraints:
			for _, pfunc in self.pconstraints['gaussian_aocp']:
				i, j = index, index + self.ocp_nsamples
				xsamples = self._cr_prob_xsamples[i:j,:]
				p1prob = pfunc(xsamples)
				for k, x in enumerate(xsamples):
					# Compute approximate prior predictive.
					_mean = Variable(self._aocp_params[:,0].clone().data, requires_grad=True) 
					pp_mean = self.forward(x.unsqueeze(dim=0), weights=_mean).squeeze()
					pp_mean.backward(retain_graph=False)
					b = _mean.grad
					A = (torch.logsumexp(torch.stack((self._aocp_params[:,1], torch.zeros(self.nweights))), 0) ** 2) * torch.eye(self.nweights)
					sigma2 = b @ A @ b
					kappa = (1 + (math.pi * sigma2 / 8)) ** (-0.5)
					phat1 = self._sigmoid(kappa * (b @ self._aocp_params[:,0]))
					phat1 = torch.clamp(phat1, 0.001, 0.999)
					p1 = torch.clamp(p1prob[k], 0.001, 0.999)

					# Calculate fraction of prior predictive violating the constraint.
					# This is KL divergence between two Bernoullis -- the desired p(Y=1) and the current p(Y=1).
					# To preserve symmetry, we add both KL(p||q) and KL(q||p).
					pvals += p1 * (torch.log(p1) - torch.log(phat1)) + (1 - p1) * (torch.log(1 - p1) - torch.log(1 - phat1))
					pvals += phat1 * (torch.log(phat1) - torch.log(p1)) + (1 - phat1) * (torch.log(1 - phat1) - torch.log(1 - p1))
			denom += len(self._cr_prob_xsamples)
		return pvals / denom
