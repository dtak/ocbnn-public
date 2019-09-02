
"""
Class templates for Hamiltonian Monte Carlo (HMC) on OC-BNNs.
Implemented from: https://arxiv.org/pdf/1206.1901.pdf.
"""

import numpy as np
import torch
import math
import time
import os

from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MVN

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .bnn import *


class BNNHMC(BayesianNeuralNetwork):
	""" BNN inference using HMC. """

	def __init__(self, **kwargs):
		pass

	def compute_potential(self):
		""" Computes U(q). """
		return -1 * self.log_posterior()

	def compute_kinetic(self, p):
		""" Computes K(p). """
		return 0.5 * (p ** 2).sum()

	def single(self):
		""" Computes a single iteration of HMC iteration and collects another sample of q. """

		epsilon = self.config["hmc_epsilon"]
		L = self.config["hmc_l"]

		current_p = Normal(0, 1).sample(torch.Size([self.nweights]))
		original_q = self.weights.data
		original_U = self.compute_potential()
		original_K = self.compute_kinetic(current_p)

		# Momentum half-step => position/momentum full-steps => momentum half-step => negate momentum
		self.weights.grad = None
		self.compute_potential().backward()

		current_p -= (epsilon / 2) * self.weights.grad
		for l in range(1, L + 1):
			self.weights.data += epsilon * current_p
			if l < L:
				self.weights.grad = None
				self.compute_potential().backward()
				current_p -= epsilon * self.weights.grad
		self.weights.grad = None
		self.compute_potential().backward()
		current_p -= (epsilon / 2) * self.weights.grad
		# current_p *= -1 # this step is redundant
		if sum(torch.isnan(self.weights)) > 0:
			logging.error("NaNs encountered in current set of weights.")
			raise Exception

		# Evaluate U(q) and K(p)
		current_U = self.compute_potential()
		current_K = self.compute_kinetic(current_p)

		# Metropolis-Hastings proposal
		if np.random.uniform() < torch.exp(original_U - current_U + original_K - current_K).item():
			self.accepts += 1
		else:
			self.rejects += 1
			self.weights.data = original_q

	def infer(self, verbose=True, monitor=0):
		""" Perform HMC and collects samples. """

		infer_id = len(self.all_samples) + 1
		logging.info(f"[{self.uid}] Beginning HMC inference #{infer_id}...")
		start_time = time.time()

		# Monitor randomly selected weights throughout inference process.
		if monitor:
			arange = np.arange(self.nweights)
			np.random.shuffle(arange)
			surveyed_dims = arange[:monitor]

		# Burn-in.
		# Print out whether first 10 samples are accepted/rejected to get a litmus test on hyperparameters.
		self.accepts = 0
		self.rejects = 0
		for i in range(1, self.config["hmc_nburnin"] + 1):
			self.single()
			if verbose and i < 11:
				logging.info(f'Iteration {i}: {self.accepts} accepts, {self.rejects} rejects.')
			if verbose and i % 500 == 0:
				logging.info(f'Iteration {i}. Acceptance rate is {100 * (self.accepts / i):.2f}%.')
		else:
			if self.config["hmc_nburnin"]:
				logging.info(f'All {self.config["hmc_nburnin"]} burn-in steps completed. Acceptance rate is {100 * (self.accepts / self.config["hmc_nburnin"]):.2f}%.')

		# Collect samples.
		logging.info('Collecting samples.')
		self.accepts = 0
		self.rejects = 0
		self.samples = []
		self.loglik_samples = []
		for i in range(1, self.config["hmc_nsamples"] * self.config["hmc_ninterval"] + 1):
			self.single()
			if self.config["use_tensorboard"]:
				loglik = self.log_likelihood()
				self.writer.add_scalar(f'Inf {infer_id}/Training LL', loglik, i)
				self.loglik_samples.append(loglik)
				for d in range(monitor):
					self.writer.add_scalar(f'Inf {infer_id}/Trace Weights/D{surveyed_dims[d]}', self.weights[surveyed_dims[d]], i)
			if i % self.config["hmc_ninterval"] == 0:
				self.samples.append(self.weights.data.clone())
			if verbose and i % 100 == 0:
				logging.info(f'{int(i / self.config["hmc_ninterval"])} samples collected. Acceptance rate is {100 * (self.accepts / i):.2f}%.')
		else:
			end_time = time.time()
			if self.config["hmc_nsamples"]:
				accept_perc = 100 * (self.accepts / self.config["hmc_nsamples"])
				logging.info(f'All {self.config["hmc_nsamples"]} samples collected. Acceptance rate is {accept_perc:.2f}%. Time took: {(end_time - start_time):.0f} seconds.')

		self.samples = torch.stack(self.samples)
		self.save(infer_id)
		self.samples = self.samples.data.numpy()
		self.all_samples.append((self.samples, self.config["prior_type"]))

	def plot_loglik(self, infer_id):
		""" Plot training log-likelihood. """
		ll_fig = plt.figure(figsize=tuple(self.config["plot_figsize"]))
		plt.rc('font',size=self.config["plot_fontsize"])
		plt.rc('axes',titlesize=self.config["plot_fontsize"])
		plt.rc('axes',labelsize=self.config["plot_fontsize"])
		plt.title('Training Log-Likelihood')
		plt.plot(self.loglik_samples)
		plt.xlabel('Iterations')
		plt.ylabel('Log-Likelihood')
		plt.show()

	def save(self, infer_id):
		""" Save all samples into memory. """
		torch.save(self.samples, f"history/{self.uid}_hmc{infer_id}.pt")


class BNNHMCRegressor(BNNHMC, BNNRegressor):
	""" BNN inference using HMC for regression. """

	def __init__(self, uid="bnn-0", configfile="config.json"):
		""" Instantiates the MLP. """
		BNNRegressor.__init__(self, uid=uid, configfile=configfile)
		self.all_samples = []

	def predict(self, samples, domain):
		""" Generate BNN's prediction (forward pass) over the domain for each sample. """
		domain = np.expand_dims(domain, axis=1)
		return np.apply_along_axis(lambda w: self.forward(torch.Tensor(domain), weights=torch.Tensor(w)).numpy(), 1, samples).T

	def plot_pp(self, plot_title, domain, ylims, action='save', addons=None):
		""" Plot posterior predictive. """

		# Configurations.
		assert(self.Xdim == 1)
		pp_fig = plt.figure(figsize=tuple(self.config["plot_figsize"]))
		plt.gca().set_ylim(ylims[0], ylims[1])
		plt.gca().set_xlim(domain[0], domain[-1])
		plt.rc('font',size=self.config["plot_fontsize"])
		plt.rc('axes',titlesize=self.config["plot_fontsize"])
		plt.rc('axes',labelsize=self.config["plot_fontsize"])
		plt.title(plot_title)
		plt.xlabel('X')
		plt.ylabel('φ(X)')
		plt.tight_layout()

		# For each inference run, plot results.
		colors = {'blue': ('b', 0.5, 0.15), 'red': ('#E41A1C', 0.6), 'green': ('#00B200', 0.3, 0.15), 'gray': ('k', 0.18, 0.08)}
		prior_to_colors = {'gaussian': 'gray', 'oc_positive': 'blue', 'oc_negative': 'blue'}
		for samples, pt in self.all_samples:
			results = self.predict(samples, domain).squeeze()
			means = np.mean(results, axis=1).squeeze()
			lower2 = np.percentile(results, 2.275, axis=1).squeeze()
			upper2 = np.percentile(results, 97.725, axis=1).squeeze()
			lower1 = np.percentile(results, 15.865, axis=1).squeeze()
			upper1 = np.percentile(results, 84.135, axis=1).squeeze()
			plt.fill_between(domain, lower1, upper1, facecolor=colors[prior_to_colors[pt]][0], alpha=colors[prior_to_colors[pt]][1])
			plt.fill_between(domain, lower2, upper2, facecolor=colors[prior_to_colors[pt]][0], alpha=colors[prior_to_colors[pt]][2])
			plt.plot(domain, means, color=colors[prior_to_colors[pt]][0], zorder=99)

		# Plot negative regions.
		if hasattr(self, 'pconstraints'):
			for fd, ff in self.pconstraints:
				x_lower, x_upper = fd
				dom = np.arange(x_lower, x_upper, 0.05)
				plt.plot(dom, ff(dom), color=colors['green'][0])
				plt.fill_between(dom, ff(dom) - 0.5, ff(dom) + 0.5, facecolor=colors['green'][0], alpha=colors['green'][1])
		if hasattr(self, 'nconstraints') and addons is not None:
			addons()

		# Plot training data.
		plt.plot(self.X_train.data.numpy(), self.Y_train.data.numpy(), 'kx')

		# Save and close.
		if action == 'save':
			plt.savefig(f'history/{self.uid}_hmcplot.png', dpi=self.config["plot_dpi"])
			logging.info(f'[{self.uid}] Predictive posterior plot saved to file.')
		elif action == 'show':
			plt.show()

	def test_neg_loglik(self):
		""" Compute negative log-likelihood of test set. """
		results = np.apply_along_axis(lambda w: self.forward(self.X_test, weights=torch.Tensor(w)).numpy(), 1, self.samples)
		means = torch.tensor(np.mean(results, axis=0))
		return -1 * MVN(means, self.config["sigma_noise"] * torch.eye(self.Ydim)).log_prob(self.Y_test).sum()

	def train_rmse(self):
		""" Compute RMSE of train set. """
		results = np.apply_along_axis(lambda w: self.forward(self.X_train, weights=torch.Tensor(w)).numpy(), 1, self.samples)
		means = torch.tensor(np.mean(results, axis=0))
		return torch.nn.MSELoss()(means, self.Y_train)

	def test_rmse(self):
		""" Compute RMSE of test set. """
		results = np.apply_along_axis(lambda w: self.forward(self.X_test, weights=torch.Tensor(w)).numpy(), 1, self.samples)
		means = torch.tensor(np.mean(results, axis=0))
		return torch.nn.MSELoss()(means, self.Y_test)


class BNNHMCClassifier(BNNHMC, BNNClassifier):
	""" BNN inference using Hamiltonian Monte Carlo for classification tasks. """

	def __init__(self, uid="bnn-0", configfile="config.json"):
		""" Instantiates the MLP. """
		BNNClassifier.__init__(self, uid=uid, configfile=configfile)
		self.all_samples = []

	def predict(self, samples, domain):
		""" Generate BNN's prediction (forward pass) over the domain for each sample. """
		probs = np.apply_along_axis(lambda w: self.forward(torch.Tensor(domain), weights=torch.Tensor(w)).numpy(), 1, samples) # nsamples x test_size x nclasses
		softprobs = torch.nn.Softmax(dim=2)(torch.tensor(probs))
		preds = softprobs.argmax(dim=2)
		return preds.data.numpy().T

	def plot_pp(self, plot_title, xlims, ylims, ptype='contour', action='save'):
		""" Plot posterior predictive. """

		# Configurations.
		assert(self.Xdim == 2)
		pp_fig = plt.figure(figsize=tuple(self.config["plot_figsize"]))
		plt.gca().set_xlim(xlims[0], xlims[1])
		plt.gca().set_ylim(ylims[0], ylims[1])
		plt.rc('font',size=self.config["plot_fontsize"])
		plt.rc('axes',titlesize=self.config["plot_fontsize"])
		plt.rc('axes',labelsize=self.config["plot_fontsize"])
		plt.title(plot_title)
		plt.xlabel(r'$X_1$')
		plt.ylabel(r'$X_2$')
		plt.tight_layout()

		# Plot results of the *most recent* set of samples.
		samples = self.all_samples[-1][0]
		ppd = np.arange(xlims[0], xlims[1], 0.1)
		Xte = np.transpose([np.tile(ppd, len(ppd)), np.repeat(ppd, len(ppd))])
		results = self.predict(samples, Xte)
		results_count = np.apply_along_axis(lambda v: np.bincount(v, minlength=3), 1, results)
		results_count = np.apply_along_axis(lambda v: v / results.shape[1], 1, results_count)
		colors = [tuple(v) for v in results_count.tolist()]
		if ptype == 'simplex':
			plt.scatter(Xte[:, 0], Xte[:, 1], marker='o', c=range(len(results)), cmap=ListedColormap(colors))
		elif ptype == 'contour':
			levels = [0.75, 0.9, 1.0]
			contourcolors = {'red': '#E41A1C', 'green': '#00B200', 'blue': 'b'}
			contouralpha = [0.7, 0.5]
			CSR1 = plt.gca().contourf(Xte[:, 0].reshape(100, 100), Xte[:, 1].reshape(100, 100),
				results_count[:,0].reshape(100, 100), levels=levels[1:], colors=contourcolors['red'], alpha=contouralpha[0])
			CSG1 = plt.gca().contourf(Xte[:, 0].reshape(100, 100), Xte[:, 1].reshape(100, 100),
				results_count[:,1].reshape(100, 100), levels=levels[1:], colors=contourcolors['green'], alpha=contouralpha[0])
			CSB1 = plt.gca().contourf(Xte[:, 0].reshape(100, 100), Xte[:, 1].reshape(100, 100),
				results_count[:,2].reshape(100, 100), levels=levels[1:], colors=contourcolors['blue'], alpha=contouralpha[0])
			CSR2 = plt.gca().contourf(Xte[:, 0].reshape(100, 100), Xte[:, 1].reshape(100, 100),
				results_count[:,0].reshape(100, 100), levels=levels[:2], colors=contourcolors['red'], alpha=contouralpha[1])
			CSG2 = plt.gca().contourf(Xte[:, 0].reshape(100, 100), Xte[:, 1].reshape(100, 100),
				results_count[:,1].reshape(100, 100), levels=levels[:2], colors=contourcolors['green'], alpha=contouralpha[1])
			CSB2 = plt.gca().contourf(Xte[:, 0].reshape(100, 100), Xte[:, 1].reshape(100, 100),
				results_count[:,2].reshape(100, 100), levels=levels[:2], colors=contourcolors['blue'], alpha=contouralpha[1])

		# Plot negative regions.
		if hasattr(self, 'pconstraints'):
			for xbounds, yset in self.pconstraints:
				x1_lower, x1_upper, x2_lower, x2_upper = xbounds
				plt.gca().add_patch(plt.Rectangle((x1_lower, x2_lower), x1_upper - x1_lower, x2_upper - x2_lower,
					facecolor='#00B200', linewidth=2.0, edgecolor='k'))

		# Plot training data.
		Xtr, Ytr = self.X_train.data.numpy(), self.Y_train.squeeze().data.numpy()
		for cls in range(self.Ydim):
			plt.plot(Xtr[Ytr == cls][:, 0], Xtr[Ytr == cls][:, 1], color=np.bincount([cls], minlength=3),
				marker='o', markeredgecolor='k', markeredgewidth=2.0, linewidth=0)

		# Save and close.
		if action == 'save':
			plt.savefig(f'history/{self.uid}_hmcplot.png', dpi=self.config["plot_dpi"])
			logging.info(f'[{self.uid}] Predictive posterior plot saved to file.')
		elif action == 'show':
			plt.show()

	def train_metrics(self, beta):
		""" Classification evaluation metrics on train set. """
		preds = self.predict(self.samples, self.X_train.numpy())
		accuracy = accuracy_score(self.Y_train, preds)
		if self.Ydim == 2:
			precision, recall, fscore, _ = precision_recall_fscore_support(self.Y_train, preds, beta=beta)
			return accuracy, precision, recall, fscore
		return accuracy, None, None, None

	def test_metrics(self, beta):
		""" Classification evaluation metrics on test set. """
		preds = self.predict(self.samples, self.X_test.numpy())
		accuracy = accuracy_score(self.Y_test, preds)
		if self.Ydim == 2:
			precision, recall, fscore, _ = precision_recall_fscore_support(self.Y_test, preds, beta=beta)
			return accuracy, precision, recall, fscore
		return accuracy, None, None, None
