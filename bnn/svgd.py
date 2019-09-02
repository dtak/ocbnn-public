
"""
Class templates for Stein Variational Gradient Descent (SVGD) on OC-BNNs.
Implemented from: https://arxiv.org/pdf/1608.04471.pdf.
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


class BNNSVGD(BayesianNeuralNetwork):
	""" BNN inference using SVGD. """

	def __init__(self, **kwargs):
		pass

	def _compute_rbf_h(self):
		""" Compute h = (med ** 2) / log(n), where med is median of pairwise distances. """
		pdist = torch.nn.PairwiseDistance()
		fkernel = torch.zeros(self.nparticles, self.nparticles)
		for i in range(self.nparticles):
			fkernel[i] = pdist(self.particles[i], self.particles)

		fkernel = fkernel.triu(diagonal=1).flatten()
		med = fkernel[fkernel.nonzero()].median()
		return (med ** 2) / math.log(self.nparticles)

	def kernel_rbf(self, x1, x2, h):
		""" Compute the RBF kernel: k(x, x') = exp(-1/h * l2-norm(x, x')). """
		k = torch.norm(x1 - x2)
		k = (k ** 2) / -h
		return torch.exp(k)

	def single(self, batch_indices=None):
		""" Computes a single SVGD epoch and updates the particles. """
		h = self._compute_rbf_h()
		kernel = torch.zeros(self.nparticles, self.nparticles)

		# Repulsive term
		gradk_matrix = torch.zeros(self.nparticles, self.nparticles, self.nweights)
		for i in range(self.nparticles):
			grad_each_i = torch.zeros(self.nparticles, self.nweights)
			for j in range(self.nparticles):
				tempw = Variable(self.particles[j], requires_grad=True)
				tempw.grad = None
				k = self.kernel_rbf(tempw, self.particles[i], h)
				kernel[j, i] = k
				k.backward()
				grad_each_i[j] = tempw.grad
			gradk_matrix[i] = grad_each_i

		# Smoothed gradient term
		logp_matrix = torch.zeros(self.nparticles, self.nweights)
		for j in range(self.nparticles):
			self.weights = Variable(self.particles[j], requires_grad=True)
			self.weights.grad = None
			self.log_posterior(batch_indices).backward()
			logp_matrix[j] = self.weights.grad
		update = logp_matrix.unsqueeze(dim=0).repeat(self.nparticles, 1, 1)
		for i in range(self.nparticles):
			update[i] *= kernel[:,i].unsqueeze(dim=1)

		update += gradk_matrix
		update = update.mean(dim=1)
		return update

	def infer(self, verbose=True):
		""" Perform SVGD and collects samples. """
		infer_id = len(self.all_particles) + 1
		logging.info(f"[{self.uid}] Beginning SVGD inference #{infer_id}...")

		nbatches = self.config["nbatches"]
		start_time = time.time()
		self.nparticles = self.config["svgd_nparticles"]
		self.particles = self.weight_dist.sample(torch.Size([self.nparticles, self.nweights]))
		optimizer = torch.optim.Adagrad([self.particles], lr=self.config["svgd_init_lr"])
		for epoch in range(1, self.config["svgd_epochs"]+1):
			optimizer.zero_grad()
			if nbatches:
				batch_indices = torch.arange(epoch % nbatches, self.N_train, nbatches)
				update = self.single(batch_indices)
			else:
				update = self.single()
			self.particles.grad = -update
			optimizer.step()
			if verbose and epoch % 10 == 0:
				logging.info(f'[{self.uid}] Epoch {epoch} reached.')
		else:
			end_time = time.time()
			logging.info(f'[{self.uid}] SVGD ended after {self.config["svgd_epochs"]} epochs. Time took: {(end_time - start_time):.0f} seconds.')

		# Convert to numpy for evaluation and plotting.
		self.save(infer_id)
		self.particles = self.particles.data.numpy()
		self.all_particles.append((self.particles, self.config["prior_type"]))

	def save(self, infer_id):
		""" Save particles into memory. """
		torch.save(self.particles, f"history/{self.uid}_svgd{infer_id}.pt")


class BNNSVGDRegressor(BNNSVGD, BNNRegressor):
	""" BNN inference using SVGD for regression. """

	def __init__(self, uid="bnn-0", configfile="config.json"):
		""" Instantiates the MLP. """
		BNNRegressor.__init__(self, uid=uid, configfile=configfile)
		self.all_particles = []

	def predict(self, particles, domain):
		""" Generate BNN's prediction (forward pass) over the domain for each particle. """
		domain = np.expand_dims(domain, axis=1)
		return np.apply_along_axis(lambda w: self.forward(torch.Tensor(domain), weights=torch.Tensor(w)).numpy(), 1, particles).T

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
		for particles, pt in self.all_particles:
			results = self.predict(particles, domain).squeeze()
			means = np.mean(results, axis=1).squeeze()
			lower2 = np.percentile(results, 2.275, axis=1).squeeze()
			upper2 = np.percentile(results, 97.725, axis=1).squeeze()
			lower1 = np.percentile(results, 15.865, axis=1).squeeze()
			upper1 = np.percentile(results, 84.135, axis=1).squeeze()
			for line in results.T:
				plt.plot(domain, line, color=colors[prior_to_colors[pt]][0], alpha=colors[prior_to_colors[pt]][1], linewidth=0.8)

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
			plt.savefig(f'history/{self.uid}_svgdplot.png', dpi=self.config["plot_dpi"])
			logging.info(f'[{self.uid}] Predictive posterior plot saved to file.')
		elif action == 'show':
			plt.show()

	def test_neg_loglik(self):
		""" Compute negative log-likelihood of test set. """
		results = np.apply_along_axis(lambda w: self.forward(self.X_test, weights=torch.Tensor(w)).numpy(), 1, self.particles)
		means = torch.tensor(np.mean(results, axis=0))
		return -1 * MVN(means, self.config["sigma_noise"] * torch.eye(self.Ydim)).log_prob(self.Y_test).sum()

	def train_rmse(self):
		""" Compute RMSE of train set. """
		results = np.apply_along_axis(lambda w: self.forward(self.X_train, weights=torch.Tensor(w)).numpy(), 1, self.particles)
		means = torch.tensor(np.mean(results, axis=0))
		return torch.nn.MSELoss()(means, self.Y_train)

	def test_rmse(self):
		""" Compute RMSE of test set. """
		results = np.apply_along_axis(lambda w: self.forward(self.X_test, weights=torch.Tensor(w)).numpy(), 1, self.particles)
		means = torch.tensor(np.mean(results, axis=0))
		return torch.nn.MSELoss()(means, self.Y_test)


class BNNSVGDClassifier(BNNSVGD, BNNClassifier):
	""" BNN inference using SVGD for classification. """

	def __init__(self, uid="bnn-0", configfile="config.json"):
		""" Instantiates the MLP. """
		BNNClassifier.__init__(self, uid=uid, configfile=configfile)
		self.all_particles = []

	def predict(self, particles, domain):
		""" Generate BNN's prediction (forward pass) over the domain for each particle. """
		probs = np.apply_along_axis(lambda w: self.forward(torch.Tensor(domain), weights=torch.Tensor(w)).numpy(), 1, particles) # nsamples x test_size x nclasses
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

		# Plot results of the *most recent* set of particles.
		particles = self.all_particles[-1][0]
		ppd = np.arange(xlims[0], xlims[1], 0.1)
		Xte = np.transpose([np.tile(ppd, len(ppd)), np.repeat(ppd, len(ppd))])
		results = self.predict(particles, Xte)
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
			plt.savefig(f'history/{self.uid}_svgdplot.png', dpi=self.config["plot_dpi"])
			logging.info(f'[{self.uid}] Predictive posterior plot saved to file.')
		elif action == 'show':
			plt.show()

	def train_metrics(self, beta):
		""" Classification evaluation metrics on train set. """
		preds = self.predict(self.particles, self.X_train.numpy())
		accuracy = accuracy_score(self.Y_train, preds)
		if self.Ydim == 2:
			precision, recall, fscore, _ = precision_recall_fscore_support(self.Y_train, preds, beta=beta)
			return accuracy, precision, recall, fscore
		return accuracy, None, None, None

	def test_metrics(self, beta):
		""" Classification evaluation metrics on test set. """
		preds = self.predict(self.particles, self.X_test.numpy())
		accuracy = accuracy_score(self.Y_test, preds)
		if self.Ydim == 2:
			precision, recall, fscore, _ = precision_recall_fscore_support(self.Y_test, preds, beta=beta)
			return accuracy, precision, recall, fscore
		return accuracy, None, None, None
