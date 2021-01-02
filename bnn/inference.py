
"""
Mixin classes for inference.

Adding your own inference class:
	- An inference class must contain the `self.infer()` method.
	- Running `infer()` should append (samples, inference_name) to `self.all_bayes_samples`,
		where `samples` is a (self.infer_nsamples, self.nweights) tensor of posterior samples
		and  `inference_name` is a string identifier for downstream purposes (e.g. plotting).
	- See the examples below for a template.
"""

import numpy as np
import torch
import logging
import math
import time
import os

from torch.autograd import Variable
from torch.distributions.normal import Normal


class HMCMixin:
	""" BNN inference using Hamiltonian Monte Carlo (HMC). 
		Implemented from: https://arxiv.org/pdf/1206.1901.pdf.
	"""

	def _compute_potential(self, with_data):
		""" Computes U(q). """
		if with_data:
			return -1 * self.log_posterior()
		return -1 * self.log_prior()

	def _compute_kinetic(self, p):
		""" Computes K(p). """
		return 0.5 * (p ** 2).sum()

	def single(self, with_data):
		""" Computes a single iteration of HMC iteration and collects another sample of q. """

		epsilon = self.hmc_epsilon
		L = self.hmc_l

		current_p = Normal(0, 1).sample(torch.Size([self.nweights]))
		original_q = self.weights.data
		original_U = self._compute_potential(with_data)
		original_K = self._compute_kinetic(current_p)

		# Momentum half-step => position/momentum full-steps => momentum half-step => negate momentum
		self.weights.grad = None
		self._compute_potential(with_data).backward()

		current_p -= (epsilon / 2) * self.weights.grad
		for l in range(1, L + 1):
			self.weights.data += epsilon * current_p
			if l < L:
				self.weights.grad = None
				self._compute_potential(with_data).backward()
				current_p -= epsilon * self.weights.grad
		self.weights.grad = None
		self._compute_potential(with_data).backward()
		current_p -= (epsilon / 2) * self.weights.grad
		# current_p *= -1 # this step is computationally redundant
		if sum(torch.isnan(self.weights)) > 0:
			logging.error("NaNs encountered in current set of weights.")
			raise Exception

		# Evaluate U(q) and K(p)
		current_U = self._compute_potential(with_data)
		current_K = self._compute_kinetic(current_p)

		# Metropolis-Hastings proposal
		if np.random.uniform() < torch.exp(original_U - current_U + original_K - current_K).item():
			self.accepts += 1
		else:
			self.rejects += 1
			self.weights.data = original_q

	def infer(self, verbose=True, with_data=True):
		""" Perform HMC and collects samples.
			If `with_data`, sample from posterior, else, sample from prior. 
			Note that unlike the other inference methods, HMC does not allow batched training data.  
		"""
		infer_id = len(self.all_bayes_samples) + 1
		logging.info(f"[{self.uid}] Posterior Sample #{infer_id}: Beginning HMC inference...")
		start_time = time.time()

		# Burn-in.
		# Print out whether first 10 samples are accepted/rejected to get a litmus test on hyperparameters.
		self.accepts = 0
		self.rejects = 0
		for i in range(1, self.hmc_nburnin + 1):
			self.single(with_data)
			if verbose and i < 11:
				logging.info(f'  Iteration {i} (litmus): {self.accepts} accepts, {self.rejects} rejects.')
			if verbose and i % 500 == 0:
				logging.info(f'  Iteration {i}: Acceptance rate is {100 * (self.accepts / i):.2f}%.')
		else:
			if self.hmc_nburnin:
				logging.info(f'  All {self.hmc_nburnin} burn-in steps completed. Acceptance rate is {100 * (self.accepts / self.hmc_nburnin):.2f}%.')

		# Collect samples.
		logging.info('  Collecting samples now...')
		self.accepts = 0
		self.rejects = 0
		samples = []
		for i in range(1, self.infer_nsamples * self.hmc_ninterval + 1):
			self.single(with_data)
			if i % self.hmc_ninterval == 0:
				samples.append(self.weights.data.clone())
			if verbose and i % (10 * self.hmc_ninterval) == 0:
				accept_perc = 100 * (self.accepts / i)
				logging.info(f'  {int(i / self.hmc_ninterval)} samples collected. Acceptance rate is {accept_perc:.2f}%.')
		else:
			end_time = time.time()
			if self.infer_nsamples:
				accept_perc = 100 * (self.accepts / (self.infer_nsamples * self.hmc_ninterval))
				logging.info(f'  All {self.infer_nsamples} samples collected. Acceptance rate is {accept_perc:.2f}%. Time took: {(end_time - start_time):.0f} seconds.')

		samples = torch.stack(samples)
		torch.save(samples, f"history/{self.uid}_hmc{infer_id}.pt")
		self.all_bayes_samples.append((samples, f"hmc_{'ocbnn' if self.use_ocbnn else 'baseline'}"))
		logging.info(f'[{self.uid}] Posterior Sample #{infer_id}: HMC inference completed. Samples saved as `history/{self.uid}_hmc{infer_id}.pt`.')


class BBBMixin:
	""" BNN inference using Bayes by Backprop (BBB). 
		Implemented from: https://arxiv.org/pdf/1505.05424.pdf.
	"""

	def sample_q(self, k, q_params=None):
		""" Sample a set of weights from current variational parameters. """
		if q_params is None:
			q_params = self.q_params
		mean, log_std = self.q_params[:,0], q_params[:,1]
		std = torch.logsumexp(torch.stack((log_std, torch.zeros(self.nweights))), 0)
		weights = mean + torch.randn(k, self.nweights) * std
		logq = Normal(mean, std).log_prob(weights)
		return weights, logq

	def compute_elbo(self, with_data, batch_indices=None):
		""" Computes ELBO using samples from variational parameters. """
		weights, logq = self.sample_q(self.bbb_esamples)
		elbo = 0
		for j in range(self.bbb_esamples):
			self.weights = weights[j]
			if with_data:
				elbo += self.log_posterior(batch_indices) - logq[j].sum()
			else:
				elbo += self.log_prior() - logq[j].sum()
		elbo /= self.bbb_esamples
		return elbo

	def infer(self, verbose=True, with_data=True):
		""" Perform BBB and optimize variational parameters.

			BBB allows for batched inference, which is automatically set up by defining the config `nbatches`. 
			
			Both the final variational parameters, as well as samples from these parameters are saved.
			The samples are loaded into `self.all_bayes_samples` for downstream prediction.  

			If `with_data`, sample from posterior, else, sample from prior. 
		"""
		infer_id = len(self.all_bayes_samples) + 1
		logging.info(f"[{self.uid}] Posterior Sample #{infer_id}: Beginning BBB inference...")
		start_time = time.time()

		# Initialize variational parameters.
		init_means = self.bbb_init_mean + torch.randn(self.nweights, 1)
		init_log_stds = self.bbb_init_std * torch.ones(self.nweights, 1)
		self.q_params = Variable(torch.cat([init_means, init_log_stds], dim=1), requires_grad=True)
		optimizer = torch.optim.Adagrad([self.q_params], lr=self.bbb_init_lr)

		# Optimization loop.
		self.elbos = []
		for epoch in range(1, self.bbb_epochs+1):
			optimizer.zero_grad()
			if self.nbatches:
				batch_indices = torch.arange(epoch % self.nbatches, self.N_train, self.nbatches)
				elbo = self.compute_elbo(with_data, batch_indices)
			else:
				elbo = self.compute_elbo(with_data)
			loss = -1 * elbo
			loss.backward()
			optimizer.step()
			self.elbos.append(elbo.item())
			if verbose and epoch % 100 == 0:
				logging.info(f'  Epoch {epoch}: {self.elbos[-1]:.2f}')
		else:
			end_time = time.time()
			logging.info(f'  {self.bbb_epochs} epochs completed. Final ELBO is {self.elbos[-1]:.2f}. Time took: {(end_time - start_time):.0f} seconds.')

		self.q_params.detach_()
		torch.save(self.q_params, f"history/{self.uid}_bbb{infer_id}_qparams.pt")
		samples, _ = self.sample_q(self.infer_nsamples)
		self.all_bayes_samples.append((samples, f"bbb_{'ocbnn' if self.use_ocbnn else 'baseline'}"))
		torch.save(samples, f"history/{self.uid}_bbb{infer_id}.pt")
		logging.info(f'[{self.uid}] Posterior Sample #{infer_id}: BBB inference completed.') 
		logging.info(f'[{self.uid}] Variational parameters saved as `history/{self.uid}_bbb{infer_id}_qparams.pt`. Samples saved as `history/{self.uid}_bbb{infer_id}.pt`.') 


class SVGDMixin:
	""" BNN inference using Stein Variational Gradient Descent (SVGD).
		Implemented from: https://arxiv.org/pdf/1608.04471.pdf.
	"""

	def _compute_rbf_h(self):
		""" Compute h = (med ** 2) / log(n), where med is median of pairwise distances. """
		pdist = torch.nn.PairwiseDistance()
		fkernel = torch.zeros(self.infer_nsamples, self.infer_nsamples)
		for i in range(self.infer_nsamples):
			fkernel[i] = pdist(self.particles[i], self.particles)

		fkernel = fkernel.triu(diagonal=1).flatten()
		med = fkernel[fkernel.nonzero()].median()
		return (med ** 2) / math.log(self.infer_nsamples)

	def kernel_rbf(self, x1, x2, h):
		""" Compute the RBF kernel: k(x, x') = exp(-1/h * l2-norm(x, x')). """
		k = torch.norm(x1 - x2)
		k = (k ** 2) / -h
		return torch.exp(k)

	def single(self, with_data, batch_indices=None):
		""" Computes a single SVGD epoch and updates the particles. """
		h = self._compute_rbf_h()
		kernel = torch.zeros(self.infer_nsamples, self.infer_nsamples)

		# Repulsive term
		gradk_matrix = torch.zeros(self.infer_nsamples, self.infer_nsamples, self.nweights)
		for i in range(self.infer_nsamples):
			grad_each_i = torch.zeros(self.infer_nsamples, self.nweights)
			for j in range(self.infer_nsamples):
				tempw = Variable(self.particles[j], requires_grad=True)
				tempw.grad = None
				k = self.kernel_rbf(tempw, self.particles[i], h)
				kernel[j, i] = k
				k.backward()
				grad_each_i[j] = tempw.grad
			gradk_matrix[i] = grad_each_i

		# Smoothed gradient term
		logp_matrix = torch.zeros(self.infer_nsamples, self.nweights)
		for j in range(self.infer_nsamples):
			self.weights = Variable(self.particles[j], requires_grad=True)
			self.weights.grad = None
			if with_data:
				self.log_posterior(batch_indices).backward()
			else:
				self.log_prior().backward()
			logp_matrix[j] = self.weights.grad
		update = logp_matrix.unsqueeze(dim=0).repeat(self.infer_nsamples, 1, 1)
		for i in range(self.infer_nsamples):
			update[i] *= kernel[:,i].unsqueeze(dim=1)

		update += gradk_matrix
		update = update.mean(dim=1)
		return update

	def infer(self, verbose=True, with_data=True):
		""" Perform SVGD and collects samples (particles).
			If `with_data`, sample from posterior, else, sample from prior.
			SVGD allows for batched inference, which is automatically set up by defining the config `nbatches`.   
		"""
		infer_id = len(self.all_bayes_samples) + 1
		logging.info(f"[{self.uid}] Posterior Sample #{infer_id}: Beginning SVGD inference...")
		start_time = time.time()

		start_time = time.time()
		self.particles = Normal(0, self.sigma_w).sample(torch.Size([self.infer_nsamples, self.nweights]))
		optimizer = torch.optim.Adagrad([self.particles], lr=self.svgd_init_lr)
		for epoch in range(1, self.svgd_epochs+1):
			optimizer.zero_grad()
			if self.nbatches:
				batch_indices = torch.arange(epoch % self.nbatches, self.N_train, self.nbatches)
				update = self.single(with_data, batch_indices)
			else:
				update = self.single(with_data)
			self.particles.grad = -update
			optimizer.step()
			if verbose and epoch % 10 == 0:
				logging.info(f'  Epoch {epoch} reached.')
		else:
			end_time = time.time()
			logging.info(f'  {self.svgd_epochs} epochs completed. Time took: {(end_time - start_time):.0f} seconds.')

		# Convert to numpy for evaluation and plotting.
		self.particles.detach_()
		torch.save(self.particles, f"history/{self.uid}_svgd{infer_id}.pt")
		self.all_bayes_samples.append((self.particles.data.clone(), f"svgd_{'ocbnn' if self.use_ocbnn else 'baseline'}"))
		logging.info(f'[{self.uid}] Posterior Sample #{infer_id}: SVGD inference completed. Samples saved as `history/{self.uid}_svgd{infer_id}.pt`.') 


class SGLDMixin:
	""" BNN inference using Stochastic Gradient Langevin Dynamics (SGLD).
		Implemented from: https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf.
	"""

	def get_stepsize(self, t):
		""" Computes stepsize: epsilon_t = a(b + t)^(-gamma). """
		return self.sgld_epa * ((self.sgld_epb + t) ** (-1 * self.sgld_epgamma))

	def single(self, with_data, batch_indices=None):
		""" Runs a single iteration of SGLD. """
		update = torch.zeros(self.nweights)

		# Prior term.
		self.weights.grad = None
		self.log_prior().backward()
		update += self.weights.grad

		# Batched likelihood term.
		if with_data:
			self.weights.grad = None
			self.log_likelihood(batch_indices).backward()
			update += self.weights.grad

		return update

	def infer(self, verbose=True, with_data=True):
		""" Run SGLD and collect samples.
			If `with_data`, sample from posterior, else, sample from prior.
			SGLD allows for batched inference, which is automatically set up by defining the config `nbatches`.  
		"""
		infer_id = len(self.all_bayes_samples) + 1
		logging.info(f"[{self.uid}] Posterior Sample #{infer_id}: Beginning SGLD inference...")
		start_time = time.time()

		# Burn-in stage.
		for t in range(1, self.sgld_nburnin + 1):
			epsilon = self.get_stepsize(t)
			if self.nbatches:
				batch_indices = torch.arange(t % self.nbatches, self.N_train, self.nbatches)
				update = self.single(with_data, batch_indices)
			else:
				update = self.single(with_data)
			update *= epsilon / 2
			update += Normal(torch.zeros(self.nweights), (epsilon ** 0.5) * torch.ones(self.nweights)).sample()
			self.weights.data += update
			if verbose and t % 1000 == 0:
				if torch.sum(torch.isnan(self.weights)) > 0:
					logging.error(f"[{self.uid}] NaNs encountered in current set of weights.")
					raise Exception
				logging.info(f'  {t} iterations burnt-in.')
		else:
			if self.sgld_nburnin:
				logging.info(f'  All {self.sgld_nburnin} burn-in steps completed.')

		# Collecting SGLD samples.
		logging.info('  Collecting samples now...')
		samples = []
		for t in range(self.sgld_nburnin + 1, self.sgld_nburnin + self.infer_nsamples * self.sgld_ninterval + 1):
			epsilon = self.get_stepsize(t)
			if self.nbatches:
				batch_indices = torch.arange(t % self.nbatches, self.N_train, self.nbatches)
				update = self.single(with_data, batch_indices)
			else:
				update = self.single(with_data)
			update *= epsilon / 2
			update += Normal(torch.zeros(self.nweights), (epsilon ** 0.5) * torch.ones(self.nweights)).sample()
			self.weights.data += update
			if (t - self.sgld_nburnin) % self.sgld_ninterval == 0:
				samples.append(self.weights.data.clone())
			if verbose and (t - self.sgld_nburnin) % 100 == 0:
				logging.info(f'  {int((t - self.sgld_nburnin) / self.sgld_ninterval)} samples collected.')
		else:
			end_time = time.time()
			if self.infer_nsamples:
				logging.info(f'  All {self.infer_nsamples} samples collected. Time took: {(end_time - start_time):.0f} seconds.')

		samples = torch.stack(samples)
		torch.save(samples, f"history/{self.uid}_sgld{infer_id}.pt")
		self.all_bayes_samples.append((samples, f"sgld_{'ocbnn' if self.use_ocbnn else 'baseline'}"))
		logging.info(f'[{self.uid}] Posterior Sample #{infer_id}: SGLD inference completed. Samples saved as `history/{self.uid}_sgld{infer_id}.pt`.')
