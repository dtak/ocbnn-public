
"""
Basic example showing how to use this library.

"""

import numpy as np
import torch
import logging
import matplotlib.pyplot as plt

from bnn import BNNHMCRegressor
from bnn.utils import plot_1dregressor, COLORS
from data.dataloader import toy1


def example():

	################################################################################
	## Part 1: Running a baseline BNN without constraints, using a Gaussian prior.
	################################################################################
	
	# First, instantiate the BNN. Various hyperparameters and such are defined as a YAML config file.
	# See `config.yaml` (root level of the repo) for explanations of each config.
	# When the BNN is instantiated, a copy of the YAML file will be saved in `history/`.
	bnn = BNNHMCRegressor(uid="example2", configfile="repro/example.yaml")

	# Load the dataset.
	# See `data/dataloader.py` to add your own dataset.
	bnn.load(**toy1())

	# Conduct inference. The inference method is already defined by the BNN class you instantiated.
	# In this case, we use HMC.
	# After inference is complete, the samples are saved in `history/` as a .pt file.
	bnn.infer()

	# Having collected posterior samples, let us try plotting the posterior predictive.
	# Plots are automatically saved in the `history/` folder too.
	domain = torch.arange(-5, 5, 0.05).unsqueeze(dim=1)
	plot_1dregressor(bnn, plot_title="Posterior Predictive", domain=domain, ylims=(-9, 7))

	# BNN inference takes time. If you want to debug the code, you can activate debug mode.
	# In debug mode, only a few iterations of inference is run.
	bnn.clear_all_samples()
	bnn.debug_mode()
	bnn.infer()
	plot_1dregressor(bnn, plot_title="Debug Mode Posterior Predictive", domain=domain, ylims=(-9, 7))

	# By the way, here's how to load an existing posterior sample file from memory.
	# We've provided a pretrained set of HMC samples. 
	bnn.clear_all_samples()
	bnn.switch_off_debug_mode()
	bnn.load_bayes_samples('repro/example_hmc1.pt', 'hmc_gaussian')
	plot_1dregressor(bnn, plot_title="Pretrained Posterior Predictive", domain=domain, ylims=(-9, 7))


	################################################################################
	## Part 2: Specifying and obeying output constraints with OC-BNNs.
	################################################################################

	# We will add negative constraints.
	# See the docstring in `bnn/base.py` for an explanation on how to do so.
	# When a constraint is added, the BNN automatically uses output-constrained priors for inference.
	def ifunc(X): return [[(2.5, 3.0)] for _ in range(len(X))]
	bnn.add_deterministic_constraint(constrained_domain=(-0.3, 0.3), interval_func=ifunc, prior_type="negative_exponential_cocp")
	bnn.update_config(use_ocbnn=True)
	bnn.infer()

	# Let's plot the predictive distribution again, along with the constraint we specified.
	# We also plot the baseline posterior predictive in Part 1 for comparison.
	def addons():
		dom = np.arange(-0.3, 0.3, 0.05)
		plt.fill_between(dom, 3.0, plt.ylim()[1], facecolor=COLORS['red'][0], alpha=0.5, zorder=101)
		plt.fill_between(dom, plt.ylim()[0], 2.5, facecolor=COLORS['red'][0], alpha=0.5, zorder=101)
	plot_1dregressor(bnn, plot_title="Posterior Predictive With Constraints", domain=domain, ylims=(-9, 7), addons=addons)

	# Finally, here's both our pretrained HMC samples for comparison.
	bnn.clear_all_samples()
	bnn.load_bayes_samples('repro/example_hmc1.pt', 'hmc_gaussian')
	bnn.load_bayes_samples('repro/example_hmc2.pt', 'hmc_ocbnn')
	plot_1dregressor(bnn, plot_title="Pretrained Posterior Predictive With Constraints", domain=domain, ylims=(-9, 7), addons=addons)



if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(name)s-%(levelname)s: %(message)s")
	logging.info("Running OC-BNN library...")
	example()
	logging.info("Completed.")
