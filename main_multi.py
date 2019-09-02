
"""
A high-dimensional example on the "Energy Efficiency" UCI dataset: https://archive.ics.uci.edu/ml/datasets/Energy+efficiency.

This is a regression test with (standardized) 8-dimensional input and 1-dimensional output.

We use positive constraints:
	- For points where dimension X_3 is in the range [-2.0, -1.0], the output Y is constrained to be 0.
	
Inference is done with SVGD, and the train set is batched into 12 batches. 
We compute evaluation metrics on both the OC-BNN and baseline vanilla BNN.
We also plot the regression on X_3 on the test set.

"""

import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from bnn.svgd import *
from data.uci import energyset


def plot_energy(ocbnn):
	plt.figure(figsize=(10,4))
	plt.title("Regression on X_3 (test set)", fontsize=14)

	plt.rc('font',size=ocbnn.config["plot_fontsize"])
	plt.rc('axes',titlesize=ocbnn.config["plot_fontsize"])
	plt.rc('axes',labelsize=ocbnn.config["plot_fontsize"])
	plt.xlabel('X_3')
	plt.ylabel('Y')
	plt.tight_layout()

	# Constraint
	plt.plot(np.arange(-2.0, -1.0, 0.05), [0.0] * len(np.arange(-2.0, -1.0, 0.05)), 'g-')

	# Ground-truth
	plt.plot(ocbnn.X_test[:,2], ocbnn.Y_test[:,0], 'k.', label="Ground Truth")
	
	# Vanilla BNN
	results = np.apply_along_axis(lambda w: ocbnn.forward(ocbnn.X_test, weights=torch.Tensor(w)).numpy(), 1, ocbnn.all_particles[0][0])
	pred_means = torch.Tensor(np.mean(results, axis=0))
	plt.plot(ocbnn.X_test[:,2], pred_means[:,0], 'r.', label="Vanilla BNN")
	
	# OC-BNN
	results = np.apply_along_axis(lambda w: ocbnn.forward(ocbnn.X_test, weights=torch.Tensor(w)).numpy(), 1, ocbnn.all_particles[1][0])
	pred_means = torch.Tensor(np.mean(results, axis=0))
	plt.plot(ocbnn.X_test[:,2], pred_means[:,0], 'g.', label="OC-BNN")

	plt.legend()
	plt.savefig("multi_vanilla.png", dpi=ocbnn.config["plot_dpi"])


def energy_regression():
	bnn = BNNSVGDRegressor(uid="bnn-multi-example", configfile="configs/bnn-multi-example.json")
	bnn.load(**energyset())
	bnn.add_positive_constraint((-np.inf, np.inf, -np.inf, np.inf, -2.0, -1.0, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf), lambda x: torch.zeros(len(x)).unsqueeze(1))
	bnn.infer()
	bnn.evaluate()
	
	bnn.config["prior_type"] = "oc_positive"
	bnn.infer()
	bnn.evaluate()
	plot_energy(bnn)


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(name)s-%(levelname)s: %(message)s")
	logging.info("Running OC-BNN library...")
	energy_regression()
	logging.info("Completed.")