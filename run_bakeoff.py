
"""
Visualize differences in predictive distributions across inference methods, on a toy regression and a classification task.
No output constraints.

Default behavior when running this file:
	All experiments are fully run from scratch, i.e. full inference is performed.
	The config files from the `repro/` folder are used.
	As BNN inference is approximate, results will slightly deviate from those in the paper, especially for VI methods.

There are three optional command-line arguments:
	--debug :: run experiments in debug mode (only a couple iterations of inference)
	--pretrained :: use our pre-trained posterior samples
	--custom_config <filename.yaml> :: use your own config file --> <filename.yaml> if specified, else, `config.yaml` in the root directory

"""

import numpy as np
import torch
import logging
import argparse
import matplotlib.pyplot as plt

from bnn import *
from bnn.utils import plot_1dregressor, plot_2d3classifier
from data.dataloader import toy1, toy2


def bakeoff():

	# Regression
	# To allow for better comparison, predictive distribution is plotted as individual functions instead of credible intervals.
	domain = torch.arange(-5, 5, 0.05).unsqueeze(dim=1)
	ylims = (-9, 7)

	bnn = BNNHMCRegressor(uid="rbakeoff", configfile=args.custom_config or "repro/rbakeoff.yaml")
	bnn.load(**toy1())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/rbakeoff_hmc1.pt', 'bakeoff')
	else:
		bnn.infer()
	plot_1dregressor(bnn, plot_title="HMC Posterior Predictive", domain=domain, ylims=ylims, plot_type="full")

	bnn = BNNBBBRegressor(uid="rbakeoff", configfile=args.custom_config or "repro/rbakeoff.yaml")
	bnn.load(**toy1())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/rbakeoff_bbb1.pt', 'bakeoff')
	else:
		bnn.infer()
	plot_1dregressor(bnn, plot_title="BBB Posterior Predictive", domain=domain, ylims=ylims, plot_type="full")

	bnn = BNNSGLDRegressor(uid="rbakeoff", configfile=args.custom_config or "repro/rbakeoff.yaml")
	bnn.load(**toy1())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/rbakeoff_sgld1.pt', 'bakeoff')
	else:
		bnn.infer()
	plot_1dregressor(bnn, plot_title="SGLD Posterior Predictive", domain=domain, ylims=ylims, plot_type="full")

	bnn = BNNSVGDRegressor(uid="rbakeoff", configfile=args.custom_config or "repro/rbakeoff.yaml")
	bnn.update_config(infer_nsamples=100) # SVGD has O(n^3) complexity in the number of particles. If too slow, use 50 or 75 particles.
	bnn.load(**toy1())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/rbakeoff_svgd1.pt', 'bakeoff')
	else:
		bnn.infer()
	plot_1dregressor(bnn, plot_title="SVGD Posterior Predictive", domain=domain, ylims=ylims, plot_type="full")


	# Classification (3 output classes)
	# To allow for better comparison, predictive distribution is plotted as individual functions instead of credible intervals.
	# (In the 2D classification case, this means the entire plot is shaded with a RGB simplex.) 
	x1_domain = x2_domain = torch.arange(-5, 5, 0.1)

	bnn = BNNHMCClassifier(uid="cbakeoff", configfile=args.custom_config or "repro/cbakeoff.yaml")
	bnn.load(**toy2())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/cbakeoff_hmc1.pt', 'bakeoff')
	else:
		bnn.infer()
	plot_2d3classifier(bnn, plot_title="HMC Posterior Predictive", x1_domain=x1_domain, x2_domain=x2_domain, plot_type="full")

	bnn = BNNBBBClassifier(uid="cbakeoff", configfile=args.custom_config or "repro/cbakeoff.yaml")
	bnn.load(**toy2())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/cbakeoff_bbb1.pt', 'bakeoff')
	else:
		bnn.infer()
	plot_2d3classifier(bnn, plot_title="BBB Posterior Predictive", x1_domain=x1_domain, x2_domain=x2_domain, plot_type="full")

	bnn = BNNSGLDClassifier(uid="cbakeoff", configfile=args.custom_config or "repro/cbakeoff.yaml")
	bnn.load(**toy2())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/cbakeoff_sgld1.pt', 'bakeoff')
	else:
		bnn.infer()
	plot_2d3classifier(bnn, plot_title="SGLD Posterior Predictive", x1_domain=x1_domain, x2_domain=x2_domain, plot_type="full")

	bnn = BNNSVGDClassifier(uid="cbakeoff", configfile=args.custom_config or "repro/cbakeoff.yaml")
	bnn.update_config(infer_nsamples=100)
	bnn.load(**toy2())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/cbakeoff_svgd1.pt', 'bakeoff')
	else:
		bnn.infer()
	plot_2d3classifier(bnn, plot_title="SVGD Posterior Predictive", x1_domain=x1_domain, x2_domain=x2_domain, plot_type="full")
	


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(name)s-%(levelname)s: %(message)s")
	logging.info("Running OC-BNN library...")
	parser = argparse.ArgumentParser(description='Process command-line arguments.')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--pretrained', action='store_true')
	parser.add_argument('--custom_config', nargs='?', const='config.yaml')
	args = parser.parse_args()
	bakeoff()
	logging.info("Completed.")
