
"""
All experiments in Section 5 and Appendix D of our paper: https://arxiv.org/pdf/2010.10969.pdf.

Default behavior when running this file:
	All experiments are fully run from scratch, i.e. full inference is performed.
	The config files from the `repro/` folder are used.
	As BNN inference is approximate, results will slightly deviate from those in the paper, especially for VI methods.

There are three optional command-line arguments:
	--debug :: run experiments in debug mode (only a couple iterations of inference)
	--pretrained :: use our pre-trained posterior samples and AOCP parameters
	--custom_config <filename.yaml> :: use your own config file --> <filename.yaml> if specified, else, `config.yaml` in the root directory

Note: The pretrained samples in `repro/` are not identical to the samples used to produce the actual plots in our paper. 
They are separate runs of the same experiments to prove reproducibility of our results.

"""

import numpy as np
import torch
import logging
import argparse
import matplotlib.pyplot as plt

from bnn import *
from bnn.utils import *
from data.dataloader import *


def figure1and5():
	""" 
	Section 5, Figure 1a and 1b.
	Appendix D, Figure 5a and 5b. 
	"""
	domain = torch.arange(-5, 5, 0.05).unsqueeze(dim=1)
	def ifunc(X): return [[(2.5, 3.0)] for _ in range(len(X))]

	# Plot constraints.
	def addons():
		dom = np.arange(-0.3, 0.3, 0.05)
		plt.fill_between(dom, 3.0, plt.ylim()[1], facecolor=COLORS['red'][0], alpha=0.5, zorder=101)
		plt.fill_between(dom, plt.ylim()[0], 2.5, facecolor=COLORS['red'][0], alpha=0.5, zorder=101)

	# Figure 1a: prior predictive
	bnn = BNNHMCRegressor(uid="F1a", configfile=args.custom_config or "repro/F1a.yaml")
	bnn.load(**toy1())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/F1a_hmc1.pt', 'hmc_baseline')
	else:
		bnn.infer(with_data=False)
	bnn.add_deterministic_constraint(constrained_domain=(-0.3, 0.3), interval_func=ifunc, prior_type="negative_exponential_cocp")
	bnn.update_config(use_ocbnn=True)
	if args.pretrained:
		bnn.load_bayes_samples('repro/F1a_hmc2.pt', 'hmc_ocbnn')
	else:
		bnn.infer(with_data=False)
	plot_1dregressor(bnn, plot_title="Prior Predictive", domain=domain, ylims=(-9, 7), addons=addons, with_data=False)

	# Figure 1b: posterior predictive
	bnn.clear_all_samples()
	bnn.update_config(uid="F1b", use_ocbnn=False)
	if args.pretrained:
		bnn.load_bayes_samples('repro/F1b_hmc1.pt', 'hmc_baseline')
	else:
		bnn.infer()
	bnn.update_config(use_ocbnn=True)
	if args.pretrained:
		bnn.load_bayes_samples('repro/F1b_hmc2.pt', 'hmc_ocbnn')
	else:
		bnn.infer()
	plot_1dregressor(bnn, plot_title="Posterior Predictive", domain=domain, ylims=(-9, 7), addons=addons)

	# Figure 5a: posterior predictive using SVGD
	bnn = BNNSVGDRegressor(uid="F5a", configfile=args.custom_config or "repro/F5a.yaml")
	bnn.load(**toy1())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/F5a_svgd1.pt', 'hmc_baseline')
	else:
		bnn.infer()
	bnn.add_deterministic_constraint(constrained_domain=(-0.3, 0.3), interval_func=ifunc, prior_type="negative_exponential_cocp")
	bnn.update_config(use_ocbnn=True)
	if args.pretrained:
		bnn.load_bayes_samples('repro/F5a_svgd2.pt', 'hmc_ocbnn')
	else:
		bnn.infer()
	plot_1dregressor(bnn, plot_title="Posterior Predictive", domain=domain, ylims=(-9, 7), plot_type="full", addons=addons)
	
	# Figure 5b: posterior predictive using BBB
	bnn = BNNBBBRegressor(uid="F5b", configfile=args.custom_config or "repro/F5b.yaml")
	bnn.load(**toy1())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/F5b_bbb1.pt', 'hmc_baseline')
	else:
		bnn.infer()
	bnn.add_deterministic_constraint(constrained_domain=(-0.3, 0.3), interval_func=ifunc, prior_type="negative_exponential_cocp")
	bnn.update_config(use_ocbnn=True)
	if args.pretrained:
		bnn.load_bayes_samples('repro/F5b_bbb2.pt', 'hmc_ocbnn')
	else:
		bnn.infer()
	plot_1dregressor(bnn, plot_title="Posterior Predictive", domain=domain, ylims=(-9, 7), addons=addons)


def figure2():
	"""
	Section 5, Figure 2a and 2b (insets are baselines).
	"""
	domain = torch.arange(-5, 5, 0.1)

	# Plot constraints.
	def addons():
		plt.gca().add_patch(plt.Rectangle((1.0, -2.0), 2.0, 2.0, facecolor=COLORS['green'][0], linewidth=2.0, edgecolor='k', alpha=0.5))

	# Figure 2a: prior predictive
	bnn = BNNHMCClassifier(uid='F2a', configfile=args.custom_config or "repro/F2a.yaml")
	bnn.load(**toy2())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/F2a_hmc1.pt', 'hmc_baseline')
	else:
		bnn.infer(with_data=False)
	plot_2d3classifier(bnn, plot_title="Prior Predictive (Baseline)", x1_domain=domain, x2_domain=domain, addons=addons, with_data=False)
	bnn.add_deterministic_constraint(constrained_domain=(1.0, 3.0, -2.0, 0.0), forbidden_classes=[0, 2], prior_type="positive_dirichlet_cocp")
	bnn.update_config(use_ocbnn=True)
	if args.pretrained:
		bnn.load_bayes_samples('repro/F2a_hmc2.pt', 'hmc_ocbnn')
	else:
		bnn.infer(with_data=False)
	plot_2d3classifier(bnn, plot_title="Prior Predictive", x1_domain=domain, x2_domain=domain, addons=addons, with_data=False)

	# Figure 2b: posterior predictive
	bnn.clear_all_samples()
	bnn.update_config(uid="F2b", use_ocbnn=False)
	if args.pretrained:
		bnn.load_bayes_samples('repro/F2b_hmc1.pt', 'hmc_baseline')
	else:
		bnn.infer()
	plot_2d3classifier(bnn, plot_title="Posterior Predictive (Baseline)", x1_domain=domain, x2_domain=domain, addons=addons)
	bnn.update_config(use_ocbnn=True)
	if args.pretrained:
		bnn.load_bayes_samples('repro/F2b_hmc2.pt', 'hmc_ocbnn')
	else:
		bnn.infer()
	plot_2d3classifier(bnn, plot_title="Posterior Predictive", x1_domain=domain, x2_domain=domain, addons=addons)


def figure3a():
	"""
	Section 5, Figure 3a.
	"""
	domain = torch.arange(-8, 8, 0.05).unsqueeze(dim=1)
	def ifunc(X): return [[(-np.inf, 0) if x.item() < 0 else (0, np.inf)] for x in X]

	def addons():
		plt.fill_between(np.arange(-10, 0, 0.05), -3.0, 0.0, facecolor=COLORS['green'][0], alpha=COLORS['green'][1], zorder=101)
		plt.fill_between(np.arange(0, 10, 0.05), 0.0, 3.0, facecolor=COLORS['green'][0], alpha=COLORS['green'][1], zorder=101)	
	
	bnn = BNNHMCRegressor(uid="F3a", configfile=args.custom_config or "repro/F3a.yaml")
	bnn.load(**toy3())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/F3a_hmc1.pt', 'hmc_baseline')
	else:
		bnn.infer()
	bnn.add_deterministic_constraint(constrained_domain=(-3, 3), interval_func=ifunc, prior_type="gaussian_aocp")
	if args.pretrained:
		bnn.load_gaussian_aocp_parameters('repro/F3a_aocp.pt')
	else:
		bnn.learn_gaussian_aocp()
	bnn.update_config(use_ocbnn=True)
	if args.pretrained:
		bnn.load_bayes_samples('repro/F3a_hmc2.pt', 'hmc_ocbnn')
	else:
		bnn.infer()
	plot_1dregressor(bnn, plot_title="Posterior Predictive", domain=domain, ylims=(-3, 3), addons=addons)


def figure3bc():
	"""
	Section 5, Figure 3b and c.
	"""
	domain = torch.arange(-5, 5, 0.05).unsqueeze(dim=1)
	def ifunc(X): return [[(-np.inf, 1.0), (2.5, np.inf)] for _ in range(len(X))]

	# Plot constraints.
	def addons():
		plt.fill_between(np.arange(-1.0, 1.0, 0.05), 1.0, 2.5, facecolor=COLORS['red'][0], alpha=0.5, zorder=101)

	# Figure 3b: posterior predictive
	bnn = BNNSVGDRegressor(uid='F3b', configfile=args.custom_config or "repro/F3b.yaml")
	bnn.load(**toy4())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/F3b_svgd1.pt', 'svgd_baseline')
	else:
		bnn.infer()
	bnn.add_deterministic_constraint(constrained_domain=(-1.0, 1.0), interval_func=ifunc, prior_type="negative_exponential_cocp")
	bnn.update_config(use_ocbnn=True)
	if args.pretrained:
		bnn.load_bayes_samples('repro/F3b_svgd2.pt', 'svgd_ocbnn')
	else:
		bnn.infer()
	plot_1dregressor(bnn, plot_title="Posterior Predictive", domain=domain, ylims=(-3, 6), plot_type="full", addons=addons)

	# Figure 3c: rejection sampling efficiency
	# This plot will be time-consuming because we use 100 SVGD particles for inference.
	svgd_ocbnn = []
	rs_nsamples = [1, 5, 10, 100, 500, 1000]
	rs_domain = torch.arange(-1.0, 1.0, 0.05).unsqueeze(dim=1)

	bnn = BNNSVGDRegressor(uid=f'F3c-base', configfile=args.custom_config or "repro/F3c.yaml")
	bnn.load(**toy4())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/F3b_svgd1.pt', 'svgd_baseline')
	else:
		bnn.infer()
	results = bnn.predict(bnn.all_bayes_samples[0][0], rs_domain).squeeze().t()
	violation_count = 0
	for line in results.T:
		violation_count += any([(point > 1.0 and point < 2.5) for point in line])
	svgd_baseline = violation_count / len(results.T)
	
	for c in rs_nsamples:
		bnn = BNNSVGDRegressor(uid=f'F3c-{c}', configfile=args.custom_config or "repro/F3c.yaml")
		bnn.load(**toy4())
		if args.debug:
			bnn.debug_mode()
		bnn.update_config(ocp_nsamples=c, use_ocbnn=True)
		bnn.add_deterministic_constraint(constrained_domain=(-1.0, 1.0), interval_func=ifunc, prior_type="negative_exponential_cocp")
		if args.pretrained:
			bnn.load_bayes_samples(f'repro/F3c-{c}_svgd1.pt', f'svgd_ocbnn_{c}')
		else:
			bnn.infer()
		results = bnn.predict(bnn.all_bayes_samples[0][0], rs_domain).squeeze().t()
		violation_count = 0
		for line in results.T:
			violation_count += any([(point > 1.0 and point < 2.5) for point in line])
		svgd_ocbnn.append(violation_count / len(results.T))

	def rs_addons():
		plt.plot([0, 7], [svgd_baseline, svgd_baseline], 'k--')
		plt.plot(np.log(rs_nsamples), svgd_ocbnn, 'bP--')

	generic_plot(plot_title="Fraction of Rejected Posterior Samples", xlims=(-0.5, 7.5), ylims=(0.0, 1.1),
		xlabel=r'log(samples) from $\mathcal{C}_\mathbf{x}$', ylabel='Fraction', addons=rs_addons)


def figure4():
	"""
	Section 5, Figure 4.
	"""
	domain = torch.arange(-0.5, 1.5, 0.01)
	def pfunc(X): return X[:,1]

	bnn = BNNHMCBinaryClassifier(uid="F4", configfile=args.custom_config or "repro/F4.yaml")
	bnn.load(**toy5())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/F4_hmc1.pt', 'hmc_baseline')
	else:
		bnn.infer()
	plot_2d2classifier(bnn, plot_title="Posterior Predictive (Baseline)", x1_domain=domain, x2_domain=domain, addons=None)
	bnn.add_probabilistic_constraint(constrained_domain=(-0.5, 1.5, 0.0, 1.0), prob_func=pfunc, prior_type="gaussian_aocp")
	if args.pretrained:
		bnn.load_gaussian_aocp_parameters('repro/F4_aocp.pt')
	else:
		bnn.learn_gaussian_aocp()
	bnn.update_config(use_ocbnn=True)
	if args.pretrained:
		bnn.load_bayes_samples('repro/F4_hmc2.pt', 'hmc_ocbnn')
	else:
		bnn.infer()
	plot_2d2classifier(bnn, plot_title="Posterior Predictive", x1_domain=domain, x2_domain=domain, addons=None)


def figure6():
	"""
	Appendix D, Figure 6.
	"""
	domain = torch.arange(-20, 20, 0.05).unsqueeze(dim=1)
	queries = [3.89, -2.01, 15.03]
	def ifunc(X): return [[(5 * np.cos(x.item() / 1.7), 5 * np.cos(x.item() / 1.7))] for x in X]
	def ground_truth(x): return 5 * np.cos(x / 1.7)	

	# Plot constraints and ground-truth function.
	def addons():
		plt.plot(domain.squeeze(), ground_truth(domain).squeeze(), 'g--', linewidth=2.0, zorder=199)
		for s in queries:
			dom = torch.arange(s-0.5, s+0.5, 0.05)
			dyl = torch.ones(len(dom)) * -10.0
			dyu = torch.ones(len(dom)) * 8.0
			plt.fill_between(dom, dyl, dyu, facecolor=COLORS['green'][0], alpha=COLORS['green'][1])

	bnn = BNNHMCRegressor(uid="F6", configfile=args.custom_config or "repro/F6.yaml")
	bnn.load(**toy6())
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/F6_hmc1.pt', 'hmc_baseline')
	else:
		bnn.infer()
	for s in queries:
		bnn.add_deterministic_constraint(constrained_domain=(s-0.5, s+0.5), interval_func=ifunc, prior_type="positive_gaussian_cocp")
	bnn.update_config(use_ocbnn=True)
	if args.pretrained:
		bnn.load_bayes_samples('repro/F6_hmc2.pt', 'hmc_ocbnn')
	else:
		bnn.infer()	
	plot_1dregressor(bnn, plot_title="Posterior Predictive", domain=domain, ylims=(-9, 7), addons=addons)



if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(name)s-%(levelname)s: %(message)s")
	logging.info("Running OC-BNN library...")
	parser = argparse.ArgumentParser(description='Process command-line arguments.')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--pretrained', action='store_true')
	parser.add_argument('--custom_config', nargs='?', const='config.yaml')
	args = parser.parse_args()
	figure1and5()
	figure2()
	figure3a()
	figure3bc()
	figure4()
	figure6()
	logging.info("Completed.")
