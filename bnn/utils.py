
"""
Utility functions.
"""

import numpy as np
import torch
import logging
import re
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib import rcParams
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


COLORS = {
	'blue': ('b', 0.5, 0.15), 
	'red': ('#E41A1C', 0.6, None), 
	'green': ('#00B200', 0.3, None), 
	'gray': ('k', 0.18, 0.08),
	'ocean': ('#266DD3', None, None),
	'sand': ('#FF8C61', None, None)
}


def plot_1dregressor(bnn, plot_title, domain, ylims, plot_type="interval", with_data=True, 
	plot_fontsize=18, plot_figsize=(8,6), plot_dpi=400, addons=None):
		""" Plot posterior/prior predictive for 1D regression tasks. 
			
			`plot_type`: "full" for individual predictive functions, "interval" for credible intervals
		"""
		# Configurations.
		assert(bnn.Xdim == 1)
		rcParams['font.family'] = 'serif'
		rcParams['font.serif'] = ['Charter']
		pp_fig = plt.figure(figsize=tuple(plot_figsize))
		plt.gca().set_ylim(ylims[0], ylims[1])
		plt.gca().set_xlim(domain[0], domain[-1])
		plt.xticks(fontsize=plot_fontsize)
		plt.yticks(fontsize=plot_fontsize)
		plt.title(plot_title, fontsize=plot_fontsize+4)
		plt.xlabel(r'$\mathcal{X}$', fontsize=plot_fontsize)
		plt.ylabel(r'$\mathcal{Y}$', fontsize=plot_fontsize)
		plt.tight_layout()

		# Plot training data, if it exists.
		if with_data:
			plt.scatter(bnn.X_train.data.numpy(), bnn.Y_train.data.numpy(), facecolors='w', edgecolors='k', zorder=999)

		# For each inference run, plot results.
		for samples, name in bnn.all_bayes_samples:
			if "ocbnn" in name:
				cdict = COLORS['blue']
			else:
				cdict = COLORS['gray']
			results = bnn.predict(samples, domain).squeeze().t()
			if plot_type == "interval":
				means = results.mean(dim=1).squeeze()
				lower2 = np.percentile(results, 2.275, axis=1).squeeze()
				upper2 = np.percentile(results, 97.725, axis=1).squeeze()
				lower1 = np.percentile(results, 15.865, axis=1).squeeze()
				upper1 = np.percentile(results, 84.135, axis=1).squeeze()
				plt.fill_between(domain.squeeze(), lower1, upper1, facecolor=cdict[0], alpha=cdict[1])
				plt.fill_between(domain.squeeze(), lower2, upper2, facecolor=cdict[0], alpha=cdict[2])
				plt.plot(domain.squeeze(), means, color=cdict[0], zorder=99)
			elif plot_type == "full":
				for line in results.T:
					plt.plot(domain, line, color=cdict[0], alpha=cdict[1], linewidth=0.8)

		# Plot additional/custom structures.
		if addons is not None:
			addons()

		# Save and close.
		plot_savename = re.sub(r'[^\w\s]', '', plot_title).lower().replace(' ' , '_')
		plt.savefig(f'history/{bnn.uid}_plot_{plot_savename}.png', dpi=plot_dpi)
		logging.info(f'[{bnn.uid}] {"Posterior" if with_data else "Prior"} predictive plot saved as `history/{bnn.uid}_plot_{plot_savename}.png`.')



def plot_2d2classifier(bnn, plot_title, x1_domain, x2_domain, plot_type="interval", with_data=True, 
	plot_fontsize=18, plot_figsize=(8,6), plot_dpi=400, addons=None):
		""" Plot posterior/prior predictive for 2D classication tasks with 2 output classes (i.e. binary classification). 
			Note that unlike regression, each plot can only show the results of ONE inference. By default, `bnn.all_bayes_samples[-1]` is taken.

			`plot_type`: "full" for individual predictive functions (grayscale shading), "interval" for credible intervals (contour lines)
		"""
		# Configurations.
		assert(bnn.Xdim == 2)
		rcParams['font.family'] = 'serif'
		rcParams['font.serif'] = ['Charter']
		pp_fig = plt.figure(figsize=tuple(plot_figsize))
		plt.gca().set_ylim(x2_domain[0], x2_domain[-1])
		plt.gca().set_xlim(x1_domain[0], x1_domain[-1])
		plt.xticks(fontsize=plot_fontsize)
		plt.yticks(fontsize=plot_fontsize)
		plt.title(plot_title, fontsize=plot_fontsize+4)
		plt.xlabel(r'$\mathcal{X}_1$', fontsize=plot_fontsize)
		plt.ylabel(r'$\mathcal{X}_2$', fontsize=plot_fontsize)
		plt.tight_layout()

		# Plot training data.
		Xtr, Ytr = bnn.X_train.data.numpy(), bnn.Y_train.squeeze().data.numpy()
		plt.plot(Xtr[Ytr == 1][:, 0], Xtr[Ytr == 1][:, 1], color=COLORS['sand'][0],
			marker='o', markeredgecolor='k', markeredgewidth=2.0, linewidth=0)
		plt.plot(Xtr[Ytr == 0][:, 0], Xtr[Ytr == 0][:, 1], color=COLORS['ocean'][0],
			marker='o', markeredgecolor='k', markeredgewidth=2.0, linewidth=0)

		# Plot results of the MOST RECENT set of samples.
		samples, _ = bnn.all_bayes_samples[-1]
		Xte = torch.stack((x1_domain.repeat(len(x2_domain)), x2_domain.repeat_interleave(len(x1_domain)))).t()
		results = bnn.predict(samples, Xte).t()
		results_count = np.apply_along_axis(lambda v: np.bincount(v, minlength=2), 1, results)
		results_count = np.apply_along_axis(lambda v: v / results.shape[1], 1, results_count)
		colors = [(v[1], v[1], v[1]) for v in results_count.tolist()]
		if plot_type == "full":
			# grayscale shading: white --> positive (1) output class
			plt.scatter(Xte[:, 0], Xte[:, 1], marker='o', c=range(len(results)), cmap=ListedColormap(colors))
		elif plot_type == "interval":
			# shading: orange for positive (1) and blue for negative (0)
			levels = [0.75, 0.9, 1.0]
			for idx, c in enumerate(["ocean", "sand"]):
				for jdx, alpha in enumerate([0.5, 0.7]):
					_ = plt.gca().contourf(Xte[:, 0].reshape(len(x1_domain), len(x2_domain)), Xte[:, 1].reshape(len(x1_domain), len(x2_domain)),
						results_count[:,idx].reshape(len(x1_domain), len(x2_domain)), levels=levels[jdx:jdx+2], colors=COLORS[c][0], alpha=alpha)	

		# Plot additional/custom structures.
		if addons is not None:
			addons()

		# Save and close.
		plot_savename = re.sub(r'[^\w\s]', '', plot_title).lower().replace(' ' , '_')
		plt.savefig(f'history/{bnn.uid}_plot_{plot_savename}.png', dpi=plot_dpi)
		logging.info(f'[{bnn.uid}] {"Posterior" if with_data else "Prior"} predictive plot saved as `history/{bnn.uid}_plot_{plot_savename}.png`.')



def plot_2d3classifier(bnn, plot_title, x1_domain, x2_domain, plot_type="interval", with_data=True, 
	plot_fontsize=18, plot_figsize=(8,6), plot_dpi=400, addons=None):
		""" Plot posterior/prior predictive for 2D classication tasks with 3 output classes. 
			Note that unlike regression, each plot can only show the results of ONE inference. By default, `bnn.all_bayes_samples[-1]` is taken.

			`plot_type`: "full" for individual predictive functions (3-simplex RGB shading), "interval" for credible intervals (contour lines)
		"""
		# Configurations.
		assert(bnn.Xdim == 2)
		rcParams['font.family'] = 'serif'
		rcParams['font.serif'] = ['Charter']
		pp_fig = plt.figure(figsize=tuple(plot_figsize))
		plt.gca().set_ylim(x2_domain[0], x2_domain[-1])
		plt.gca().set_xlim(x1_domain[0], x1_domain[-1])
		plt.xticks(fontsize=plot_fontsize)
		plt.yticks(fontsize=plot_fontsize)
		plt.title(plot_title, fontsize=plot_fontsize+4)
		plt.xlabel(r'$\mathcal{X}_1$', fontsize=plot_fontsize)
		plt.ylabel(r'$\mathcal{X}_2$', fontsize=plot_fontsize)
		plt.tight_layout()

		# Plot training data.
		Xtr, Ytr = bnn.X_train.data.numpy(), bnn.Y_train.squeeze().data.numpy()
		for oc in range(bnn.Ydim):
			plt.plot(Xtr[Ytr == oc][:, 0], Xtr[Ytr == oc][:, 1], color=np.bincount([oc], minlength=3),
				marker='o', markeredgecolor='k', markeredgewidth=2.0, linewidth=0)

		# Plot results of the MOST RECENT set of samples.
		samples, _ = bnn.all_bayes_samples[-1]
		Xte = torch.stack((x1_domain.repeat(len(x2_domain)), x2_domain.repeat_interleave(len(x1_domain)))).t()
		results = bnn.predict(samples, Xte).t()
		results_count = np.apply_along_axis(lambda v: np.bincount(v, minlength=3), 1, results)
		results_count = np.apply_along_axis(lambda v: v / results.shape[1], 1, results_count)
		colors = [tuple(v) for v in results_count.tolist()]
		if plot_type == "full":
			plt.scatter(Xte[:, 0], Xte[:, 1], marker='o', c=range(len(results)), cmap=ListedColormap(colors))
		elif plot_type == "interval":
			levels = [0.75, 0.9, 1.0]
			for idx, c in enumerate(["red", "green", "blue"]):
				for jdx, alpha in enumerate([0.5, 0.7]):
					_ = plt.gca().contourf(Xte[:, 0].reshape(len(x1_domain), len(x2_domain)), Xte[:, 1].reshape(len(x1_domain), len(x2_domain)),
						results_count[:,idx].reshape(len(x1_domain), len(x2_domain)), levels=levels[jdx:jdx+2], colors=COLORS[c][0], alpha=alpha)			

		# Plot additional/custom structures.
		if addons is not None:
			addons()

		# Save and close.
		plot_savename = re.sub(r'[^\w\s]', '', plot_title).lower().replace(' ' , '_')
		plt.savefig(f'history/{bnn.uid}_plot_{plot_savename}.png', dpi=plot_dpi)
		logging.info(f'[{bnn.uid}] {"Posterior" if with_data else "Prior"} predictive plot saved as `history/{bnn.uid}_plot_{plot_savename}.png`.')



def generic_plot(plot_title, xlims, ylims, xlabel, ylabel, addons, plot_fontsize=18, plot_figsize=(8,6), plot_dpi=400):
	""" Wrapper function for plot formatting. """

	# Configurations.
	rcParams['font.family'] = 'serif'
	rcParams['font.serif'] = ['Charter']
	pp_fig = plt.figure(figsize=tuple(plot_figsize))
	plt.gca().set_ylim(ylims[0], ylims[-1])
	plt.gca().set_xlim(xlims[0], xlims[-1])
	plt.xticks(fontsize=plot_fontsize)
	plt.yticks(fontsize=plot_fontsize)
	plt.title(plot_title, fontsize=plot_fontsize+4)
	plt.xlabel(xlabel, fontsize=plot_fontsize)
	plt.ylabel(ylabel, fontsize=plot_fontsize)
	plt.tight_layout()

	addons()

	plot_savename = re.sub(r'[^\w\s]', '', plot_title).lower().replace(' ' , '_')
	plt.savefig(f'history/plot_{plot_savename}.png', dpi=plot_dpi)
	logging.info(f'Plot saved as `history/plot_{plot_savename}.png`.')



def eval_accuracy_and_f1_score(bnn, is_binary=False, X_eval=None, Y_eval=None):
	""" Evaluate accuracy and F1 score on the test set.
		Binary classification only, i.e. assume `bnn.predict()` returns a Boolean tensor. 
		Uses the most recent posterior sample: `bnn.all_bayes_samples[-1]`. 
	"""
	if X_eval is None:
		X_eval, Y_eval = bnn.X_test, bnn.Y_test

	samples, _ = bnn.all_bayes_samples[-1]
	if is_binary:
		preds = bnn.predict(samples, X_eval, return_probs=True).mean(dim=0) >= 0.5
	else:
		preds = bnn.predict(samples, X_eval, return_probs=True).mean(dim=0).argmax(dim=1)
	acc = accuracy_score(Y_eval, preds)
	_, _, f1, _ = precision_recall_fscore_support(Y_eval, preds, average='binary')
	return acc, f1
