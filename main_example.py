
"""
Basic example for:
	- no constraints
	- positive constraints
	- negative constraints.

"""

import numpy as np
import torch
import logging
import matplotlib.pyplot as plt

from bnn.svgd import *
from bnn.sgld import *
from bnn.hmc import *
from bnn.bbb import *
from data.toydata import toy1, toy3


def run_vanilla():
	""" Vanilla BNN. """
	bnn = BNNSVGDRegressor(uid="bnn-vanilla-eg", configfile="configs/bnn-vanilla-eg.json")
	bnn.load(**toy1())
	bnn.infer()
	bnn.plot_pp(plot_title="Predictive Posterior Plot", domain=np.arange(-5, 5, 0.05), ylims=(-9, 7))


def run_negative():
	""" OC-BNN with negative constraints. """
	def cy1(x, y): return y[:,0] - 2.5    # y < 2.5
	def cy2(x, y): return 3 - y[:,0]      # y > 3
	def cx1(x, y): return -0.3 - x[:,0]   # x > -0.3
	def cx2(x, y): return x[:,0] - 0.3    # x < 0.3

	def addons():
		dom = np.arange(-0.3, 0.3, 0.05)
		plt.fill_between(dom, 3.0, plt.ylim()[1], facecolor='#E41A1C', alpha=0.5, zorder=101)
		plt.fill_between(dom, plt.ylim()[0], 2.5, facecolor='#E41A1C', alpha=0.5, zorder=101)

	bnn = BNNSVGDRegressor(uid="bnn-negative-eg", configfile="configs/bnn-negative-eg.json")
	bnn.load(**toy1())
	bnn.add_negative_constraint((-5.0, 5.0), [cy1, cx1, cx2])
	bnn.add_negative_constraint((-5.0, 5.0), [cy2, cx1, cx2])
	bnn.infer()
	bnn.plot_pp(plot_title="Predictive Posterior Plot", domain=np.arange(-5, 5, 0.05), ylims=(-9, 7), addons=addons)


def run_positive():
	""" OC-BNN with positive constraints. """
	bnn = BNNSVGDRegressor(uid="bnn-positive-eg", configfile="configs/bnn-positive-eg.json")
	bnn.load(**toy2())
	bnn.add_positive_constraint((-5.0, -3.0), lambda x: -x + 5)
	bnn.add_positive_constraint((3.0, 5.0), lambda x: x + 5)
	bnn.infer()
	bnn.plot_pp(plot_title="Predictive Posterior Plot", domain=np.arange(-4, 4, 0.05), ylims=(0, 14))


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(name)s-%(levelname)s: %(message)s")
	logging.info("Running OC-BNN library...")
	run_vanilla()
	run_positive()
	run_negative()
	logging.info("Completed.")
