
"""
Comparing all inference methods for a toy regression and toy classification task.

"""

import numpy as np
import torch
import logging
import matplotlib.pyplot as plt

from bnn.svgd import *
from bnn.sgld import *
from bnn.hmc import *
from bnn.bbb import *
from data.toydata import toy1, toy2, toy3


def bakeoff():
	""" Compare inference algorithms on a simple regression and classification task, without constraints. """

	# Regression
	bnn = BNNHMCRegressor(uid="bnn-rbakeoff", configfile="configs/bnn-rbakeoff.json")
	bnn.load(**toy1())
	bnn.infer()
	bnn.plot_pp(plot_title="HMC", domain=np.arange(-5, 5, 0.05), ylims=(-9, 7))
	bnn = BNNSGLDRegressor(uid="bnn-rbakeoff", configfile="configs/bnn-rbakeoff.json")
	bnn.load(**toy1())
	bnn.infer()
	bnn.plot_pp(plot_title="SGLD", domain=np.arange(-5, 5, 0.05), ylims=(-9, 7))
	bnn = BNNSVGDRegressor(uid="bnn-rbakeoff", configfile="configs/bnn-rbakeoff.json")
	bnn.load(**toy1())
	bnn.infer()
	bnn.plot_pp(plot_title="SVGD", domain=np.arange(-5, 5, 0.05), ylims=(-9, 7))
	bnn = BNNBBBRegressor(uid="bnn-rbakeoff", configfile="configs/bnn-rbakeoff.json")
	bnn.load(**toy1())
	bnn.infer()
	bnn.plot_pp(plot_title="BBB", domain=np.arange(-5, 5, 0.05), ylims=(-9, 7))

	# Classification
	bnn = BNNHMCClassifier(uid="bnn-cbakeoff", configfile="configs/bnn-cbakeoff.json")
	bnn.load(**toy3())
	bnn.infer()
	bnn.plot_pp(plot_title="HMC", xlims=(-5, 5), ylims=(-5, 5))
	bnn = BNNSGLDClassifier(uid="bnn-cbakeoff", configfile="configs/bnn-cbakeoff.json")
	bnn.load(**toy3())
	bnn.infer()
	bnn.plot_pp(plot_title="SGLD", xlims=(-5, 5), ylims=(-5, 5))
	bnn = BNNSVGDClassifier(uid="bnn-cbakeoff", configfile="configs/bnn-cbakeoff.json")
	bnn.load(**toy3())
	bnn.infer()
	bnn.plot_pp(plot_title="SVGD", xlims=(-5, 5), ylims=(-5, 5))
	bnn = BNNBBBClassifier(uid="bnn-cbakeoff", configfile="configs/bnn-cbakeoff.json")
	bnn.load(**toy3())
	bnn.infer()
	bnn.plot_pp(plot_title="BBB", xlims=(-5, 5), ylims=(-5, 5))


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(name)s-%(levelname)s: %(message)s")
	logging.info("Running OC-BNN library...")
	bakeoff()
	logging.info("Completed.")
