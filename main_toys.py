
"""
All toy examples in the OC-BNN paper: https://arxiv.org/pdf/1905.06287.pdf.

"""

import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from bnn.svgd import *
from bnn.hmc import *
from data.toydata import *


def example1():
	""" 
	Figure 1 (left) in the paper. 
		
	Constraints:
		Lower bounded by y=2.5 and upper bounded by y=3.0 for the domain (-0.3, 0.3).

	Negative constraint prior used.
	"""
	def cy1(x, y): return y[:,0] - 2.5    # y < 2.5
	def cy2(x, y): return 3 - y[:,0]      # y > 3
	def cx1(x, y): return -0.3 - x[:,0]   # x > -0.3
	def cx2(x, y): return x[:,0] - 0.3    # x < 0.3

	def addons():
		dom = np.arange(-0.3, 0.3, 0.05)
		plt.fill_between(dom, 3.0, plt.ylim()[1], facecolor='#E41A1C', alpha=0.5, zorder=101)
		plt.fill_between(dom, plt.ylim()[0], 2.5, facecolor='#E41A1C', alpha=0.5, zorder=101)

	bnn = BNNHMCRegressor(uid='EX1', configfile="configs/EX1.json")
	bnn.load(**toy1())
	bnn.add_negative_constraint((-5.0, 5.0), [cy1, cx1, cx2])
	bnn.add_negative_constraint((-5.0, 5.0), [cy2, cx1, cx2])
	bnn.infer()
	bnn.config["prior_type"] = "gaussian"
	bnn.infer()
	bnn.all_samples = bnn.all_samples[::-1]
	bnn.plot_pp(plot_title="Example 1 (Negative Constraint)", domain=np.arange(-5, 5, 0.05), ylims=(-9, 7), addons=addons)


def example2():
	"""
	Figure 1 (right) in the paper.

	Constraints:
		For x_0 between (1.0, 3.0), x_1 between (-2.0, 0.0), points cannot be in class 1 or 2.

	Positive (Dirichlet) constraint prior used.

	**Note: We use two separate BNN objects for this example because for 2D classification, it is not possible 
		to plot the OC-BNN and baseline results in the same image.
	"""
	bnn = BNNHMCClassifier(uid='EX2', configfile="configs/EX2.json")
	bnn.load(**toy3())
	bnn.add_positive_constraint((1.0, 3.0, -2.0, 0.0), [0, 2])
	bnn.infer()
	bnn.plot_pp(plot_title="Example 2 (Positive Constraint)", xlims=(-5, 5), ylims=(-5, 5))
	
	bnn = BNNHMCClassifier(uid='EX2baseline', configfile="configs/EX2baseline.json")
	bnn.load(**toy3())
	bnn.infer()
	bnn.plot_pp(plot_title="Example 2 (Baseline)", xlims=(-5, 5), ylims=(-5, 5))


def example3():
	"""
	Figure 2 (left) in the paper.

	Constraints:
		Points lie on y = -x + 5 for the domain (-5.0, -3.0).
		Points lie on y = x + 5 for the domain (3.0, 5.0).

	Positive (Gaussian) constraint prior used.
	"""
	bnn = BNNHMCRegressor(uid='EX3', configfile="configs/EX3.json")
	bnn.load(**toy2())
	bnn.add_positive_constraint((-5.0, -3.0), lambda x: -x + 5)
	bnn.add_positive_constraint((3.0, 5.0), lambda x: x + 5)
	bnn.infer()
	bnn.config["prior_type"] = "gaussian"
	bnn.infer()
	bnn.all_samples = bnn.all_samples[::-1]
	bnn.plot_pp(plot_title="Example 3 (Positive Constraint)", domain=np.arange(-4, 4, 0.05), ylims=(0, 14))


def example4():
	"""
	Figure 2 (right) in the paper.

	Constraints:
		Forbidden region is {-1.0 < x < 1.0, -5.0 < y < 3.0}.

	Negative constraint prior used.
	"""
	def cx0(x, y): return x[:,0] - 1.
	def cx1(x, y): return - x[:,0] - 1
	def cy0(x, y): return y[:,0] + 3.
	def cy1(x, y): return - y[:,0] - 5.

	def addons():
		plt.gca().add_patch(Rectangle((-1.0, -5.0), 2, 2, alpha=0.5, color='#E41A1C', linewidth=0, zorder=101))

	bnn = BNNSVGDRegressor(uid='EX4', configfile="configs/EX4.json")
	bnn.load(**toy4())
	bnn.add_negative_constraint((-1.0, 1.0), [cx0, cx1, cy0, cy1])
	bnn.infer()
	bnn.config["prior_type"] = "gaussian"
	bnn.infer()
	bnn.all_particles = bnn.all_particles[::-1]
	bnn.plot_pp(plot_title="Example 4 (Negative Constraint)", domain=np.arange(-5, 5, 0.05), ylims=(-28, 10), addons=addons)


def example5():
	"""
	Figure 4 (left) in the paper.

	Constraints:
		Lower bounded by y=-x+2 and upper bounded by y=-x+7 for the domain (-5.0, -3.0).
		Lower bounded by y=x+2 and upper bounded by y=x+7 for the domain (3.0, 5.0).

	Negative constraint prior used.
	"""
	def cyl1(x, y): return y[:,0] + x[:,0] - 2   # y < -x + 2
	def cyu1(x, y): return -y[:,0] - x[:,0] + 7  # y > -x + 7
	def cyl2(x, y): return y[:,0] - x[:,0] - 2   # y < x + 2
	def cyu2(x, y): return -y[:,0] + x[:,0] + 7  # y > x + 7
	def cxl1(x, y): return x[:,0] + 3  # x < -3
	def cxu1(x, y): return -x[:,0] - 5    # x > -5
	def cxl2(x, y): return 3 - x[:,0]   # x > 3
	def cxu2(x, y): return x[:,0] - 5    # x < 5

	def addons():
		udom, ldom = np.arange(3.0, 5.0, 0.05), np.arange(-5.0, -3.0, 0.05)
		plt.fill_between(ldom, np.vectorize(lambda x: -x+7)(ldom), plt.ylim()[1], facecolor='#E41A1C', alpha=0.5, zorder=101)
		plt.fill_between(ldom, plt.ylim()[0], np.vectorize(lambda x: -x+2)(ldom), facecolor='#E41A1C', alpha=0.5, zorder=101)
		plt.fill_between(udom, np.vectorize(lambda x: x+7)(udom), plt.ylim()[1], facecolor='#E41A1C', alpha=0.5, zorder=101)
		plt.fill_between(udom, plt.ylim()[0], np.vectorize(lambda x: x+2)(udom), facecolor='#E41A1C', alpha=0.5, zorder=101)

	bnn = BNNHMCRegressor(uid='EX5', configfile="configs/EX5.json")
	bnn.load(**toy2())
	bnn.add_negative_constraint((-5.0, 5.0), [cyu1, cxl1, cxu1])
	bnn.add_negative_constraint((-5.0, 5.0), [cyl1, cxl1, cxu1])
	bnn.add_negative_constraint((-5.0, 5.0), [cyu2, cxl2, cxu2])
	bnn.add_negative_constraint((-5.0, 5.0), [cyl2, cxl2, cxu2])
	bnn.infer()
	bnn.config["prior_type"] = "gaussian"
	bnn.infer()
	bnn.all_samples = bnn.all_samples[::-1]
	bnn.plot_pp(plot_title="Example 5 (Negative Constraint)", domain=np.arange(-4, 4, 0.05), ylims=(0, 14), addons=addons)


def example6():
	"""
	Figure 4 (right) in the paper.

	Constraints:
		Points lie on y = -0.2x^3 + 0.5x^2 + 0.7x - 0.5 for the domain (-1.0, 1.0).
		Points lie on y = 0.2x^3 - 0.15x^2 + 3.5 for the domain (-1.0, 1.0).

	Positive (bimodal Gaussian) constraint prior used.
	"""
	bnn = BNNSVGDRegressor(uid='EX6', configfile="configs/EX6.json")
	bnn.load(**toy5())
	bnn.add_positive_constraint((-1.0, 1.0), lambda x: -0.2*(x**3) + 0.5*(x**2) + 0.7*x - 0.5)
	bnn.add_positive_constraint((-1.0, 1.0), lambda x: 0.2*(x**3) - 0.15*(x**2) + 3.5)
	bnn.infer()
	bnn.config["prior_type"] = "gaussian"
	bnn.infer()
	bnn.all_particles = bnn.all_particles[::-1]
	bnn.plot_pp(plot_title="Example 6 (Positive Constraint)", domain=np.arange(-5, 5, 0.05), ylims=(-3, 6))


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(name)s-%(levelname)s: %(message)s")
	logging.info("Running OC-BNN library...")
	example1()
	example2()
	example3()
	example4()
	example5()
	example6()
	logging.info("Completed.")
