
"""
Toy datasets.
"""

import numpy as np
import torch

from torch.distributions.normal import Normal


def toy1():
	""" 1D regression. Codename: GAP """
	def f(x):
		return (-1 * (x ** 4)) + (3 * (x ** 2)) + 1

	X_train = torch.tensor([-2, -1.8, -1, 1, 1.8, 2]).unsqueeze(1)
	Y_train = f(X_train)
	return {"dataset_name": "toy-gap", "X_train": X_train, "Y_train": Y_train}


def toy2():
	""" 1D regression. Codename: TUNNEL """
	def f(x):
		return 0.15 * (x ** 3) - 0.3 * (x **2) + 17.5 + (0.03 / (x - 0.1) ** 2)

	X_train = torch.Tensor([-1.5, 0., 1.0]).unsqueeze(dim=1)
	Y_train = torch.Tensor([6, 10, 6.]).unsqueeze(dim=1)
	return {"dataset_name": "toy-tunnel", "X_train": X_train, "Y_train": Y_train}


def toy3():
	""" 2D classification. Codename: RGB """
	class_means = [[0, -3], [-3., 1.], [2, 3]]
	sigma = 0.3
	size = 8

	X_train = torch.tensor([[-0.5568, -2.8603],
							[-2.8323,  0.9413],
							[ 2.0402,  2.6044],
							[-0.3121, -3.2628],
							[-3.2933,  0.8421],
							[ 2.2045,  2.6692],
							[-0.0165, -3.1644],
							[-2.9325,  0.8448],
							[ 2.3423,  2.8354],
							[-0.2595, -2.9736],
							[-2.9650,  0.7760],
							[ 2.2982,  2.6264],
							[-0.0605, -2.8752],
							[-2.4977,  1.2420],
							[ 1.8175,  3.1518],
							[-0.2743, -3.0461],
							[-2.7137,  1.7121],
							[ 1.9846,  3.5413],
							[ 0.3204, -2.8159],
							[-2.6090,  1.1718],
							[ 2.0794,  3.1475],
							[ 0.1280, -2.9005],
							[-2.9434,  1.1424],
							[ 1.9929,  2.9455]])
	Y_train = torch.tensor(range(3)).repeat(size)
	return {"dataset_name": "toy-rgb", "X_train": X_train, "Y_train": Y_train}


def toy4():
	""" 1D regression. Codename: BOX """
	
	def f(x):
		return (-1 * (x ** 4)) + (3 * (x ** 2)) + 1

	X_train = torch.tensor([-2.5, -2.3, -2, -1.8, -1.5, 1.3, 1.6, 2, 2.1, 2.2, 2.3, 2.31]).unsqueeze(dim=1)
	Y_train = f(X_train)
	X_test = torch.tensor([-2.4, -1.6, -1.0, -0.3, 0.5, 1.0, 1.4, 1.9]).unsqueeze(dim=1)
	Y_test = f(X_test)
	
	return {"dataset_name": "toy-box", "X_train": X_train, "Y_train": Y_train, "X_test": X_test, "Y_test": Y_test}


def toy5():
	""" 1D regression. Codename: EYE """
	
	X_train = torch.tensor([-3.5, -2, 2, 3.5]).unsqueeze(dim=1)
	Y_train = torch.tensor([-1, 0.25, 3.5, 4.5]).unsqueeze(dim=1)
	return {"dataset_name": "toy-eye", "X_train": X_train, "Y_train": Y_train}
