
"""
UCI datasets.
"""

import torch
import pandas as pd


def energyset():
	""" UCI "Energy Efficiency" dataset: https://archive.ics.uci.edu/ml/datasets/Energy+efficiency. 
		Regression on Y1 using {X1, ..., X8}.
		Standardized.
	"""
	df = pd.read_excel("data/energy_efficiency.xlsx")

	XY = torch.tensor(df[["X1","X2","X3","X4","X5","X6","X7","X8","Y1"]].values)
	XY = XY.float()

	train, test = XY[:int(0.85 * len(XY))], XY[int(0.85 * len(XY)):]
	means = train.mean(dim=0)
	stds = train.std(dim=0)
	train = (train - means) / stds
	test = (test - means) / stds
	train = train[train[:,2] > -1.0]

	X_train, Y_train = train[:,:-1], train[:,-1].unsqueeze(1)
	X_test, Y_test = test[:,:-1], test[:,-1].unsqueeze(1)

	return {"dataset_name": "uci_energy", "X_train": X_train, "Y_train": Y_train, "X_test": X_test, "Y_test": Y_test}
