
"""
High-dimensional applications in Section 6 of our paper: https://arxiv.org/pdf/2010.10969.pdf.
Due to data privacy reasons, we omit Section 6.1: Clinical Action Prediction with the MIMIC-III database.

Default behavior when running this file::
	All experiments are fully run from scratch, i.e. full inference is performed.
	The config files from the `repro/` folder are used.
	As BNN inference is approximate, results will slightly deviate from those in the paper, especially for VI methods.

There are three optional command-line arguments:
	--debug :: run experiments in debug mode (only a couple iterations of inference)
	--pretrained :: use our pre-trained posterior samples and AOCP parameters
	--custom_config <filename.yaml> :: use your own config file --> <filename.yaml> if specified, else, `config.yaml` in the root directory

Note: The pretrained samples in `repro/` are not identical to the samples used to produce the actual tables in our paper. 
They are separate runs of the same experiments to prove reproducibility of our results.

"""

import numpy as np
import torch
import logging
import argparse

from bnn import *
from bnn.utils import *
from data.dataloader import *


def recid_evaluate_high_risk(bnn):
	""" Recidivism prediction task: evaluate high-risk fraction of both groups """
	samples, _ = bnn.all_bayes_samples[-1]
	preds = (bnn.predict(samples, bnn.X_train, return_probs=True).mean(dim=0) >= 0.5).float()

	eval_data = torch.stack((bnn.X_train_race, preds)).t()
	aa_highrisk = len(eval_data[(eval_data[:, 0] == 1.0) & (eval_data[:, 1] == 1.0)])
	aa_total = len(eval_data[(eval_data[:, 0] == 1.0)])
	nonaa_highrisk = len(eval_data[(eval_data[:, 0] == 0.0) & (eval_data[:, 1] == 1.0)])
	nonaa_total = len(eval_data[(eval_data[:, 0] == 0.0)])
	return aa_highrisk, aa_total, nonaa_highrisk, nonaa_total


def recidivism_task():
	""" Section 6.2: Recidivism Prediction Task """

	# With race as an explicit feature, baseline.
	bnn = BNNSVGDBinaryClassifier(uid="recid", configfile=args.custom_config or "repro/recid.yaml")
	bnn.load(**compas_dataset(csv_xfilename="data/compas_X.csv", csv_yfilename="data/compas_Y.csv"))
	logging.info(f'[{bnn.uid}] Dataset <{bnn.dataset_name}>: {(100 * sum(bnn.Y_train) // bnn.N_train):.2f}% of training points are positive.')
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/recid_svgd1.pt', 'svgd_baseline')
	else:
		bnn.infer()
	logging.info(f'[{bnn.uid}] Baseline evaluation, with race as feature:')
	acc, f1 = eval_accuracy_and_f1_score(bnn, is_binary=True, X_eval=bnn.X_train, Y_eval=bnn.Y_train)
	logging.info(f'[{bnn.uid}]   Train accuracy: {acc:.3f}.')
	logging.info(f'[{bnn.uid}]   Train F1 score: {f1:.3f}.')
	aa_highrisk, aa_total, nonaa_highrisk, nonaa_total = recid_evaluate_high_risk(bnn)
	logging.info(f'[{bnn.uid}]   African American High-Risk Fraction: {(aa_highrisk / aa_total):.3f}.')
	logging.info(f'[{bnn.uid}]   Non-African American High-Risk Fraction: {(nonaa_highrisk / nonaa_total):.3f}.')
	
	# With race as an explicit feature, OC-BNN.
	cdomain = ()
	for i in range(9):
		if i == 1:
			cdomain += (0., 1.)
		else:
			mm = bnn.X_train[:,i].mean().item()
			ss = bnn.X_train[:,i].std().item()
			cdomain += (mm - 0.2 * ss, mm + 0.2 * ss)
	bnn.add_probabilistic_constraint(constrained_domain=cdomain, prob_func=lambda X: X[:,1], prior_type="gaussian_aocp")
	if args.pretrained:
		bnn.load_gaussian_aocp_parameters('repro/recid_aocp.pt')
	else:
		bnn.learn_gaussian_aocp()
	bnn.update_config(use_ocbnn=True)
	if args.pretrained:
		bnn.load_bayes_samples('repro/recid_svgd2.pt', 'svgd_ocbnn')
	else:
		bnn.infer()
	logging.info(f'[{bnn.uid}] OC-BNN evaluation, with race as feature:')
	acc, f1 = eval_accuracy_and_f1_score(bnn, is_binary=True, X_eval=bnn.X_train, Y_eval=bnn.Y_train)
	logging.info(f'[{bnn.uid}]   Train accuracy: {acc:.3f}.')
	logging.info(f'[{bnn.uid}]   Train F1 score: {f1:.3f}.')
	aa_highrisk, aa_total, nonaa_highrisk, nonaa_total = recid_evaluate_high_risk(bnn)
	logging.info(f'[{bnn.uid}]   African American High-Risk Fraction: {(aa_highrisk / aa_total):.3f}.')
	logging.info(f'[{bnn.uid}]   Non-African American High-Risk Fraction: {(nonaa_highrisk / nonaa_total):.3f}.')


	# Without race as feature, baseline.
	bnn = BNNSVGDBinaryClassifier(uid="recid-no-race", configfile=args.custom_config or "repro/recid-no-race.yaml")
	bnn.load(**compas_dataset(csv_xfilename="data/compas_X.csv", csv_yfilename="data/compas_Y.csv", with_race=False))
	if args.debug:
		bnn.debug_mode()
	if args.pretrained:
		bnn.load_bayes_samples('repro/recid-no-race_svgd1.pt', 'svgd_baseline')
	else:
		bnn.infer()
	logging.info(f'[{bnn.uid}] Baseline evaluation, with race as feature:')
	acc, f1 = eval_accuracy_and_f1_score(bnn, is_binary=True, X_eval=bnn.X_train, Y_eval=bnn.Y_train)
	logging.info(f'[{bnn.uid}]   Train accuracy: {acc:.3f}.')
	logging.info(f'[{bnn.uid}]   Train F1 score: {f1:.3f}.')
	aa_highrisk, aa_total, nonaa_highrisk, nonaa_total = recid_evaluate_high_risk(bnn)
	logging.info(f'[{bnn.uid}]   African American High-Risk Fraction: {(aa_highrisk / aa_total):.3f}.')
	logging.info(f'[{bnn.uid}]   Non-African American High-Risk Fraction: {(nonaa_highrisk / nonaa_total):.3f}.')

	# Without race as feature, OC-BNN.
	bnn.add_probabilistic_constraint(constrained_domain=cdomain[:-2], prob_func=lambda X: X[:,1], prior_type="gaussian_aocp")
	if args.pretrained:
		bnn.load_gaussian_aocp_parameters('repro/recid-no-race_aocp.pt')
	else:
		bnn.learn_gaussian_aocp()
	bnn.update_config(use_ocbnn=True)
	if args.pretrained:
		bnn.load_bayes_samples('repro/recid-no-race_svgd2.pt', 'svgd_ocbnn')
	else:
		bnn.infer()
	logging.info(f'[{bnn.uid}] OC-BNN evaluation, with race as feature:')
	acc, f1 = eval_accuracy_and_f1_score(bnn, is_binary=True, X_eval=bnn.X_train, Y_eval=bnn.Y_train)
	logging.info(f'[{bnn.uid}]   Train accuracy: {acc:.3f}.')
	logging.info(f'[{bnn.uid}]   Train F1 score: {f1:.3f}.')
	aa_highrisk, aa_total, nonaa_highrisk, nonaa_total = recid_evaluate_high_risk(bnn)
	logging.info(f'[{bnn.uid}]   African American High-Risk Fraction: {(aa_highrisk / aa_total):.3f}.')
	logging.info(f'[{bnn.uid}]   Non-African American High-Risk Fraction: {(nonaa_highrisk / nonaa_total):.3f}.')


def credit_evaluate_recourse(bnn):
	""" Credit scoring task: evaluate effort of recourse on young adults. """
	samples, _ = bnn.all_bayes_samples[-1]
	preds = bnn.predict(samples, bnn.X_test, return_probs=True).mean(dim=0).argmax(dim=1)
	is_young = (bnn.X_test[:, 1] * bnn.X_train_std[1] + bnn.X_train_mean[1] < 35).nonzero().squeeze(dim=1)
	X_test_young = bnn.X_test[is_young]
	Y_test_young = bnn.Y_test[is_young]
	preds_young = preds[is_young]
	
	gt_ruul_neg = X_test_young[(Y_test_young == 0).nonzero(), 0].mean()
	gt_ruul_pos = X_test_young[(Y_test_young == 1).nonzero(), 0].mean()
	preds_ruul_neg = X_test_young[(preds_young == 0).nonzero(), 0].mean()
	preds_ruul_pos = X_test_young[(preds_young == 1).nonzero(), 0].mean()
	return gt_ruul_neg, gt_ruul_pos, preds_ruul_neg, preds_ruul_pos 


def credit_scoring_task():
	""" Section 6.3: Credit Scoring Task """

	bnn = BNNBBBClassifier(uid='credit', configfile=args.custom_config or "repro/credit.yaml")
	bnn.load(**credit_dataset(csv_filename="data/give_me_some_credit.csv"))
	logging.info(f'[{bnn.uid}] Dataset <{bnn.dataset_name}> blind dataset: {bnn.X_train_blind.shape[0]} training points.')
	logging.info(f'[{bnn.uid}] Dataset <{bnn.dataset_name}>: {(100 * sum(bnn.Y_train) // bnn.N_train):.2f}% of training points are positive.')
	logging.info(f'[{bnn.uid}] Dataset <{bnn.dataset_name}>: {(100 * sum(bnn.Y_train_blind) // bnn.X_train_blind.shape[0]):.2f}% of blind training points are positive.')
	logging.info(f'[{bnn.uid}] Dataset <{bnn.dataset_name}>: {(100 * sum(bnn.Y_test) // bnn.N_test):.2f}% of test points are positive.')
	if args.debug:
		bnn.debug_mode()	
	
	# Baseline inference and evaluation on full dataset.
	if args.pretrained:
		bnn.load_bayes_samples('repro/credit_bbb1.pt', 'bbb_baseline')
	else:
		bnn.infer()
	logging.info(f'[{bnn.uid}] Baseline evaluation on full dataset:')
	acc, f1 = eval_accuracy_and_f1_score(bnn)
	logging.info(f'[{bnn.uid}]   Test accuracy: {acc:.3f}.')
	logging.info(f'[{bnn.uid}]   Test F1 score: {f1:.3f}.')
	gt_ruul_neg, gt_ruul_pos, preds_ruul_neg, preds_ruul_pos = credit_evaluate_recourse(bnn)
	logging.info(f'[{bnn.uid}]   Effort of recourse: {preds_ruul_pos - preds_ruul_neg:.3f}.')
	logging.info(f'[{bnn.uid}]   Ground-truth effort of recourse: {gt_ruul_pos - gt_ruul_neg:.3f}.')

	# Calculating constrained domain boundaries for OC-BNNs.
	# Define constrained region as: young with high RUUL. For other features, sample around training distribution mean.
	age_lb = bnn.X_train[:, 1].min().item()
	age_ub = (35 - bnn.X_train_mean[1]) / bnn.X_train_std[1]
	Y_train_young = bnn.Y_train[(bnn.X_train[:,1] * bnn.X_train_std[1] + bnn.X_train_mean[1] < 35).nonzero().squeeze(dim=1)]
	X_train_young = bnn.X_train[(bnn.X_train[:,1] * bnn.X_train_std[1] + bnn.X_train_mean[1] < 35).nonzero().squeeze(dim=1)]
	ruul_lb = np.quantile(X_train_young[:, 0], 0.6)
	ruul_ub = np.quantile(X_train_young[:, 0], 1.0)
	cdomain = (ruul_lb, ruul_ub, age_lb, age_ub)
	for i in range(2, 10):
		dim_mean = bnn.X_train[bnn.X_train[:,1] * bnn.X_train_std[1] + bnn.X_train_mean[1] < 35].mean(dim=0)[i].item()
		dim_std = bnn.X_train[bnn.X_train[:,1] * bnn.X_train_std[1] + bnn.X_train_mean[1] < 35].std(dim=0)[i].item()
		cdomain = cdomain + (dim_mean - 0.2 * dim_std, dim_mean + 0.2 * dim_std)

	# Baseline inference and evaluation on blind dataset.
	bnn.X_train, bnn.Y_train = bnn.X_train_blind, bnn.Y_train_blind
	bnn.N_train = bnn.X_train.shape[0]
	if args.pretrained:
		bnn.load_bayes_samples('repro/credit_bbb2.pt', 'bbb_blind_baseline')
	else:
		bnn.infer()
	logging.info(f'[{bnn.uid}] Baseline evaluation on blind dataset:')
	acc, f1 = eval_accuracy_and_f1_score(bnn)
	logging.info(f'[{bnn.uid}]   Test accuracy: {acc:.3f}.')
	logging.info(f'[{bnn.uid}]   Test F1 score: {f1:.3f}.')
	gt_ruul_neg, gt_ruul_pos, preds_ruul_neg, preds_ruul_pos = credit_evaluate_recourse(bnn)
	logging.info(f'[{bnn.uid}]   Effort of recourse: {preds_ruul_pos - preds_ruul_neg:.3f}.')
	logging.info(f'[{bnn.uid}]   Ground-truth effort of recourse: {gt_ruul_pos - gt_ruul_neg:.3f}.')

	# OC-BNN inference and evaluation.
	bnn.add_deterministic_constraint(constrained_domain=cdomain, forbidden_classes=[1], prior_type="positive_dirichlet_cocp")
	bnn.update_config(use_ocbnn=True)
	if args.pretrained:
		bnn.load_bayes_samples('repro/credit_bbb3.pt', 'bbb_ocbnn')
	else:
		bnn.infer()
	logging.info(f'[{bnn.uid}] OC-BNN evaluation:')
	acc, f1 = eval_accuracy_and_f1_score(bnn)
	logging.info(f'[{bnn.uid}]   Test accuracy: {acc:.3f}.')
	logging.info(f'[{bnn.uid}]   Test F1 score: {f1:.3f}.')
	gt_ruul_neg, gt_ruul_pos, preds_ruul_neg, preds_ruul_pos = credit_evaluate_recourse(bnn)
	logging.info(f'[{bnn.uid}]   Effort of recourse: {preds_ruul_pos - preds_ruul_neg:.3f}.')
	logging.info(f'[{bnn.uid}]   Ground-truth effort of recourse: {gt_ruul_pos - gt_ruul_neg:.3f}.')



if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(name)s-%(levelname)s: %(message)s")
	logging.info("Running OC-BNN library...")
	parser = argparse.ArgumentParser(description='Process command-line arguments.')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--pretrained', action='store_true')
	parser.add_argument('--custom_config', nargs='?', const='config.yaml')
	args = parser.parse_args()
	recidivism_task()
	credit_scoring_task()
	logging.info("Completed.")