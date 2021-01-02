# [Output-Constrained Bayesian Neural Networks (OC-BNN)](https://arxiv.org/abs/2010.10969)

This open-source repo is both **(i)** a general purpose implementation of BNNs, as well as **(ii)** an implementation of **OC-BNNs**, which are introduced in our NeurIPS 2020 paper as a way for users to specify output constraints into BNNs. In addition to reproducing the results in our paper, we hope that this codebase will be a helpful resource for researchers working with BNNs.

Feel free to send a pull request for bugs or extra features.

Our NeurIPS paper follows an earlier non-archival [workshop paper](https://arxiv.org/abs/1905.06287) in the 2019 ICML Workshop on Uncertainty and Robustness in Deep Learning. To see an earlier snapshot of this repo as released for the workshop paper, switch to the `workshop` branch. 


## Brief Introduction

If you're not familiar with BNNs, check out Chapter 1 of Wanqian's [senior thesis](https://dash.harvard.edu/handle/1/37364721) for an introduction. Some other good resources are: Radford Neal's [thesis](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.446.9306&rep=rep1&type=pdf) and David MacKay's [paper](https://pdfs.semanticscholar.org/3ce9/da2d2182a2fbc4b460bdb56d3c34110b3e39.pdf?_ga=2.30467516.2029890183.1609586621-2139722624.1609586621), both of which are early seminal works on BNNs. More recent (as of 2020) high-level overviews: a [primer](https://cims.nyu.edu/~andrewgw/caseforbdl.pdf) and [paper](https://arxiv.org/pdf/2002.08791.pdf), both by Andrew Gordon Wilson's group. Finally, the annual [Bayesian Deep Learning workshop](http://bayesiandeeplearning.org/) at NeurIPS is always a good resource for important contributions to the field.

The key challenge that our paper targets is imposing **output constraints** on BNNs, which constrain the posterior predictive distribution _Y|X, D_ for some set of inputs _X_, _e.g. if 1 < X < 2, then the distribution over Y should only contain probability mass on negative output values._ We formulate various tractable prior distributions to allow the BNN to learn such constraints effectively. The ability to incorporate output constraints is useful because they are a currency for **functional, interpretable knowledge**. Model users, who may not have technical ML expertise, can easily specify constraints for prior knowledge they possess, that are not always reflected in the training distribution. For example, a doctor could specify the model to never predict certain classes of drugs if the patient's systolic blood pressure is below, say, 90 mm Hg.

For more details, check out our paper (linked above).


## Getting Started

This codebase is written in [Python 3.7.4](https://www.python.org/downloads/release/python-374/) and built on top of [PyTorch](https://pytorch.org/). The only setup instruction is to run the shell command `pip install -r requirements.txt` to install all dependencies. You might find it helpful to set up a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) for managing dependencies. That's it, you are good to go!

The best way to start using the library is to check out and run `run_tutorial.py` in the root folder. It has an example that trains a vanilla BNN on a toy dataset, and then carries out prediction. It will also show how an output constraint may be specified and learnt. 

In total, there are 4 scripts that you can run (see files themselves for optional command-line arguments):

- **`run_tutorial.py`**: short example on how to use this codebase and its fuctionalities
- **`run_toys.py`**: contains all synthetic experiments in Section 5 (and Appendix D) of our paper
- **`run_apps.py`**: contains high-dimensional applications in Section 6 of our paper (except the MIMIC-III healthcare application in 6.1, for privacy reasons)
- **`run_bakeoff.py`**: compares posterior predictive distributions of various BNN inference methods (see below) on the same examples

In particular, `run_toys.py` contains a comprehensive set of examples of all the various output-constrained priors in our paper, so if you want to implement a specific kind of constraint (e.g. positive or negative or probabilistic), check out the corresponding experiment.

The `repro/` folder contains config files and pre-trained posterior samples of all our experiments, so you can run these scripts with the `--pretrained` flag to immediately generate the relevant plots/results without having to run posterior inference yourself.

Our codebase contains implementations of 4 different inference algorithms. Together, these represent a good diversity of both MCMC and variational methods:

1. [Hamiltonian Monte Carlo](https://arxiv.org/pdf/1206.1901.pdf)
2. [Stein Variational Gradient Descent](https://arxiv.org/pdf/1608.04471.pdf)
3. [Stochastic Gradient Langevin Dynamics](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf)
4. [Bayes by Backprop](https://arxiv.org/pdf/1505.05424.pdf)


## Questions

**How do I add my own dataset?**

You must add a wrapper function in `data/dataloader.py`, check out the file's docstring for detailed instructions.

**How do I add my own inference method?**

You must add a mixin class in `bnn/inference.py`, check out the file's docstring for detailed instructions.

**Where are hyperparameters and such defined?**

A BNN object is instantiated with a YAML config file. See `config.yaml` in the root level of the repo for explanations of each hyperparameter.

**How do I add a constraint to the BNN?**

This is done by calling `bnn.add_deterministic_constraint(...)` or `bnn.add_probabilistic_constraint(...)`. See the method docstrings in `bnn/base.py` for arguments.

**How do I write my own prior distribution?**

Add your own stuff to the `bnn.log_prior()` method in `bnn/base.py`. Currently, we've implemented the baseline isotropic Gaussian prior, as well as our own output-constrained priors. Note that both prior and likelihoods functions are in log-space.


## Citation

```
@inproceedings{yang2020interpretable,
  title={Incorporating {I}nterpretable {O}utput {C}onstraints in {B}ayesian {N}eural {N}etworks},
  author={Yang, Wanqian and Lorch, Lars and Graule, Moritz A and Lakkaraju, Himabindu and Doshi-Velez, Finale},
  booktitle={Advances in {N}eural {I}nformation {P}rocessing {S}ystems},
  url={https://arxiv.org/abs/2010.10969},
  year={2020}
}
```


