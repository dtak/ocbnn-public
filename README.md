# Output-Constrained Bayesian Neural Networks (OC-BNN)

### Update: A longer version of our [ICML 2019 workshop paper](https://arxiv.org/abs/1905.06287) has recently been accepted at NeurIPS 2020 (paper [here](https://arxiv.org/abs/2010.10969)). This repository will be updated before the conference in December to include the additional work we've done, as well as optimize the existing code here + fix some minor bugs. Stay tuned! Feel free to reach out at wanqian at alumni dot harvard dot edu with any questions.

This open-source repository contains the codebase for the implementation of OC-BNNs, as well as synthetic data used to generate the results in the paper. We have designed this repository not only for paper reproducibility purposes, but also as a general purpose library for implementing both OC-BNNs and generic (vanilla) BNNs. We hope this resource will be helpful for practitioners working with BNNs.

## OC-BNNs: A Brief Primer

We assume some familarity on the reader's part with Bayesian Deep Learning and BNNs. If not, here is a [short introduction](https://github.com/dtak/ocbnn-public/wiki/Bayesian-Neural-Networks-101).  

A known challenge of BNNs is defining an appropriate prior. In classical Bayesian literature, the prior distribution reflects our _a priori_ beliefs over the parameters of interest, before any data has been observed. This is a straightforward task for most simple, low-dimensional cases. However, in the high-dimensional space of deep neural network parameters, doing so becomes a non-trivial task. As a result of the high dimensionality and our poor theoretical understanding of the workings of neural networks, it is largely unknown how different priors affect the quality of the resulting posterior, which prior we should even choose, and how the prior choice relates to the architecture choice, the data distribution or the inference algorithm used. In most cases, an isotropic Gaussian distribution is used as the prior, largely because it is straightforward and tractable. 

Not only so, in many use cases, we do actually have interpretable prior knowledge in *function space*. As a motivating high-stakes example, we want to use a BNN to predict clinical action (whether to give a drug or not) based on physiological features (e.g. blood pressure, blood sugar level). A doctor might already possess some useful knowledge, for example, we should never give the drug if the systolic blood pressure is below 90 _mm Hg_. However, not only does the BNN have to learn the same rule from (supervised) training data, it can still make occasional mistakes near the decision boundary. In most use cases for BNN, such training data is limited and it is unlikely that the model will learn the rule well enough from data alone. The challenge is therefore to incorporate this functional knowledge into a prior distribution.

In this context, our research is an effort towards formulating interpretable and meaningful priors. We present a novel formulation of the prior term (termed the _constraint prior_) that allows the network to incorporate prior knowledge in function (or domain) space, i.e. rules about what values the output `y` can take for certain values of `x`. The resulting network is called an output-constrained BNN (OC-BNN). Our method is a general approach that works for a broad range of functional constraints and is amenable with most inference algorithms. OC-BNNs learn the posterior distribution that not only explains the data well, but also respects whatever output constraints that the user has expressed. 

### Constraint Priors

![toy-results](https://github.com/dtak/ocbnn-public/blob/master/images/toyresults.png "Results of synthetic experiments in paper.")

The main idea behind CPs is to explictly include a "constraint term" _**g(W|C)**_, which measures how much the parameters _**W**_ respect the constraints _**C**_. The overall prior term is then _**p(W) = f(W) \* g(W|C)**_, where _**f(W)**_ is just the standard isotropic Gaussian term. It must be emphasized that (1) _**g(W|C)**_ is a proper probability distribution, and that (2) it is still defined in parameter space instead of functional space. We consider two different kinds of constraints, and formulate the appropriate _**g**_ for each constraint type:

1. Positive constraints _**C+**_: A positive constraint specifies where the output `y` should be for certain input `x` (and can be undefined everywhere else in input-space). Then _**g(W|C)**_ can be any valid distribution that assigns a higher probability to the correct outputs, e.g. a Gaussian for continuous space (regression) and a Dirichlet for discrete space (classification).  

1. Negative constraints _**C-**_: A negative constraint specifies where the output `y` should _not_ be (forbidden) for certain input `x` (and can be undefined everywhere else in input-space). Then _**g(W|C)**_ is an inverse-sigmoidal distribution penalizing forbidden outputs. We define this only for continuous cases. 

We sample points from input space to approximate _**g(W|C)**_ since _**C**_ can contain infinitely many points. Inference can be carried out as per usual, except with the new prior formulation. You can find more details in the paper.


## Using This Library

This library allows you to run an OC-BNN with either positive or negative constraints, or just a plain ol' BNN without any constraints. Regardless of the presence of constraints, you can perform inference using any of the 4 inference algorithms we have implemented:

1. [Hamiltonian Monte Carlo](https://arxiv.org/pdf/1206.1901.pdf)
2. [Stein Variational Gradient Descent](https://arxiv.org/pdf/1608.04471.pdf)
3. [Stochastic Gradient Langevin Dynamics](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf)
4. [Bayes by Backprop](https://arxiv.org/pdf/1505.05424.pdf)

These represent both MCMC and variational methods (in the case of SVGD, both at the same time ;). Here is a nice picture comparing the posterior distributions of all 4 inference methods on both regression and classification:

![bakeoff-image](https://github.com/dtak/ocbnn-public/blob/master/images/bakeoff.png "{HMC, SVGD, SGLD, BBB}")


### Getting Started

This repository is written in [Python 3.7](https://www.python.org/downloads/release/python-370/), so first ensure that you have that installed. You might also find it helpful to install a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). Next, ensure that you have all the pre-requisite packages installed by running the shell command `pip install -r requirements.txt`. All code is implemented with the [PyTorch](https://pytorch.org/) framework.

### Executable Scripts 

There are 4 scripts that you can run:

- `main_example.py` contains an example each of: (i) a vanilla BNN, (ii) OC-BNN with negative constraints, and (iii) OC-BNN with positive constraints.  
- `main_toys.py` contains all 6 synthetic experiments showcased in the paper (with the vanilla BNN as the baseline). [Results [here](https://github.com/dtak/ocbnn-public/blob/master/images/toyresults.png)]
- `main_bakeoff.py` contains an example each of all 4 inference algorithms, run on both (i) a regression task and (ii) a classification task. [Results [here](https://github.com/dtak/ocbnn-public/blob/master/images/bakeoff.png)]
- `main_multi.py` contains an example on a high-dimensional dataset, using a public dataset from UCI. [Results [here](https://github.com/dtak/ocbnn-public/blob/master/images/energyresults.png)]

You can refer to these scripts as examples to write your own OC-BNN (or plain BNN). A complete BNN run requires only 4 lines: 

1. Instantiation of the BNN object. Instantiating a BNN requires a JSON configfile, which is where you define all the hyperparameters. [This page](https://github.com/dtak/ocbnn-public/wiki/JSON-Configfile) contains documentation on the configfile.   

2. Load the dataset by calling `self.load(**datafunc())` where `datafunc()` is a wrapper function over the dataset, see more details in [this page](https://github.com/dtak/ocbnn-public/wiki/Dataset-Loading).   

3. (Optional) Add any (positive or negative) constraints by calling `self.add_positive_constraint()` or `self.add_negative_constraint()`. [This page](https://github.com/dtak/ocbnn-public/wiki/Defining-Constraints) contains documentation on defining positive and negative constraints.

4. Running the inference algorithm itself by calling `self.infer()`. 

Additional post-inference functionality includes plotting (written for 1D regression and 2D 3-class classification tasks only) and computing evaluation metrics. It is possible to run multiple inferences per BNN object, as you can see in `main_toys.py`.


### Codebase Structure

A BNN object (as instantiated in Step 1 above) is defined by the inference method used + the type of task (regression or classification). For example, the object `BNNSVGDRegressor` is the object for performing regression using SVGD inference. As such, calling `self.infer()` on this object automatically runs SVGD. 

- The folder `bnn/` contains all BNN code. The file `bnn.py` contains the base objects `BayesianNeuralNetwork`, `BNNRegressor` and `BNNClassifier`. Each inference method has its own file, and contains the objects `BNNSomeMethod`, `BNNSomeMethodRegressor` and `BNNSomeMethodClassifier`.  
- The folder `history/` contains the associated data saved for any BNN object: (i) a copy of `config.json` when the BNN is instantiated, (ii) saved samples (or variational parameters) as `.pt` files for each inference run, and (iii) any saved plots as `.png` files.  
- The folder `data/` contains synthetic datasets.  
- The folder `images/` contains some plots.  
- The folder `configs/` contains the configfiles for all the `main_*.py` scripts, so that you can replicate these scripts.

Here is a diagram for the inheritance structure of all objects:

![inheritance-image](https://github.com/dtak/ocbnn-public/blob/master/images/objects.png "Inheritance Structure")


### Writing your own BNN

This codebase is designed to make customizability easy:

1. If you want to write a custom prior function, simply create a new method `self.log_custom_prior()` in `BNNRegressor` and/or `BNNClassifier`, and modify `self.log_prior()` in `BayesianNeuralNetwork` to include an condition for your custom prior. Your `config.json` should then have the correct argument for the key `prior_type`.

2. If you will like to add an inference algorithm, you will need to create a new file `newmethod.py` that should contain the objects `BNNNewMethod(BayesianNeuralNetwork)`, `BNNNewMethodRegressor(BNNNewMethod, BNNRegressor)` and `BNNNewMethodClassifier(BNNNewMethod, BNNClassifier)`, following the style of existing inference algorithms. 

Feel free to send a pull request if you think your custom features will have widespread usefulness.


## Citation

```
@inproceedings{yang2019output,
  title={Output-Constrained Bayesian Neural Networks},
  author={Yang, Wanqian and Lorch, Lars and Graule, Moritz A and Srinivasan, Srivatsan and Suresh, Anirudh and Yao, Jiayu and Pradier, Melanie F and Doshi-Velez, Finale},
  booktitle = {2019 ICML Workshop on Uncertainty and Robustness in Deep Learning (UDL) and Workshop on Understanding and Improving Generalization in Deep Learning},
  url={https://arxiv.org/abs/1905.06287},
  year={2019}
}
```


