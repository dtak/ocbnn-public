
"""
OC-BNN models.
"""

from .base import BayesianNeuralNetwork, RegressorMixin, ClassifierMixin, BinaryClassifierMixin
from .inference import HMCMixin, BBBMixin, SVGDMixin, SGLDMixin


class BNNHMCRegressor(BayesianNeuralNetwork, RegressorMixin, HMCMixin):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)


class BNNHMCClassifier(BayesianNeuralNetwork, ClassifierMixin, HMCMixin):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)


class BNNHMCBinaryClassifier(BayesianNeuralNetwork, BinaryClassifierMixin, HMCMixin):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)


class BNNBBBRegressor(BayesianNeuralNetwork, RegressorMixin, BBBMixin):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)


class BNNBBBClassifier(BayesianNeuralNetwork, ClassifierMixin, BBBMixin):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)


class BNNBBBBinaryClassifier(BayesianNeuralNetwork, BinaryClassifierMixin, BBBMixin):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)


class BNNSVGDRegressor(BayesianNeuralNetwork, RegressorMixin, SVGDMixin):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)


class BNNSVGDClassifier(BayesianNeuralNetwork, ClassifierMixin, SVGDMixin):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)


class BNNSVGDBinaryClassifier(BayesianNeuralNetwork, BinaryClassifierMixin, SVGDMixin):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)


class BNNSGLDRegressor(BayesianNeuralNetwork, RegressorMixin, SGLDMixin):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)


class BNNSGLDClassifier(BayesianNeuralNetwork, ClassifierMixin, SGLDMixin):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)


class BNNSGLDBinaryClassifier(BayesianNeuralNetwork, BinaryClassifierMixin, SGLDMixin):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		