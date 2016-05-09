import numpy as np

from base import Layer


class SoftmaxCrossEntropyLoss(Layer):
	def forward(self, X, target):
		T, _, B = X.shape

		exp = np.exp(X)
		probs = exp / np.sum(exp, axis=1, keepdims=True)
		
		loss = -1.0 * np.sum(target * np.log(probs)) / (T * B)

		self.probs = probs
		self.target, self.T, self.B =  target, T, B
		
		return loss

	def backward(self):
		target, T, B = self.target, self.T, self.B
		
		dX = self.probs - target
		
		return dX / (T * B)