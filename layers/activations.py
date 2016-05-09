import numpy as np

from base import Layer

class TanH(Layer):
	def forward(self, X):
		Y = np.tanh(X)
		self.Y = Y
		return Y

	def backward(self, dY):
		Y = self.Y
		dX = (1.0 - Y ** 2) * dY
		return dX


class Sigmoid(Layer):
	def forward(self, X):
		Y = 1.0 / (1.0 + np.exp(-X))
		self.Y = Y
		return Y

	def backward(self, dY):
		Y = self.Y
		dX = Y * (1.0 - Y) * dY
		return dX