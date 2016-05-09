import numpy as np

from base import Layer
from layer_utils import glorotize


class Linear(Layer):
	def __init__(self, n_input, n_output):
		W = np.random.randn(n_output, n_input + 1)						# +1 for the bias
		W = glorotize(W)
		self.W = W

		self.n_input = n_input
		self.n_output = n_output

	def forward(self, X):
		T, n, B = X.shape
		
		X_flat = X.swapaxes(0, 1).reshape(n, -1)						# flatten over time and batch
		X_flat = np.concatenate([X_flat, np.ones((1, B * T))], axis=0)	# add bias
		
		Y = np.dot(self.W, X_flat)
		Y = Y.reshape((-1, T, B)).swapaxes(0,1)
		
		self.X_flat = X_flat

		return Y

	def backward(self, dY):
		T, n, B = dY.shape
		
		dY = dY.swapaxes(0,1).reshape(n, -1)
		
		self.dW = np.dot(dY, self.X_flat.T)
		
		dX = np.dot(self.W.T, dY)
		
		dX = dX[:-1]								# skip the bias we added above
		dX = dX.reshape((-1, T, B)).swapaxes(0,1)
		
		return dX

	def get_params(self):
		return self.W.flatten()

	def set_params(self, W):
		self.W = W.reshape(self.W.shape)

	def get_grads(self):
		return self.dW.flatten()

	def clear_grads(self):
		self.dW = None
