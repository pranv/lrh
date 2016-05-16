import numpy as np

from base import Layer
from layer_utils import glorotize, orthogonalize


class Softmax(Layer):
	def forward(self, X):
		exp = np.exp(X)
		probs = exp / np.sum(exp, axis=0, keepdims=True)
		self.probs = probs	
		return probs

	def backward(self, dY):
		Y = self.probs
		dX = Y * dY
		sumdX = np.sum(dX, axis=0, keepdims=True)
		dX -= Y * sumdX
		return dX


class CWRNN(Layer):
	def __init__(self, n_input, n_hidden, n_modules, T_max, last_state_only=False):
		assert(n_hidden % n_modules == 0)

		W = np.random.randn(n_hidden, n_input + n_hidden + 1) 	# +1 for bias, single combined matrix 
																# for recurrent and input projections

		# glorotize and orthogonalize the recurrent and non recurrent aspects respectively
		W[:, :n_input] = glorotize(W[:, :n_input])
		W[:, n_input:-1] = orthogonalize(W[:, n_input:-1])
		
		# time kernel (T_max x n_clocks)
		C = np.repeat(np.arange(T_max).reshape(1, -1), T_max, axis=0)
		C = ((C % np.arange(1, T_max + 1).reshape(-1, 1)) == 0) * 1.0
		C = C.T

		# distribution over clocks for each module (T_max x n_modules)
		d = np.zeros((T_max, n_modules))

		self.softmax = Softmax()

		self.W = W
		self.d = d
		self.C = C
		self.n_input, self.n_hidden, self.n_modules, self.T_max, self.last_state_only = n_input, n_hidden, n_modules, T_max, last_state_only


	def forward(self, X):
		T, n, B = X.shape
		n_input = self.n_input
		n_hidden = self.n_hidden
		n_modules = self.n_modules
		
		D = self.softmax.forward(self.d)				# get activations
		a = np.dot(self.C, D)
		A = np.repeat(a, n_hidden / n_modules, axis=1) 	# for each state in a module
		A = A[:, :, np.newaxis]

		V = np.zeros((T, n_input + n_hidden + 1, B))
		h_new = np.zeros((T, n_hidden, B))
		H_new = np.zeros((T, n_hidden, B))
		H = np.zeros((T, n_hidden, B))

		H_prev = np.zeros((n_hidden, B))

		for t in xrange(T):
			V[t] = np.concatenate([X[t], H_prev, np.ones((1, B))], axis=0)
			h_new[t] = np.dot(self.W, V[t])
			H_new[t] = np.tanh(h_new[t])
			H[t] = A[t] * H_new[t] + (1 - A[t]) * H_prev		# leaky update
			H_prev = H[t]

		self.A, self.a = A, a
		self.V, self.h_new, self.H_new, self.H = V, h_new, H_new, H

		if self.last_state_only:
			return H[-1:]
		else:
			return H

	def backward(self, dH):
		if self.last_state_only:
			last_step_error = dH.copy()
			dH = np.zeros_like(self.H)
			dH[-1:] = last_step_error[:]

		T, _, B = dH.shape
		n_input = self.n_input
		n_hidden = self.n_hidden
		n_modules = self.n_modules

		A = self.A
		V, h_new, H_new, H = self.V, self.h_new, self.H_new, self.H 
		dA, dH_prev, dW, dX = np.zeros_like(A), np.zeros((n_hidden, B)), \
								np.zeros_like(self.W), np.zeros((T, n_input, B))

		for t in reversed(xrange(T)):
			if t == 0:
				H_prev = np.zeros((n_hidden, B))
			else:
				H_prev = H[t - 1]

			dH_t = dH[t] + dH_prev
			
			dH_new = A[t] * dH_t
			dH_prev = (1 - A[t]) * dH_t
			dA[t] = np.sum((H_new[t] - H_prev) * dH_t, axis=1, keepdims=True)

			dh_new = (1.0 - H_new[t] ** 2) * dH_new
			
			dW += np.dot(dh_new, V[t].T)
			dV = np.dot(self.W.T, dh_new)

			dX[t] = dV[:n_input]
			dH_prev += dV[n_input:-1]

		dA = dA[:, :, 0]
		da = dA.reshape(self.T_max, -1, n_hidden / n_modules).sum(axis=-1)
		dD = np.dot(self.C.T, da)
		dd = self.softmax.backward(dD)

		
		self.dW = dW + 0.01 * self.W
		self.dd = dd
		 
		return dX

	def get_params(self):
		W = self.W.flatten()
		d = self.d.flatten()
		return np.concatenate([W, d])

	def set_params(self, P):
		a, b = self.W.size, self.d.size
		W, d = np.split(P, [a])
		self.W = W.reshape(self.W.shape)
		self.d = d.reshape(self.d.shape)

	def get_grads(self):
		dW = self.dW.flatten()
		dd = self.dd.flatten()
		return np.concatenate([dW, dd])

	def clear_grads(self):
		self.dW = None
		self.dd = None

	def forget(self):
		pass

	def remember(self):
		pass

	def print_info(self):
		_D = self.d.copy()
		print 'dominant wave period: ', _D.argmax(axis=0) + 1
		print 'avg. power (all): ', _D.mean()
		print 'avg. power waves: ', self.A.mean()