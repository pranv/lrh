import numpy as np

from __init__ import *


def finite_difference_check(layer, fwd, all_values, backpropagated_gradients, names, delta, error_threshold):
	error_count = 0
	for v in range(len(names)):
		values = all_values[v]
		dvalues = backpropagated_gradients[v]
		name = names[v]
		
		for i in range(values.size):
			actual = values.flat[i]
			values.flat[i] = actual + delta
			loss_minus = fwd()
			values.flat[i] = actual - delta
			loss_plus = fwd()
			values.flat[i] = actual
			backpropagated_gradient = dvalues.flat[i]
			numerical_gradient = (loss_minus - loss_plus) / (2 * delta)
			
			if numerical_gradient == 0 and backpropagated_gradient == 0:
				error = 0 
			elif abs(numerical_gradient) < 1e-7 and abs(backpropagated_gradient) < 1e-7:
				error = 0 
			else:
				error = abs(backpropagated_gradient - numerical_gradient) / abs(numerical_gradient + backpropagated_gradient)
			
			if error > error_threshold:
				print 'FAILURE!!!\n'
				print '\tparameter: ', name, '\tindex: ', np.unravel_index(i, values.shape)
				print '\tvalues: ', actual
				print '\tbackpropagated_gradient: ', backpropagated_gradient 
				print '\tnumerical_gradient', numerical_gradient 
				print '\terror: ', error
				print '----' * 20
				print '\n\n'

				error_count += 1

	if error_count == 0:
		print layer.__class__.__name__, 'Layer Gradient Check Passed'
	else:
		param_count = 0
		for val in all_values:
			param_count += val.size
		print layer.__class__.__name__, 'Layer Gradient Check Failed for {}/{} parameters'.format(error_count, param_count)


def test_layer(layer):
	P = layer.get_params()
	Y = layer.forward(X)
	T_rand = np.random.randn(*Y.shape)		# random target for a multiplicative loss
	loss = np.sum(Y * T_rand)				# loss

	dX = layer.backward(T_rand)
	dP = layer.get_grads()

	def fwd():
		layer.forget()
		layer.set_params(P)
		return np.sum(layer.forward(X) * T_rand)

	all_values = [X, P]
	backpropagated_gradients = [dX, dP]
	names = ['X', 'P']


	finite_difference_check(layer, fwd, all_values, backpropagated_gradients, names, delta, error_threshold)


def test_loss(layer):
			
	exp = np.exp(np.random.random(X.shape))
	target = exp / np.sum(exp, axis=1, keepdims=True)	# random target for a multiplicative loss
	loss = layer.forward(X, target)

	dX = layer.backward()
	
	def fwd():
		return layer.forward(X, target)

	all_values = [X]
	backpropagated_gradients = [dX]
	names = ['X']

	finite_difference_check(layer, fwd, all_values, backpropagated_gradients, names, delta, error_threshold)


def test_activation(layer):
	Y = layer.forward(X)
	T_rand = np.random.randn(*Y.shape)		# random target for a multiplicative loss
	loss = np.sum(Y * T_rand)				# loss

	dX = layer.backward(T_rand)

	def fwd():
		return np.sum(layer.forward(X) * T_rand)

	all_values = [X]
	backpropagated_gradients = [dX]
	names = ['X']

	finite_difference_check(layer, fwd, all_values, backpropagated_gradients, names, delta, error_threshold)


delta = 1e-4
error_threshold = 1e-3
time_steps = 5
n_input = 3
batch_size = 7

X = np.random.randn(time_steps, n_input, batch_size)

n_output = 20
layer = Linear(n_input, n_output)
test_layer(layer=layer)

layer = SoftmaxCrossEntropyLoss()
test_loss(layer=layer)

layer = CWRNN(n_input=n_input, n_hidden=8, n_modules=4, T_max=time_steps, last_state_only=False)
test_layer(layer=layer)

layer = CWRNN_NORM(n_input=n_input, n_hidden=8, n_modules=4, T_max=time_steps, last_state_only=False)
test_layer(layer=layer)

layer = CWRNN_L1(n_input=n_input, n_hidden=8, n_modules=4, T_max=time_steps, last_state_only=False)
test_layer(layer=layer)

layer = TanH()
test_activation(layer=layer)

layer = Sigmoid()
test_activation(layer=layer)
