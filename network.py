import numpy as np


def forward(model, input, target):
	for layer in model[:-1]:
		input = layer.forward(input)
	output = model[-1].forward(input, target)
	return output


def backward(model):
	gradient = model[-1].backward()
	for layer in reversed(model[:-1]):
		gradient = layer.backward(gradient)
	return gradient


def load_params(model, W):
	for layer in model:
		w = layer.get_params()
		if w is None:
			continue
		w_shape = w.shape
		w, W = np.split(W, [np.prod(w_shape)])
		layer.set_params(w.reshape(w_shape))


def extract_params(model):
	params = []
	for layer in model:
		w = layer.get_params()
		if w is None:
			continue
		params.append(w)
	W = np.concatenate(params)
	return np.array(W)


def extract_grads(model):
	grads = []
	for layer in model:
		g = layer.get_grads()
		if g is None:
			continue
		grads.append(g)
	dW = np.concatenate(grads)
	return np.array(dW)


def forget(model):
	for layer in model:
		layer.forget()
