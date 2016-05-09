import numpy as np

from layers import *
from network import *
from climin import RmsProp, Adam, GradientDescent
from datasets.mnist import loader

import time
import pickle
import os

import matplotlib.pyplot as plt
plt.ion()
plt.style.use('kosh')
plt.figure(figsize=(12, 7))

np.random.seed(np.random.randint(1213))

experiment_name = 'mnist_basic'

permuted = False

n_input = 1
n_hidden = 128
n_modules = 8
n_output = 10

batch_size = 50
learning_rate = 2e-3
niterations = 20000
momentum = 0.9

gradient_clip = (-1.0, 1.0)

save_every = 1000
plot_every = 100

logs = {}

data = loader(batch_size=batch_size, permuted=permuted)

def dW(W):
	load_params(model, W)
	forget(model)	
	inputs, targets = data.fetch_train()
	loss = forward(model, inputs, targets)
	backward(model)

	gradients = extract_grads(model)
	clipped_gradients = np.clip(gradients, gradient_clip[0], gradient_clip[1])
	
	gradient_norm = (gradients ** 2).sum() / gradients.size
	clipped_gradient_norm = (clipped_gradients ** 2).sum() / gradients.size
	
	logs['loss'].append(loss)
	logs['smooth_loss'].append(loss * 0.01 + logs['smooth_loss'][-1] * 0.99)
	logs['gradient_norm'].append(gradient_norm) 
	logs['clipped_gradient_norm'].append(clipped_gradient_norm)
	
	return clipped_gradients


os.system('mkdir results/' + experiment_name)
path = 'results/' + experiment_name + '/'

logs['loss'] = []
logs['val_loss'] = []
logs['smooth_loss'] = [np.log(10)]
logs['gradient_norm'] = []
logs['clipped_gradient_norm'] = []


model = [
			CWRNN(n_input=n_input, n_hidden=n_hidden, n_modules=n_modules, T_max=784, last_state_only=True),
			Linear(n_hidden, 128),
			TanH(),
			Linear(128, n_output),
 			SoftmaxCrossEntropyLoss()
 		]

W = extract_params(model)

optimizer = Adam(W, dW, learning_rate, momentum=momentum)

print 'Approx. Parameters: ', W.size

for i in optimizer:
	if i['n_iter'] > niterations:
		break

	print str(data.epoch) + '\t' + str(i['n_iter']), '\t',
	print logs['loss'][-1], '\t',
	print logs['gradient_norm'][-1]

	if data.epoch_complete:
		inputs, labels = data.fetch_val()
		nsamples = inputs.shape[2]
		inputs = np.split(inputs, nsamples / batch_size, axis=2)
		labels = np.split(labels, nsamples / batch_size, axis=2)
		val_loss = 0
		for j in range(len(inputs)):
			forget(model)
			input = inputs[j]
			label = labels[j]
			val_loss += forward(model, input, label)
		logs['val_loss'].append(val_loss)
		print '..' * 20
		print 'validation loss: ', val_loss

		'''
		inputs, labels = data.fetch_test()
		nsamples = inputs.shape[2]
		inputs = np.split(inputs, nsamples / batch_size, axis=2)
		labels = np.split(labels, nsamples / batch_size, axis=2)

		correct = 0
		for j in range(len(inputs)):
			forget(model1)
			input = inputs[j]
			label = labels[j]
			pred = forward(model1, input, label)
			good = np.sum(label.argmax(axis=1) == pred.argmax(axis=1))
			correct += good

		correct /= float(nsamples)

		print 'accuracy: ', correct * 100
		print '..' * 20
		'''
		
		data.epoch_complete = False



	if i['n_iter'] % save_every == 0:
		print 'serializing model... '
		f = open(path + 'iter_' + str(i['n_iter']) +'.model', 'w')
		pickle.dump(model, f)
		f.close()

	if i['n_iter'] % plot_every == 0:
		plt.clf()
		plt.plot(logs['smooth_loss'], label='training')
		#plt.plot(logs['val_loss'], label='validation')
		plt.legend()
		plt.draw()

print 'serializing logs... '
f = open(path + 'logs.logs', 'w')
pickle.dump(logs, f)
f.close()

print 'serializing final model... '
f = open(path + 'final.model', 'w')
pickle.dump(model, f)
f.close()

plt.savefig(path + 'loss_curve')
