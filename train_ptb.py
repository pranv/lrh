import numpy as np

from layers import *
from network import *
from climin import RmsProp, Adam, GradientDescent
from data.text import loader

import time
import pickle
import os

import matplotlib.pyplot as plt
plt.ion()
plt.style.use('kosh')
plt.figure(figsize=(12, 7))


np.random.seed(np.random.randint(1213))

experiment_name = 'penn_random_init_noscale_0.5binaryActi_0.01WDecay'

text_file = 'ptb.txt'

vocabulary_size = 49

n_output = n_input = vocabulary_size
n_hidden = 1024
n_modules = 8
noutputs = vocabulary_size

sequence_length = 100

batch_size = 64
learning_rate = 2e-3
niterations = 100000
momentum = 0.9

forget_every = 100
gradient_clip = (-1.0, 1.0)

sample_every = 1000
save_every = 1000
plot_every = 100

logs = {}

data = loader('data/' + text_file, sequence_length, batch_size)


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
logs['accuracy'] = []
logs['smooth_loss'] = [np.log(49)]
logs['gradient_norm'] = []
logs['clipped_gradient_norm'] = []

model = [
			CWRNN2(n_input=n_input, n_hidden=n_hidden, n_modules=n_modules, T_max=sequence_length),
			Linear(n_hidden, n_output),
 			SoftmaxCrossEntropyLoss()
 		]

W = extract_params(model)

optimizer = Adam(W, dW, learning_rate, momentum=momentum)

print 'Approx. Parameters: ', W.size

for i in optimizer:
	if i['n_iter'] > niterations:
		break

	print '\n\n'
	print str(i['n_iter']), '\t',
	print logs['loss'][-1], '\t',
	print logs['gradient_norm'][-1]
	print_info(model)
	print '\n', '----' * 20, '\n'

	if i['n_iter'] % sample_every == 0:
		forget(model)
		x = np.zeros((20, vocabulary_size, 1))
		input, _ = data.fetch_train()
		x[:20, :, :] = input[:20, :, 0:1]
		ixes = []
		for t in xrange(1000):
			forward(model, np.array(x), 1.0)
			p = model[-1].probs
			p = p[-1]
			ix = np.random.choice(range(vocabulary_size), p=p.ravel())
			x = np.zeros((1, vocabulary_size, 1))
			x[0, ix, 0] = 1
			ixes.append(ix)
		sample = ''.join(data.decoder.to_c[ix] for ix in ixes)
		print '----' * 20
		print sample
		print '----' * 20
		forget(model)
	
	if i['n_iter'] % save_every == 0:
		print 'serializing model... '
		f = open(path + 'iter_' + str(i['n_iter']) +'.model', 'w')
		pickle.dump(model, f)
		f.close()

	if i['n_iter'] % plot_every == 0:
		plt.clf()
		plt.plot(logs['smooth_loss'])
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
