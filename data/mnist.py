import numpy as np

import gzip, cPickle, sys

def to_categorical(y):
    y = np.asarray(y)
    Y = np.zeros((len(y), 10))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

class loader(object):
	def __init__(self, batch_size=50, permuted=False):
		path = 'data/mnist.pkl.gz'
		f = gzip.open(path, 'rb')
		(X_train, y_train), (X_val, y_val), (X_test, y_test) = cPickle.load(f)
		f.close()

		X_train = X_train.reshape(X_train.shape[0], -1, 1)
		X_val = X_val.reshape(X_val.shape[0], -1, 1)
		X_test = X_test.reshape(X_test.shape[0], -1, 1)

		X_train = X_train.swapaxes(0, 1).swapaxes(1, 2)
		X_val = X_val.swapaxes(0, 1).swapaxes(1, 2)
		X_test = X_test.swapaxes(0, 1).swapaxes(1, 2)

		if permuted:
			p = range(28*28)
			np.random.shuffle(p)
			X_train = X_train[p]
			X_val = X_val[p]
			X_test = X_test[p]

		self.i = 0

		self.X_train = X_train
		self.X_val = X_val 
		self.X_test = X_test
		self.y_train = to_categorical(y_train).T.reshape(1, 10, -1)
		self.y_val = to_categorical(y_val ).T.reshape(1, 10, -1)
		self.y_test = to_categorical(y_test).T.reshape(1, 10, -1)

		self.batch_size = batch_size
		self.permuted = permuted
		self.epoch = 1
		self.epoch_complete = False

	def fetch_train(self):
		X = self.X_train[:, :, self.i * self.batch_size: (self.i + 1) * self.batch_size]
		y = self.y_train[:, :, self.i * self.batch_size: (self.i + 1) * self.batch_size]
		self.i = (self.i + 1)
		if (self.i * self.batch_size) >= self.X_train.shape[2]:
			self.epoch_complete = True
			self.epoch += 1
			self.i = self.i % (self.X_train.shape[2] / self.batch_size)
		return (X, y)

	def fetch_val(self):
		return self.X_val, self.y_val

	def fetch_test(self):
		return self.X_test, self.y_test
