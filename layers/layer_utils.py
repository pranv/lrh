import numpy as np


def glorotize(W):
	W *= np.sqrt(6)
	W /= np.sqrt(np.sum(W.shape))
	return W 


def orthogonalize(W):
	W, _, _ = np.linalg.svd(W)
	return W
