import numpy as np

class OneHot(object):
	def __init__(self, alphabet_size, char_to_i):
		self.alphabet_size = alphabet_size
		self.matrix = np.eye(alphabet_size, dtype='uint8')
		self.to_i = char_to_i

	def __call__(self, chars):
		I = np.zeros((len(chars), self.alphabet_size))
		for c in range(len(chars)):
			i = self.to_i[chars[c]]
			I[c][i] = 1
		return I


class UnOneHot(object):
	def __init__(self, i_to_char):
		self.to_c = i_to_char

	def __call__(self, vectors):
		chars = ''
		for vector in vectors:
			i = vector.argmax()
			chars += self.to_c[i]
		return chars

class loader(object):
	def __init__(self, filename, sequence_length, batch_size):
		f = open(filename, 'r')
		lines = f.readlines()

		string = ''.join(lines)

		vocabulary = list(set(string))
		vocabulary_size = len(vocabulary)
		data_size = len(string)

		char_to_i = {ch:i for i,ch in enumerate(vocabulary)}
		i_to_char = {i:ch for i,ch in enumerate(vocabulary)}

		encoder = OneHot(vocabulary_size, char_to_i)
		decoder = UnOneHot(i_to_char)

		chars_per_batch = data_size / batch_size
		total_used_chars = (data_size / chars_per_batch) * chars_per_batch
		string = string[:total_used_chars]
		data_size = len(string)
		chars_per_batch = data_size / batch_size
		iterators = range(0, total_used_chars, chars_per_batch)

		self.sequence_length = sequence_length
		self.batch_size = batch_size
		self.string = string
		self.vocabulary = vocabulary
		self.vocabulary_size = vocabulary_size
		self.data_size = data_size
		self.char_to_i = char_to_i
		self.i_to_char = i_to_char
		self.encoder = encoder
		self.decoder = decoder
		self.chars_per_batch = chars_per_batch
		self.total_used_chars = total_used_chars
		self.string = string
		self.chars_per_batch = chars_per_batch
		self.iterators = iterators

	def fetch_train(self):
		T = self.sequence_length
		batch_string = ''

		for i in range(len(self.iterators)):
			batch_string += self.string[self.iterators[i]:self.iterators[i] + T]
			self.iterators[i] += T

		if self.iterators[0] + T >= self.chars_per_batch:
			self.iterators = range(0, self.total_used_chars, self.chars_per_batch)
		
		batch_x = self.encoder(batch_string)
		batch_y = self.encoder(batch_string[1:] + batch_string[0])

		X = batch_x.reshape((self.batch_size, T, self.vocabulary_size), order='C').swapaxes(1, 2).swapaxes(0, 2)
		Y = batch_y.reshape((self.batch_size, T, self.vocabulary_size), order='C').swapaxes(1, 2).swapaxes(0, 2)
		
		return X, Y