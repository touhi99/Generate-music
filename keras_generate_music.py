import os 
from keras import backend as K, layers
import numpy as np
from keras.callbacks import LambdaCallback
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM, Input, Bidirectional
from keras.layers import Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import LambdaCallback
import random
import sys
import io
data_folder = "./mdata"
saved_model = "saved_model.h5"
"""
From Keras Tutorial Page
"""

class MusicGenerate():
	def __init__(self):
		self.char_to_index = {}
		self.maxlen = 40
		self.data = data

	def on_epoch_end(self, epoch, _):
	    # Function invoked at end of each epoch. Prints generated text.
	    print()
	    print('----- Generating text after Epoch: %d' % epoch)

	    start_index = random.randint(0, len(self.char_to_index) - self.maxlen - 1)
	    for diversity in [0.2, 0.5, 1.0, 1.2]:
	        print('----- diversity:', diversity)

	        generated = ''
	        pattern = self.data[start_index: start_index + self.maxlen]
	        #pattern = ''
	        generated += pattern
	        #generated = 'X:'
	        print('----- Generating with seed: "' + pattern + '"')
	        sys.stdout.write(generated)

	        for i in range(400):
	            x_pred = np.zeros((1, self.maxlen, len(self.char_to_index)))
	            for t, char in enumerate(pattern):
	                x_pred[0, t, self.char_to_index[char]] = 1.

	            preds = self.model.predict(x_pred, verbose=0)[0]
	            next_index = self.sample(preds, diversity)
	            next_char = self.index_to_char[next_index]

	            generated += next_char
	            pattern = pattern[1:] + next_char

	            sys.stdout.write(next_char)
	            sys.stdout.flush()
	        print()

	def sample(self, preds, temperature=1.0):
	    # helper function to sample an index from a probability array
	    preds = np.asarray(preds).astype('float64')
	    preds = np.log(preds) / temperature
	    exp_preds = np.exp(preds)
	    preds = exp_preds / np.sum(exp_preds)
	    probas = np.random.multinomial(1, preds, 1)
	    return np.argmax(probas)

	def t_model(self, data, epochs=1):
		self.data = data
		self.char_to_index = {ch: i for (i, ch) in enumerate(sorted(list(set(data))))}
		print("Unique char: ",len(self.char_to_index))

		self.index_to_char = {i: ch for (ch, i) in self.char_to_index.items()}
		unique_chars = len(self.char_to_index)

		encoded = np.array([self.char_to_index[char] for char in data])
		print(len(encoded))

		# cut the sequences in semi-redundant sequences of maxlen characters
		step = 3
		patterns = []
		next_chars = []
		for i in range(0, len(encoded) - self.maxlen, step):
		    patterns.append(encoded[i: i + self.maxlen])
		    next_chars.append(encoded[i + self.maxlen])
		print('nb sequences:', len(patterns))

		#print('Vectorization...')
		x = np.zeros((len(patterns), self.maxlen, len(self.char_to_index)), dtype=np.bool)
		y = np.zeros((len(patterns), len(self.char_to_index)), dtype=np.bool)
		for i, pattern in enumerate(patterns):
		    for t, char in enumerate(pattern):
		        x[i, t, char] = 1
		    y[i, next_chars[i]] = 1

		print('Build model...')
		self.model = Sequential()
		self.model.add(LSTM(256, input_shape=(self.maxlen, len(self.char_to_index))))
		self.model.add(Dense(len(self.char_to_index), activation='softmax'))
		self.model.add(Dropout(0.2))
		optimizer = Adam(lr=0.001)
		if os.path.exists(saved_model):
			self.model = load_model(saved_model)
			self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
			self.on_epoch_end()
		else:
			self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
			print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
			self.model.fit(x, y,
		          batch_size=128,
		          epochs=1,
		          callbacks=[print_callback])
			self.model.save(saved_model)



if __name__ == '__main__':
	data = ''

	for r, d, f in os.walk(data_folder):
		for file in f:
			open_file = open(os.path.join(r, file))
			data += open_file.read()
			open_file.close()
		data = data + "\n\n\n"

	MusicGenerate().t_model(data)