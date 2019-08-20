#from music21 import *

#littleMelody = converter.parse("tinynotation: 3/4 c4 d8 f g16 a g f#")
#littleMelody.show()

import os
import json
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

data_folder = "./data"

def train_model(data, epochs=1, cuda=False):
	char_to_index = {ch: i for (i, ch) in enumerate(sorted(list(set(data))))}
	print("Unique char: ",len(char_to_index))

	#with open(charIndex, mode='w') as f:
	#	json.dump(char_to_index, f)

	index_to_char = {i: ch for (ch, i) in char_to_index.items()}
	unique_chars = len(char_to_index)

	encoded = np.array([char_to_index[char] for char in data])
	print(len(encoded))

	net = CharLSTM(sequence_len=64, vocab_size=unique_chars, hidden_size=512, batch_size=32, char2int=char_to_index, int2char=index_to_char)
	optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
	criterion = nn.CrossEntropyLoss()

	val_idx = int(len(encoded) * (1-0.1))
	data, val_data = encoded[:val_idx], encoded[val_idx:]

	if cuda:
		net.cuda()
	val_losses = list()
	samples = list()
	counter = 0
	for epoch in range(epochs):
		hc =  net.init_hidden(net.batch_size)
		for x,y in get_batches(data, 32, 64):
			counter+=1
			x = one_hot_encode(x, unique_chars)
			x, y = torch.from_numpy(x), torch.from_numpy(y)
			if cuda:
				x, y = x.cuda(), y.cuda()
			optimizer.zero_grad()
			output, val_h = net(x, hc)
			loss = criterion(output, y.view(32*64))
			loss.backward()
			optimizer.step()

			if counter%100 == 0:
				val_h =  net.init_hidden(net.batch_size)
				for val_x, val_y in get_batches(val_data, 32, 64):
					x = one_hot_encode(val_x, unique_chars)
					x, y = torch.from_numpy(x), torch.from_numpy(val_y)
					if cuda:
						x, y = x.cuda(), y.cuda()
					val_output, val_h = net.forward(x, val_h)
					val_loss = criterion(val_output, y.view(32*64))
					val_losses.append(val_loss.data)
				print("Epoch: {}, Batch: {}, Train Loss: {:.6f}, Validation Loss: {:.6f}".format(epoch, counter, loss.data, val_loss.data))
		#if epoch%10==0:
	return net
def get_batches(array, n_seq_in_a_batch, n_char):
	batch_size = n_seq_in_a_batch * n_char
	n_batches = len(array)//batch_size

	array = array[:n_batches * batch_size]
	array = array.reshape((n_seq_in_a_batch, -1))

	for n in range(0, array.shape[1], n_char):
		x = array[:, n:n+n_char]
		y = np.zeros_like(x)
		try:
			y[:,:-1], y[:,-1] = x[:, 1:], array[:, n+n_char]
		except IndexError:
			y[:, :-1], y[:, -1] = x[:, 1:], array[:, 0]
		yield x, y

def one_hot_encode(arr, n_labels):
	one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
	one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
	one_hot = one_hot.reshape((*arr.shape, n_labels))
	return one_hot

class CharLSTM(nn.ModuleList):
	def __init__(self, sequence_len, vocab_size, hidden_size, batch_size, char2int, int2char):
		super().__init__()
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.sequence_len = sequence_len
		self.vocab_size = vocab_size
		self.n_layers = 2
		self.drop_prob = 0.1
		self.char2int = char2int
		self.int2char = int2char
		#self.lstm_1 = nn.LSTMCell(input_size=vocab_size, hidden_size=hidden_size)
		#self.lstm_2 = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
		self.dropout = nn.Dropout(self.drop_prob)
		self.lstm = nn.LSTM(self.vocab_size, hidden_size, 2, dropout=self.drop_prob, batch_first=True)
		self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)
		self.init_weights()

	def init_weights(self):
		initrange = 0.1
		self.fc.bias.data.fill_(0)
		self.fc.weight.data.uniform_(-1, 1)

	def forward(self, x, hc):
		#output_seq = torch.empty((self.sequence_len, self.batch_size, self.vocab_size))
		x, (h, c) = self.lstm(x, hc)
		x = self.dropout(x)

		x = x.view(x.size()[0]*x.size()[1], self.hidden_size)
		x = self.fc(x)

		#hc1, hc2 = hc, hc
		'''
		for t in range(self.sequence_len):

			hc1 = self.lstm_1(x[t], hc1)
			h1, c1 = hc1
			hc2 = self.lstm_2(h1, hc2)
			h2, c2 = hc2

			output_seq[t] = self.fc(self.dropout(h2))
		'''
		#return output_seq.view((self.sequence_len * self.batch_size, -1))
		return x, (h, c)

	def init_hidden(self, n_seqs):
		weight = next(self.parameters()).data
		return (Variable(weight.new(self.n_layers, n_seqs, self.hidden_size).zero_()),
			Variable(weight.new(self.n_layers, n_seqs, self.hidden_size).zero_()))

	def predict(self, char, h=None, cuda=False, top_k=None):
		if cuda:
			self.cuda()       
		if h is None:
			h = self.init_hidden(1)
		x = np.array([[self.char2int[char]]])
		x = one_hot_encode(x, self.vocab_size)
		inputs = Variable(torch.from_numpy(x), volatile=True)
		if cuda:
			inputs = inputs.cuda()
		h = tuple([Variable(each.data, volatile=True) for each in h])
		out, h = self.forward(inputs, h)

		p = F.softmax(out).data
		if cuda:
			p = p.cpu()
		if top_k is None:
			top_ch = np.arange(self.vocab_size)
		else:
			p, top_ch = p.topk(top_k)
			top_ch = top_ch.numpy().squeeze()
		p = p.numpy().squeeze()
		char = np.random.choice(top_ch, p=p/p.sum())
		return self.int2char[char], h

def sample(model, size, prime='X', top_k=None, cuda=False):
    if cuda:
    	model.cuda()

    model.eval()
    chars = [ch for ch in prime]
    h = model.init_hidden(1)
    for ch in prime:
        char, h = model.predict(ch, h, cuda=cuda, top_k=top_k)

    chars.append(char)

    for ii in range(size):
        char, h = model.predict(chars[-1], h, cuda=cuda, top_k=top_k)
        chars.append(char)

    return ''.join(chars)
if __name__ == '__main__':
	data = ''

	for r, d, f in os.walk(data_folder):
		for file in f:
			open_file = open(os.path.join(r, file))
			data += open_file.read()
			open_file.close()
		data = data + "\n\n\n"

	model = train_model(data, cuda=True)
	for i in range(10):
		print("---------------")
		print("No. ",i+1)
		print("---------------")
		print(sample(model, 200, cuda=True))
	