import numpy as np
import re
import codecs
import tensorflow as tf



def load_data_and_labels(positive_data_file, negative_data_file):
	"""
	Loads MR polarity data from files, splits the data into words and generates labels.
	Returns split sentences and labels.
	"""
	# Load data from files
	
	padding = [0] * 300
	word_dict = {}
	W = []
	i = 0
	maxnum = 46
	W.append(padding)
	
	nv = open("neg_k_clear_kfast.txt", "r", encoding='utf-8').read().split(' ')
	ii=0
	
	while ii < len(nv):
		if len(nv[ii])==0:
			ii +=301
			continue
		si=nv[ii].strip()
		if len(si)> maxnum:
			maxnum = len(si)
		if si in word_dict:
			ii+=301
			continue
		else:
			i+=1
			word_dict[si]=i
			W.append(list(map(float, nv[ii + 1:ii + 301])))
			ii+=301
		
	pv = open("pos_k_clear_kfast.txt", "r", encoding='utf-8').read().split(' ')
	ii = 0
	while ii < len(pv):
		if len(pv[ii]) == 0:
			ii += 301
			continue
		si = pv[ii].strip()
		if len(si)> maxnum:
			maxnum = len(si)
		if si in word_dict:
			ii += 301
			continue
		else:
			i += 1
			word_dict[si] = i
			W.append(list(map(float, pv[ii + 1:ii + 301])))
			ii += 301
		
	negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
	neg_result = []

	for s in negative_examples:
		s = s.strip()
		words = s.split(' ')
		ans = []
		
		for w in words:
			w = w.strip()
			ans.append(word_dict[w])
		# add padding
		padding_cnt = 46 - len(ans)
		k = 0
		for k in range(padding_cnt):
			ans.append(0)
		neg_result.append(ans)
		
	positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
	
	i=301
	pos_result = []
	for s in positive_examples:
		s = s.strip()
		words = s.split(' ')
		ans = []
		for w in words:
			w = w.strip()
			ans.append(word_dict[w])
		# add padding
		padding_cnt = 46 - len(ans)
		k = 0
		for k in range(padding_cnt):
			ans.append(0)
		pos_result.append(ans)
	
	
	
	x_text = pos_result + neg_result *1
	
	
	# Generate labels
	positive_labels = [[0, 1] for _ in pos_result]
	negative_labels = [[1, 0] for _ in neg_result]
	y = positive_labels + negative_labels * 1
	
	return [x_text, y, word_dict, W]
	#return [pos_result, neg_result, positive_labels, negative_labels, word_dict, W]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
	# print("\n\nbatch:", int((len(data)-1)/batch_size) + 1)
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]