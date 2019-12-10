import numpy as np
import re
import codecs
import tensorflow as tf


def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
	padding = [0] * 300
	word_dict = {}
	W = []
	i = 0
	maxnum = 44
	W.append(padding)
	
	nv = open("text/outvec.vec", "r", encoding='utf-8').read().split(' ')
	ii = 0
	
	while ii < len(nv):
		if len(nv[ii]) == 0:
			ii += 301
			continue
		si = nv[ii].strip()
		
		if si in word_dict:
			ii += 301
			continue
		else:
			i += 1
			word_dict[si] = i
			if len(nv[ii + 1:ii + 301])!=300:
				i-=1
				ii+=301
				continue
			W.append(list(map(float, nv[ii + 1:ii + 301])))
			ii += 301
	
	negative_examples = list(open(negative_data_file, "r", encoding='utf-8-sig').readlines())
	neg_result = []
	
	for s in negative_examples:
		s = s.strip()
		words = s.split(' ')
		ans = []
		if len(words) > maxnum:
			maxnum = len(words)
		
		for w in words:
			w = w.strip()
			ans.append(word_dict[w])
		# add padding
		padding_cnt = maxnum - len(ans)
		k = 0
		for k in range(padding_cnt):
			ans.append(0)
		neg_result.append(ans)
	
	positive_examples = list(open(positive_data_file, "r", encoding='utf-8-sig').readlines())
	
	i = 301
	pos_result = []
	for s in positive_examples:
		s = s.strip()
		words = s.split(' ')
		if len(words) > maxnum:
			maxnum = len(words)
		ans = []
		for w in words:
			w = w.strip()
			ans.append(word_dict[w])
		# add padding
		padding_cnt = maxnum - len(ans)
		k = 0
		for k in range(padding_cnt):
			ans.append(0)
		pos_result.append(ans)
	
	import random
	
	

	ps = sorted(random.sample(list(range(len(pos_result))),55), reverse= True)
	ns = sorted(random.sample(list(range(len(neg_result))),55), reverse =True)

	x_test_pos = []
	x_test_neg = []

	for i in ps:
		z = pos_result.pop(i)
		x_test_pos.append(z)

	for i in ns:
		z = neg_result.pop(i)
		x_test_neg.append(z)

	x_text = pos_result + neg_result*6
	print('---')
	print(len(pos_result),len(neg_result))
	x_test = x_test_pos + x_test_neg

	# Generate labels
	positive_labels = [[1, 0] for _ in x_test_pos]
	negative_labels = [[0, 1] for _ in x_test_neg]
	y_test = positive_labels + negative_labels
		
	# Generate labels
	positive_labels = [[1, 0] for _ in pos_result]
	negative_labels = [[0, 1] for _ in neg_result]
	y = positive_labels + negative_labels*6
	
	return [x_text, y, word_dict, W, x_test,y_test]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
	#print("\n\nbatch:", int((len(data)-1)/batch_size) + 1)
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