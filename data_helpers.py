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
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    
    pv = open("pos_vec.txt", "r", encoding='utf-8').read().split(' ')
    padding= [0]*300
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    word_dict={}
    W=[]
    i=301
    idx=0
    W.append(padding)
    pos_result=[]
    for s in positive_examples:
        s=s.strip()
        words = s.split(' ')
        ans = []
        for w in words:
            w= w.strip()
            if len(w)==0:
                continue
            if w==pv[i].strip():
                if w in word_dict:

                    ans.append(word_dict[w])
                else:
                    idx+=1
                    word_dict[w]=idx
                    int_list = list(map(float, pv[i+1 :i + 301]))
                    W.append(int_list)
                    ans.append(idx)
            i+=301
        # add padding
        padding_cnt = 38-len(ans)
        for j in range(padding_cnt):
            ans.append(0)
        pos_result.append(ans)

    nv = open("neg_vec.txt", "r", encoding='utf-8').read().split(' ')
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    neg_result=[]
    i = 0
    for s in negative_examples:
        s=s.strip()
        words = s.split(' ')
        ans = []

        for w in words:
            w= w.strip()
            if len(w)==0:
                continue
            if w == nv[i].strip():
                if w in word_dict:
                    ans.append(word_dict[w])
                else:
                    idx+=1
                    word_dict[w]=idx
                    #W.append(np.array(nv[i+1 :i + 301], dtype='f'))
                    int_list = list(map(float, nv[i+1 :i + 301]))
                    W.append(int_list)
                    ans.append(idx)
            i+=301
        # add padding
        padding_cnt = 38-len(ans)
        k=0
        for k in range(padding_cnt):
            ans.append(0)
        neg_result.append(ans)

    x_text = pos_result + neg_result


    # Generate labels
    positive_labels = [[0, 1] for _ in pos_result]
    negative_labels = [[1, 0] for _ in neg_result]
    y = positive_labels + negative_labels

    return [x_text, y, word_dict, W]


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