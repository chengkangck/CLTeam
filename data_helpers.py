import numpy as np
import re
import itertools
from collections import Counter

emotion_label = {'sad':[1,0,0,0,0,0], 'joy':[0,1,0,0,0,0], 'disgust':[0,0,1,0,0,0] , 'surprise':[0,0,0,1,0,0], 'anger':[0,0,0,0,1,0], 'fear':[0,0,0,0,0,1]}
gold_label = {'sad':0, 'joy':1, 'disgust':2 , 'surprise':3, 'anger':4, 'fear':5}

def clean_str(string):
	regex_substr = [
		r'<[^>]+>', # HTML tags
		r'(?:@[\w_]+)', # @personName
		r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
		r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
		r'NEWLINE' #the special word 'NEWLINE'
	]
	__del_re = re.compile(r'('+'|'.join(regex_substr)+')', re.VERBOSE | re.IGNORECASE)
	__hash_re = re.compile(r'(?:\#+)([\w_]+[\w\'_\-]*[\w_]+)') 
	string = __del_re.sub(r'', string) #delete HTMLtags, @personName, URLs, numbers and 'NEWLINE'
	string = __hash_re.sub(r'\1', string)#delete hashtag but leaving the word after hashtag
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


def load_data_and_labels(train_file, gold_labels=None):
	instances = list(open(train_file, "r", encoding='utf-8').readlines())
	instances = [s.strip().split('\t') for s in instances]
	x_text = [clean_str(s[1]) for s in instances]
	if (gold_labels == None) :
		y = [emotion_label[s[0]] for s in instances]
	else:
		y = list(open(gold_labels, "r", encoding='utf-8').readlines())
		y = [gold_label[s.strip()] for s in y]
	return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
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
