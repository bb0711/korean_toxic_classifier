# -*- coding: utf-8 -*-
import sys
import codecs
import hgtk
import copy

import re
import tensorflow as tf

EMPTY_JS_CHAR = "e"

def clean_str(string):
	"""
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
	string = re.sub(r"\"+", " ", string)
	string = re.sub(r"\'+", " ", string)
	string = re.sub(r",+", ",", string)
	string = re.sub(r"\.+", ".", string)
	string = re.sub(r"\?+", "?", string)
	string = re.sub(r"!+", "!", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " ( ", string)
	string = re.sub(r"\)", " ) ", string)
	string = re.sub(r"\?", " ? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()

def jamo_split(sentence):
	result = []
	for word in sentence.split(' '):
		decomposed_word = ""
		for char in word:
			try:
				cho_joong_jong = hgtk.letter.decompose(char)
				char_seq = ""
				for cvc in cho_joong_jong:
					if cvc == '':
						cvc = EMPTY_JS_CHAR
					char_seq += cvc
				decomposed_word += char_seq
			except hgtk.exception.NotHangulException:
				decomposed_word += char
				continue
		result.append(decomposed_word)
	return " ".join(result)


def main():
	INPUT_FILE_PATH = sys.argv[1]
	OUTPUT_FILE_PATH = sys.argv[2]
	with codecs.open(OUTPUT_FILE_PATH, 'w', encoding='utf8') as jamo:
		with codecs.open(INPUT_FILE_PATH, 'r', encoding="utf8" ,errors='ignore') as input_:
			for sentence in input_:
				sentence = sentence.strip()
				#sentence = sentence.split(",")[1]
				#sentence = sentence.replace(u'\xa0', ' ')
				#sentence = sentence.strip()
				#sentence = clean_str(sentence)
				jamo_sentences = jamo_split(sentence)
				#jamo.write(sentence + '\r\n')
				jamo.write(jamo_sentences + '\r\n')


if __name__ == '__main__':
	main()
