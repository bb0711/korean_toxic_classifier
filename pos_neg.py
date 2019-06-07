# -*- coding: utf-8 -*-
import sys
import codecs
import hgtk

EMPTY_JS_CHAR = "e"


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


# UTF-8을 EUC-KR로 변환
def utf2euc(str):
	return unicode(str, 'utf-8').encode('euc-kr')


# EUC-KR을 UTF-8로 변환
def euc2utf(str):
	return unicode(str, 'euc-kr').encode('utf-8')

def main():
	INPUT_FILE_PATH = sys.argv[1]
	OUTPUT_FILE_PATH1 = sys.argv[2]
	OUTPUT_FILE_PATH2 = sys.argv[3]
	with codecs.open(OUTPUT_FILE_PATH1, 'w', encoding='utf8') as pos:
		with codecs.open(OUTPUT_FILE_PATH2, 'w', encoding='utf8') as neg:
			with codecs.open(INPUT_FILE_PATH, 'r', encoding="utf8" ,errors='strict') as input_:
				for sentence in input_:
					arr = sentence.split(",")
					sentence = arr[1]
					result = jamo_split(sentence.strip())
					#result = result.encode('euc-kr')
					#jamo_sentence = jamo_sentence.encode('utf-8')
					#result = euc2utf(jamo_sentence)
					print(type(result),result)
					
					count =0
					
					if arr[2]=='toxic':
						count +=1
					if arr[3] == 'toxic':
						count += 1
					if arr[4] == 'toxic':
						count += 1
						
					if count >=1:
						neg.write(result+'\r\n')
					else:
						pos.write(result + '\r\n')
					#sentence = sentence.replace(u'\xa0', ' ')


if __name__ == '__main__':
	main()