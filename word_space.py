import codecs
import sys
import re
# input : after_clear_str.txt
def main():
    word_d=[]
    INPUT_FILE_PATH = sys.argv[1]
    OUTPUT_FILE_PATH = sys.argv[2]
    with codecs.open(OUTPUT_FILE_PATH, 'w', encoding='utf8') as output:
        with codecs.open(INPUT_FILE_PATH, 'r', encoding="utf8" ,errors='ignore') as input_:
            for sentence in input_:
                sentence = sentence.strip()
                for word in sentence.split(' '):
                    word= word.strip()
                    if not word in word_d:
                        check = re.match(r"^[A-Za-z?!]+", word)
                        # put korean word only with number
                        if(check == None):
                            if len(word)<6 and len(word)>=3:
                                word_d.append(word)

            #print(word_d)
        with codecs.open(INPUT_FILE_PATH, 'r', encoding="utf8" ,errors='ignore') as input_:
            for sentence in input_:
                sentence = sentence.strip()
                new=[]
                #print(sentence)

                for word in sentence.split(' '):
                    for key in word_d:
                        word= word.strip()
                        i = word.find(key)
                        if i > 0 and word!=key:
                            word = word[:i] + ' ' + word[i:]

                    word = re.sub(r"\s{2,}", " ", word)
                    new.append(word)
                sen = " ".join(new)
                output.write(sen + '\r\n')


if __name__ == '__main__':
    main()