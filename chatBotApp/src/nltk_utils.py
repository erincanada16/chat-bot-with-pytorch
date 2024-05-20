import nltk
import numpy as np
#package with a pretrained tokenizer for first time use
#nltk.download('punkt')
#different stemmers available -- try others
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

#tokenize
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

#stem and lower
def stem(word):
    return stemmer.stem(word.lower())

#bag of words
def bag_of_words(tokenized_sentence, all_words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """

    tokenized_sentence= [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
'''
a = "How long does shipping take"
print(a)

a = tokenize(a)
print(a)

words = ["organize","organizes","organizing"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)

sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bag = bag_of_words(sentence, words)
print(bag)
'''
