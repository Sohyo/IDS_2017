import nltk
from nltk.corpus import gutenberg
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from numpy import unique
import config as cfg
import csv
import time
import operator


time_start = time.clock()
#run your code

#Run just once
#nltk.download('gutenberg')
#nltk.download('punkt')
#nltk.download('wordnet')

path = cfg.path

#Book Titles
titles = gutenberg.fileids()
##  TEXT PREPROCESING
# Lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

#Tokenizer
def tokenizer(book):
    text = ''
    for word in book:
        text = text + ' ' + word
    tokens = word_tokenize(text)
    return tokens

#Case Fold, Lemmatize, Punctuation
def cleaning(tokens):
    output = []
    punctuation = string.punctuation+'``'+'--'+"''"
    for item in tokens:
        #case fold
        word = item.lower()
        #lemmatize
        lemm = wordnet_lemmatizer.lemmatize(word)
        #remove punctuation
        if lemm not in punctuation:
            output.append(lemm.encode('utf-8'))
    return output

#Term Frequency function
def term_frequency(text):
    doc_length = float(len(text))
    unique_terms = sorted(unique(text))
    dict = {}
    for word in unique_terms:
        term_count = text.count(word)
        dict[word] = term_count/doc_length
    return dict


#Main problem count the term frequency of each book
dict2 = {}
for title in titles:
    book = gutenberg.words(title)
    tokens = tokenizer(book)
    cleaned = cleaning(tokens)
    dict2[title] = term_frequency(cleaned)
    with open(path+title+'.csv', 'wb') as f:
        w = csv.writer(f)
        w.writerows(sorted(dict2[title].items(), key=lambda x: x[1], reverse=True))
    print("Done : \t" + title)

time_elapsed = (time.clock() - time_start)
print(time_elapsed)