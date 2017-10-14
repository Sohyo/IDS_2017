import nltk
from nltk.corpus import gutenberg
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from numpy import unique
import config as cfg
import csv
import time
import operator


time_start = time.clock()


#Run just once
#nltk.download('gutenberg')
#nltk.download('punkt')
#nltk.download('wordnet')

path = cfg.path
titles = gutenberg.fileids() #Book Titles
##  TEXT PREPROCESING
# Lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
#List of stopwords
stopwords = stopwords.words('english')
punctuation = string.punctuation+'``'+'--'+"''"
#Stemmer
stemmer = SnowballStemmer("english")


## Preprocessing A: remove stopwords and normalize frequency/doc length for ex. 1.5.
## ================================================================================================


time_start = time.clock()
#Case Fold, Lemmatize, Punctuation, Stemmer , Stopwords
def cleaning(book):
    output = []
    for item in book:
        #remove punctuation
        #remove stopwords
        if item not in punctuation and not in stopwords and len(item)>1:
        # if item not in punctuation and item not in stopwords and len(item)>1:
            #there are also combined punctuation as "!--"
            if any(punct in punctuation for punct in item) == 0:
                #case fold -> lemmatize -> stemming -> tokenize
                word = word_tokenize(stemmer.stem(wordnet_lemmatizer.lemmatize(item.lower())))
                output.append(word[0].encode('utf-8'))
    return output

#Term Frequency function
def term_frequency(text):
    doc_length = float(len(text))
    unique_terms = sorted(unique(text))
    dict = {}
    for word in unique_terms:
        term_count = text.count(word)
        dict[word] = term_count/doc_length #relative freq.
    return dict


#Main problem count the term frequency of each book
dict2 = {}
for title in titles:
    book = gutenberg.words(title)
    cleaned = cleaning(book)
    dict2[title] = term_frequency(cleaned)
    with open(path+title+'.csv', 'wb') as f:
        w = csv.writer(f)
        w.writerows(sorted(dict2[title].items(), key=lambda x: x[1], reverse=True))
    print("Done : \t" + title)

time_elapsed = (time.clock() - time_start)
print(time_elapsed)





## Preprocessing B: keep stopwords don't normalize for ex 1.2.
## ================================================================================================
path = '/Users/danielmlow/Dropbox/lct/data_science/team-07/Assignment6/term_frequency2/'

time_start = time.clock()
#Case Fold, Lemmatize, Punctuation, Stemmer , Stopwords
def cleaning(book):
    output = []
    for item in book:
        #remove punctuation
        #remove stopwords
        if item not in punctuation and len(item)>1:
        # if item not in punctuation and item not in stopwords and len(item)>1:
            #there are also combined punctuation as "!--"
            if any(punct in punctuation for punct in item) == 0:
                #case fold -> lemmatize -> stemming -> tokenize
                word = word_tokenize(stemmer.stem(wordnet_lemmatizer.lemmatize(item.lower())))
                output.append(word[0].encode('utf-8'))
    return output

#Term Frequency function
def term_frequency(text):
    doc_length = float(len(text))
    unique_terms = sorted(unique(text))
    dict = {}
    for word in unique_terms:
        term_count = text.count(word)
        dict[word] = term_count
        # dict[word] = term_count/doc_length #relative freq.
    return dict


#Main problem count the term frequency of each book
dict2 = {}

for title in titles:
    book = gutenberg.words(title)
    cleaned = cleaning(book)
    dict2[title] = term_frequency(cleaned)
    with open(path+title+'.csv', 'wb') as f:
        w = csv.writer(f)
        w.writerows(sorted(dict2[title].items(), key=lambda x: x[1], reverse=True))
    print("Done : \t" + title)

time_elapsed = (time.clock() - time_start)
print(time_elapsed)