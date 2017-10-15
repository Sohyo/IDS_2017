import nltk
from nltk.corpus import gutenberg
import string
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from numpy import unique
import config as cfg
import csv
import time
import operator
import pandas as pd


time_start = time.clock()


# Run just once
# nltk.download('gutenberg')
# nltk.download('punkt')
# nltk.download('wordnet')

path = cfg.path
folder1 = 'term_frequency1/' #Create this folder if it doesn't exist
folder2 = 'term_frequency2/' #Create this folder if it doesn't exist

titles = gutenberg.fileids() #Book Titles
##  TEXT PREPROCESING
# Lemmatizer

wordnet_lemmatizer = WordNetLemmatizer()
#List of stopwords

stopwords = stopwords.words('english')
punctuation = string.punctuation+'``'+'--'+"''"



## Preprocessing A: remove stopwords and normalize frequency/doc length for ex. 1.5.
## ================================================================================================


# time_start = time.clock()
# #Case Fold, Lemmatize, Punctuation, Stopwords

# def cleaning(book):
#     output = []
#     for item in book:
#         #remove punctuation
#         #remove stopwords
#         if item not in punctuation and item not in stopwords and len(item)>1:
#         # if item not in punctuation and item not in stopwords and len(item)>1:
#             #there are also combined punctuation as "!--"
#             if any(punct in punctuation for punct in item) == 0:
#                 #case fold -> lemmatize -> tokenize
#                 word = word_tokenize(wordnet_lemmatizer.lemmatize(item.lower()))
#                 output.append(word[0])
#     return output

# #Term Frequency function

# def term_frequency(text):
#     doc_length = float(len(text))
#     unique_terms = sorted(unique(text))
#     dict0 = {}
#     for word in unique_terms:
#         term_count = text.count(word)
#         dict0[word] = term_count/doc_length #relative freq.
#     return dict0


# #Main problem count the term frequency of each book


# for title in titles:
#     dict1 = {}
#     book = gutenberg.words(title)
#     cleaned = cleaning(book)
#     dict1[title] = term_frequency(cleaned)
#     dict1 = sorted(dict1[title].items(), key=lambda x: x[1], reverse=True)
#     ddf = pd.DataFrame(dict1)
#     ddf.to_csv(path+folder1+title+'.csv',index=False)
#     print("Done : \t" + title)

# time_elapsed = (time.clock() - time_start)
# print(time_elapsed)





## Preprocessing B: keep stopwords don't normalize for ex 1.2.
## ================================================================================================

time_start = time.clock()

#Case Fold, Lemmatize, Punctuation , Stopwords

def cleaning(book):
    output = []
    for item in book:
        #remove punctuation
        #remove stopwords
        if item not in punctuation and len(item)>1:
        # if item not in punctuation and item not in stopwords and len(item)>1:
            #there are also combined punctuation as "!--"
            if any(punct in punctuation for punct in item) == 0:
                #case fold -> lemmatize -> tokenize
                word = word_tokenize(wordnet_lemmatizer.lemmatize(item.lower()))
                output.append(word[0])
    return output

#Term Frequency function
def term_frequency(text):
    doc_length = float(len(text))
    unique_terms = sorted(unique(text))
    dict0 = {}
    for word in unique_terms:
        term_count = text.count(word)
        dict0[word] = term_count
        # dict[word] = term_count/doc_length #relative freq.
    return dict0


#Main problem count the term frequency of each book
dict2 = {}

for title in titles:
    dict2 = {}
    book = gutenberg.words(title)
    cleaned = cleaning(book)
    dict2[title] = term_frequency(cleaned)
    dict2 = sorted(dict2[title].items(), key=lambda x: x[1], reverse=True)
    ddf = pd.DataFrame(dict2)
    ddf.to_csv(path+folder2+title+'.csv',index=False)
    print("Done : \t" + title)

time_elapsed = (time.clock() - time_start)
print(time_elapsed)
