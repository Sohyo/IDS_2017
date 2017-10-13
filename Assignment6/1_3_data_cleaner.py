import nltk
from nltk.corpus import gutenberg
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from numpy import unique
import config as cfg
import csv
import time
import operator

#Run just once
#nltk.download('gutenberg')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

path = cfg.path

#Book Titles
titles = gutenberg.fileids()

#List of stopwords
stopwords = map(lambda x: x.encode('utf-8'),stopwords.words('english'))

##  TEXT PREPROCESING
# Lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

#Stemmer

stemmer = SnowballStemmer("english")

time_start = time.clock()

#Case Fold, Lemmatize, Punctuation, Stemmer , Stopwords
def cleaning(book):
    text = str(book).split('u')
    output = []
    punctuation = string.punctuation+'``'+'--'+"''"
    for item in book:
        #remove punctuation
        if item not in punctuation:
            #case fold -> lemmatize -> stemming -> tokenize
            word = word_tokenize(stemmer.stem(wordnet_lemmatizer.lemmatize(item.lower())))
            output.append(word[0].encode('utf-8'))
    return output

#TEST
for title in titles:
    book = gutenberg.words(title)
    cleaned = cleaning(book)
    print(title)
    print("Length before cleaning: \t" + str(len(book)) )
    print("Length after cleaning: \t" + str(len(cleaned)) )
