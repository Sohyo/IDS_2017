import nltk
from nltk.corpus import gutenberg
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from numpy import unique
import config as cfg
import csv
import operator
from nltk.tag import pos_tag
from nltk.collocations import *
import pandas as pd


#Run just once
#nltk.download('gutenberg')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

path = cfg.path
wnl = WordNetLemmatizer()
titles = gutenberg.fileids()
stopwords = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()
# punctuation = string.punctuation+'``'+'--'+"''"
tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("english")



#Case Fold, Lemmatize, Punctuation, Stemmer , Stopwords

'''
You need to keep proper nouns capitalized. if != NNP: lowercase 
then run collocation. 
Then exclude collocations that have more than 1 capital word.
'''


def cleaning_2NNP(book):
	output = [] # text = str(book).split('u')
	for item in book:
		tag = nltk.tag.pos_tag([item])[0][1]
		if tag == 'NNP':
			word = word_tokenize(item)
			output.append(word[0])
		elif item.lower() not in stopwords and len(item)>1:
			word = tokenizer.tokenize(wordnet_lemmatizer.lemmatize(item.lower())) #case fold -> stemming -> tokenize
			if word:
				output.append(word[0])
	return output


trigram_measures = nltk.collocations.TrigramAssocMeasures()
titles = gutenberg.fileids()

d1 = {}
for title in titles:
	book = gutenberg.words(title)
	cleaned = cleaning_2NNP(book)
	cleaned = [n.replace("_", "") for n in cleaned]
	finder = TrigramCollocationFinder.from_words(cleaned)
	# finder.apply_freq_filter(3)
	colloc = finder.nbest(trigram_measures.pmi, 100)
	new_colloc = []
	for col in colloc:
		if len(list(filter(None, col)))<3:
			continue 
		else:
			count = col[0][0].isupper()+col[1][0].isupper()+col[2][0].isupper()
			if count<2:
				new_colloc.append(col)
			else: 
				print(col)
	d1[title] = new_colloc[:10]
	print('Done with '+str(title))

df = pd.DataFrame(d1)




# Arbitrary amount of ngrams:
# from nltk import ngrams
# import time

# time_start = time.clock()

# sentence = 'introduction to data science is fun'
# n = 5
# amount_grams = ngrams(sentence.split(), n)

# for gram in amount_grams:
#     print(gram)

# time_elapsed = (time.clock() - time_start)
# print(time_elapsed)