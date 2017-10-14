import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import gutenberg
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import operator

titles = gutenberg.fileids() #Book Titles
d1 = {}
sentences_IDF = []

def lookup(title,word):
	global d1
	column_index = d1.get(word,-1)
	if column_index == -1:
		return 0
	i = titles.index(title)
	return XA[i][column_index]

# word = 'loves'
# titles = 'austen-emma.txt'
# column_index = d1.get(word,-1)
# i = title.index(title)
# return XA[i][column_index]


def summary(title):
	# for title in titles:
	# 	print(title)
	# title = raw_input('Please select a title: ')
	book = gutenberg.raw(title)
	sentences = sent_tokenize(book)
	for sentence in sentences:
		words = sentence.lower().split()
		sent_score = 0
		for word in words:
			sent_score += lookup(title, word)
		sentences_IDF.append([sent_score,sentence])
	print(sentences_IDF[:10])
	sorted_sentences_IDF = sentences_IDF.sort(key=operator.itemgetter(0))

def tf_idf():
	# nltk.download('gutenberg')
	titles = gutenberg.fileids()
	corpus = []
	for title in titles:
		corpus.append(gutenberg.raw(title))
	vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
	X = vectorizer.fit_transform(corpus)
	XA = X.toarray()
	global d1
	d1 = vectorizer.vocabulary_
	return XA



XA = tf_idf()
for title in titles:
	summary(title)





# dict = vectorizer.vocabulary_
# def where_was_that(phrase, corpus):
#     phrase_lower = phrase.lower()
#     words = phrase_lower.split()
#     max_score = 0
#     book_index = -1
#     # for each book, so row of the TF.IDF matrix
#     for i in range(0, XA.shape[0] - 1):
#         score = 0
#         # for each word in the search phrase
#         for word in words:
#             column_index = dict.get(word, -1)
#             # if search word is not in the corpus
#             if column_index == -1:
#                 continue
#             score = score + XA[i][column_index]
#         print score
#         if score > max_score:
#             max_score = score
#             book_index = i