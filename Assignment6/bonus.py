import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import gutenberg
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from operator import itemgetter

titles = gutenberg.fileids() #Book Titles
d1 = {}
XA = np.array([])

def lookup(title,word):
    global d1
    column_index = d1.get(word,-1)
    if column_index == -1:
        return 0
    i = titles.index(title)
    return XA[i][column_index]

def summary(title):
    book = gutenberg.raw(title)
    sentences = sent_tokenize(book)
    position = 0
    sentences_IDF = []
    for sentence in sentences:
        words = sentence.lower().split()
        word_count = len(words)
        sent_score = 0
        position = position + 1
        for word in words:
            sent_score += lookup(title, word)
        sent_score = sent_score/float(word_count)
        sentences_IDF.append([sentence, sent_score, word_count, position])
    sentences_IDF.sort(key=itemgetter(1), reverse=True)
    truncated_list = get_final_sentences(sentences_IDF)
    text = ''
    for s in truncated_list:
        text = text + s[0]
    text = text.replace('\n', ' ').replace('\r', '')
    print text
    with open('./bonus/summary_' + title, "w") as text_file:
        text_file.write(text)

def get_final_sentences(sentences_IDF):
    total_words = 0
    truncated_list = []
    iterator = 0
    while total_words < 200:
        if sentences_IDF[iterator][2] > 7:
            total_words = total_words + sentences_IDF[iterator][2]
            # We don't want to go over the 200 word limit.
            if total_words > 200:
                break
            truncated_list.append(sentences_IDF[iterator])
        iterator = iterator + 1
    truncated_list.sort(key = itemgetter(3))
    return truncated_list


def tf_idf():
    nltk.download('gutenberg')
    titles = gutenberg.fileids()
    corpus = []
    for title in titles:
        corpus.append(gutenberg.raw(title))
    vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    global XA
    XA = X.toarray()
    global d1
    d1 = vectorizer.vocabulary_


tf_idf()
for title in titles:
    print(title)
    summary(title)
    print('\n\n\n\n')