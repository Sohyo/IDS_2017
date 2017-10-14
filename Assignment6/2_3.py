import sys
import nltk
from nltk.corpus import gutenberg
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('gutenberg')
titles = gutenberg.fileids()

corpus = []

for title in titles:
    corpus.append(gutenberg.raw(title))

vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
X = vectorizer.fit_transform(corpus)

XA = X.toarray()
# print vectorizer.vocabulary_
print 'The dimensions of the TF.IDF matrix: '
print XA.shape

print 'TF.IDF computation for the Gutenberg corpus is completed\n\n'

dict = vectorizer.vocabulary_
def where_was_that(phrase, corpus):
    phrase_lower = phrase.lower()
    words = phrase_lower.split()
    max_score = 0
    book_index = -1
    # for each book, so row of the TF.IDF matrix
    for i in range(0, XA.shape[0] - 1):
        score = 0
        # for each word in the search phrase
        for word in words:
            column_index = dict.get(word, -1)
            # if search word is not in the corpus
            if column_index == -1:
                continue
            score = score + XA[i][column_index]
        print score
        if score > max_score:
            max_score = score
            book_index = i

    print max_score
    print book_index
    print gutenberg.fileids()[book_index]



while True:
    try:
        testVar = raw_input("Input your search phrase: (press Ctrl+C to exit)\n")
        where_was_that(testVar, 'gutenberg')
    except KeyboardInterrupt:
        sys.exit()