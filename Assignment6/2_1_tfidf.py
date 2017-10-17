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
print XA.shape
print X.toarray()
