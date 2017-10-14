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

# The 3 texts chosen are: carroll-alice, melville-moby_dick, shakespeare_hamlet
for row in XA[[7, 12, 15],]:
    n = 10
    top_n_tfidf = (row.argsort()[-n:][::-1])
    for idx in top_n_tfidf:
        print (vectorizer.get_feature_names()[idx] + "> " + str(row[idx]))
    print ()