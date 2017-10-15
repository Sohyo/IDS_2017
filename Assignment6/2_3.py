import sys
import nltk
from nltk.corpus import gutenberg
from nltk.corpus import state_union
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('state_union')
nltk.download('gutenberg')

#Sys handling

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

phrase = sys.argv[1]
corpora = sys.argv[2]
corpus = []

#Check corpus
if corpora == "gutenberg":
    titles = gutenberg.fileids()
    for title in titles:
        corpus.append(gutenberg.raw(title))

elif corpora == "state_union":
    titles = state_union.fileids()
    for title in titles:
        corpus.append(state_union.raw(title))
else:
    print "Choose from gutenberg or state_union"
    exit(0)


vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
X = vectorizer.fit_transform(corpus)

XA = X.toarray()
# print vectorizer.vocabulary_
print 'The dimensions of the TF.IDF matrix: '
print XA.shape

print 'TF.IDF computation for the '+ corpora +' corpus is completed\n'

dict = vectorizer.vocabulary_

phrase_lower = phrase.lower()
words = phrase_lower.split()
max_score = 0
book_index = -1
others = []
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
    if score > max_score:
        max_score = score
        book_index = i
    if corpora == "gutenberg":
        others.append((gutenberg.fileids()[i],score))
    else:
        others.append((state_union.fileids()[i], score))

sorted_books = sorted(others, key=lambda x: x[1], reverse=True)

print "Score:"
print max_score
print
print "Title"
if corpora == "gutenberg":
    print gutenberg.fileids()[book_index]
else:
    print state_union.fileids()[book_index]
print
print "Other Suggested Books:"
print
for book in sorted_books[1:4]:
    if book[1] > 0:
        print(book)
    print