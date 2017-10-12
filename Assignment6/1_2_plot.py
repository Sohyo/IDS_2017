from pylab import *
import matplotlib.pyplot  as plt

import config as cfg
import pandas

path = './term_frequency/'
matplotlib.style.use('ggplot')

df = pandas.read_csv(path + 'austen-emma.txt.csv')
words = df.ix[:,0].tolist()
freqs = df.ix[:,1].tolist()
count = len(words)

plt.plot(range(count), freqs)
plt.yscale('log')
plt.xscale('log')
plt.show()