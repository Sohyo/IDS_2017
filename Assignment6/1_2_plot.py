from pylab import *
import matplotlib.pyplot  as plt
import config as cfg
import pandas as pd
from nltk.corpus import gutenberg
from matplotlib.font_manager import FontProperties
import powerlaw
import numpy as np
import operator

path = './term_frequency2/'
matplotlib.style.use('ggplot')

titles = gutenberg.fileids()

count = 3000
freqs1 = []
for i in range(count):
    freqs1.append(count/(i+1))

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
i=0
colormap = plt.cm.nipy_spectral #I suggest to use nipy_spectral, Set1,Paired
number_of_plots = 18
ax1.set_color_cycle([colormap(i) for i in np.linspace(0, 1,number_of_plots)])
for title in titles:
	df = pd.read_csv(path + title+'.csv',header=None)	
	words = df.ix[:,0].tolist()
	freqs = df.ix[:,1].tolist()
	count = len(words)
	number_of_plots=18	
	plt.plot(range(count), freqs)
	i+=1


plt.plot(range(len(freqs1)), freqs1, '--')

titles_clean = [title[:-4] for title in titles]
plt.legend(titles_clean, loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Zipf\'s Law for 18 books')
plt.xlabel('log(rank)')
plt.ylabel('log(frequency)')
plt.yscale('log')
plt.xscale('log')
# fit = powerlaw.Fit(np.array(freqs)+1,xmin=1,discrete=True)
# fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')

plt.show(block =False)

savefig(path+'/../'+'zipfs_law.png')

# Calculate best fit to a power law

x= {}

for title in titles:
	df = pd.read_csv(path + title+'.csv',header=None)
	freqs = df.ix[:,1].tolist()	
	results = powerlaw.Fit(freqs)
	R = results.distribution_compare('power_law', 'exponential',normalized_ratio=True)
	x[title] = R

sorted_x = sorted(x.items(), key=operator.itemgetter(1))
df1 = pd.DataFrame(sorted_x)

d1 = {}
for title in titles:
	df = pd.read_csv(path + title+'.csv',header=None)
	freqs = df.ix[:,1].tolist()	
	d1[title] = freqs

df2 = pd.DataFrame(d1)
R, p = fit.distribution_compare(d1.get('bible-kjv.txt'), d1.get('austen-emma.txt'))
R, p = fit.distribution_compare(d1.get('bible-kjv.txt'), d1.get('austen-emma.txt'))


R = results.distribution_compare('power_law', 'lognormal_positive')

# Best fit is the Bible. 


