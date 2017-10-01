import pandas as pd    
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
path = "/Users/danielmlow/Dropbox/lct/data_science/assignment4/lastfm-dataset-1K/"
df = pd.read_csv(path+'last_fm_user_band.csv')
df = df.iloc[:,1:]


pd.set_option('display.float_format', lambda x: '%.4f' % x)
uniq = df.iloc[:,1].value_counts(normalize=True)
uniq_20 = uniq[:20]

uniq_20.plot.bar()
plt.xlabel('Bands')
plt.ylabel('Probability of being played')
plt.title('Highest frequency bands')
plt.savefig(path+'../'+'highest_freq_bands.png',bbox_inches='tight')


d = {}
for index, row in df.iterrows():
	user = row['user']
	band = row['band']
	if user in d:
		d[user].add(band)
	else:
		print(user)
		d[user] = set([])
		d[user].add(band)
	
l = []
for i in d:
	l.append(len(d[i]))


l = l[:-2]

arr = np.asarray(l)
plt.hist(arr,bins=20,normed=True)
plt.ylabel('Normalized frequency of users')
plt.xlabel('Amount of bands per user')
plt.title('Amount of bands per user')
plt.savefig(path+'../'+'bands per user.png',bbox_inches='tight')
