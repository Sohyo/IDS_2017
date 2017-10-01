
import pandas as pd    
import seaborn as sns
import numpy as np
from pandas import DataFrame

#Import dataframe 2 columns user and band 
path = "/Users/danielmlow/Dropbox/lct/data_science/assignment4/lastfm-dataset-1K/"
df = pd.read_csv(path+'last_fm_user_band.csv')
df = df.iloc[:,1:]

# Import user info dataframe 
df1 = DataFrame.from_csv(path+"userid-profile.tsv", sep="\t")

#Turn user info dataframe into user info dictionary
d1 = {}
for index, row in df1.iterrows():
	user = index
	d1[index] = [row[0],row[1],row[2]]

# less than 20 years old = yound, more than 20 = old. 
import math

for i in d1.keys():
	if math.isnan(d1[i][1]):
		continue
	elif d1[i][1] <= 20:
		d1[i][1] = 'young'
	elif d1[i][1] >20:
		d1[i][1] = 'old'

#Take only first 20% of dta. 
# perc_20= int(np.round(df.shape[0]*0.01))
# df_subsample = df[:perc_20]

# Random sample
df_subsample = df.sample(frac=0.10)

# Add three columns (sex,age,country) with empty values (0s or "" for categorical)
sex1 = np.full(df_subsample.shape[0],'')
df_subsample = df_subsample.assign(sex=pd.Series(sex1).values)
age1 = np.full(df_subsample.shape[0],0)
df_subsample = df_subsample.assign(age=pd.Series(age1).values)
country1 = sex1
df_subsample = df_subsample.assign(country=pd.Series(country1).values)

# Insert user info
for i in d1.keys():
	df_subsample.loc[df_subsample.user == i, 'sex'] = d1[i][0]
	df_subsample.loc[df_subsample.user == i, 'age'] = d1[i][1]
	df_subsample.loc[df_subsample.user == i, 'country'] = d1[i][2]

# Trim to band+certain user info
df_sex = df_subsample[['band','sex']]
df_age = df_subsample[['band','age']]
df_country = df_subsample[['band','country']]

#Save 
df_subsample.to_csv(path+'../'+'20%_sex_age_country.csv')
df_sex.to_csv(path+'../'+'20%_sex.csv')
df_age.to_csv(path+'../'+'20%_age.csv')
df_country.to_csv(path+'../'+'20%_country.csv')







