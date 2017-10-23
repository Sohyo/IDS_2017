import pandas as pd
import config as cfg
import os

path = cfg.path 

df3 = pd.read_csv(os.join(path,'data3.csv'))
df5 = pd.read_csv(os.join(path,'data6.csv'))

