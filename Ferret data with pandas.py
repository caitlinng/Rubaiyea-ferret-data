#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd

'''
Convert dataset to csv file. Delimiter = seperator 
Read csv file and convert to pd dataframe (2D array)
Headers taken from first column (alternatively use names = ['',''])
'''
pd.read_csv('zoo.csv', delimiter = ',') 

# Store output in article_read
article_read = pd.read_csv('zoo.csv', delimiter = ',') 

# Can read select lines only with .head(), .tail(), .sample(n)
print(article_read)

article_read[['animal', 'water_need']]

article_read[article_read.animal == 'elephant']


# In[14]:


def ReadFemPreg(dct_file='2002FemPreg.dct',
                dat_file='2002FemPreg.dat.gz'):
    dct = thinkstats2.ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dat_file, compression='gzip')
    CleanFemPreg(df)
    return df


# In[33]:


'''
Creating dictionary where 
key: influenza strains
index: ferret data (as pd dataframe)
'''

ferrets_file = pd.ExcelFile('Time_series_ForJM.xlsx')
ferrets_sheets = {} 

for sheet in ferrets_file.sheet_names:
    sheet[['OST']]
    ferrets_sheets[sheet] = ferrets_file.parse(sheet)

# Transpose data

# Validate
    
# Create df with only OST and/or plc titres
OST_sheets = {}
for i in ferrets_sheets:
    ferrets_sheets.get(i)


# In[ ]:




