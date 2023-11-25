#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from scipy.stats import skew,kurtosis


# In[4]:


df = pd.read_csv('Q9_b.csv',index_col=0)


# In[9]:


df.head()


# In[7]:


print(skew(df)) #SP has a positive skewness - longer right tail - +ve values of mean, mode and median. 
                #WT has a negative skewness - longer left tail - -ve values of mean, mode, and median.


# In[8]:


print(kurtosis(df)) #SP and WT have positive kurtosis - values are more at the tails and less at the peak.


# In[ ]:




