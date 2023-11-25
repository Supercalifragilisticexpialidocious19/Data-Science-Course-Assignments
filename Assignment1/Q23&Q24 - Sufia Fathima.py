#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import scipy as stats
import math
from scipy import stats
from scipy.stats import t


# In[16]:


#n = 18
#260
# STD = 90
#18 ---> 260?


# In[17]:


#t_score = (x - pop mean) / (sample standard daviation / square root of sample size)


# In[18]:


tscore=(260-270)/(90/(math.sqrt(18)))
tscore


# In[19]:


# tscore = -0.4714045207910317 
#degree of freedom = n-1 = 17
dof=18-1
dof


# In[25]:


p_val = stats.t.cdf(tscore,dof)
p_val


# ## Q23

# In[26]:


#sample size = 25
dof = 25-1
dof


# In[28]:


#tscore for 95% confidence Interval
confidence_interval=0.95
lower_lim = t.ppf((1-confidence_interval)/2,dof)
upper_lim = t.ppf((1+confidence_interval)/2,dof)
print(lower_lim)
print(upper_lim)


# In[29]:


#tscore for 96% confidence interval
confidence_interval = 0.96
lower_lim = t.ppf((1-confidence_interval)/2,dof)
upper_lim = t.ppf((1+confidence_interval)/2,dof)
print(lower_lim)
print(upper_lim)


# In[30]:


#tscore for 99% confidence interval
confidence_interval = 0.99
lower_lim = t.ppf((1-confidence_interval)/2,dof)
upper_lim = t.ppf((1+confidence_interval)/2,dof)
print(lower_lim)
print(upper_lim)


# In[ ]:




