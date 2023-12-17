#!/usr/bin/env python
# coding: utf-8

# In[2]:


## 1) Delivery_time -> Predict delivery time using sorting time 
##2) Salary_hike -> Build a prediction model for Salary_hike

##------------------------------------------------------------

##Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.'''


# In[49]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error


# In[4]:


df=pd.read_csv('delivery_time.csv')
df


# In[5]:


df.isnull().sum()


# In[6]:


df.duplicated()


# In[7]:


df.boxplot('Delivery Time')


# In[8]:


target = df['Delivery Time']
target


# In[9]:


features = df.drop('Delivery Time',axis=1)
features


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(features, target, train_size=0.75, random_state = 3)


# In[12]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[13]:


from sklearn.linear_model import LinearRegression


# In[14]:


lin_model = LinearRegression()


# In[15]:


lin_model.fit(x_train,y_train)


# In[16]:


y_pred=lin_model.predict(x_test)


# In[17]:


y_pred


# In[18]:


plt.scatter(df['Sorting Time'],df['Delivery Time'])
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.title('Simple Linear Regression')
plt.plot(df['Sorting Time'],lin_model.predict(df[['Sorting Time']]),color='green')
plt.show()


# In[19]:


lin_model.score(x_test,y_test)


# In[123]:


rscore = r2_score(y_test, y_pred)
print("The accuracy of our model is {}%".format(round(rscore, 2) *100))


# ## Transforming the data to Logarithm values

# In[20]:


df['SortingTime_log2'] = np.log2(df['Sorting Time'])


# In[21]:


df['DelTime_log'] = np.log2(df['Delivery Time'])


# In[22]:


DelTimeLogdf = df[['SortingTime_log2','DelTime_log']]
DelTimeLogdf


# In[103]:


Del_features_log = DelTimeLogdf.drop('DelTime_log',axis=1)
Del_target_log = DelTimeLogdf['DelTime_log']


# In[104]:


x_trainl1, x_testl1, y_trainl1, y_testl1 = train_test_split(Del_features_log, Del_target_log, train_size=0.75, random_state = 10)


# In[105]:


print(x_trainl1.shape)
print(y_trainl1.shape)
print(x_testl1.shape)
print(y_testl1.shape)


# In[106]:


logarithmlin_model = LinearRegression()


# In[107]:


logarithmlin_model.fit(x_trainl1,y_trainl1)


# In[108]:


y_pred_log = logarithmlin_model.predict(x_testl1)


# In[109]:


y_pred_log


# In[110]:


plt.scatter(df['SortingTime_log2'],df['DelTime_log'])
plt.xlabel('Sorting Time in Log')
plt.ylabel('DelTime_log')
plt.title('Simple Linear Regression')
plt.plot(DelTimeLogdf['SortingTime_log2'],logarithmlin_model.predict(DelTimeLogdf[['SortingTime_log2']]),color='green')
plt.show()


# In[111]:


logarithmlin_model.score(x_testl1,y_testl1)


# In[125]:


delscore = r2_score(y_testl1, y_pred_log)
print("The accuracy of our model is {}%".format(round(delscore, 2) *100))


# ## Transforming data to sqrt values

# In[93]:


df['SortingTime_sqrt'] = np.sqrt(df['Sorting Time'])
df['DelTime_sqrt'] = np.sqrt(df['Delivery Time'])


# In[94]:


Delsqrtdf = df[['SortingTime_sqrt','DelTime_sqrt']]
Delsqrtdf


# In[112]:


Del_features_sq = Delsqrtdf.drop('DelTime_sqrt',axis=1)
Del_target_sq = Delsqrtdf['DelTime_sqrt']


# In[113]:


x_trainsq, x_testsq, y_trainsq, y_testsq = train_test_split(Del_features_sq, Del_target_sq, train_size=0.75, random_state = 10)


# In[114]:


print(x_trainsq.shape)
print(y_trainsq.shape)
print(x_testsq.shape)
print(y_testsq.shape)


# In[115]:


sqrtlin_model = LinearRegression()


# In[116]:


sqrtlin_model.fit(x_trainsq,y_trainsq)


# In[117]:


y_predsq=sqrtlin_model.predict(x_testsq)


# In[118]:


y_predsq


# In[120]:


plt.scatter(Delsqrtdf['SortingTime_sqrt'],Delsqrtdf['DelTime_sqrt'])
plt.xlabel('sqrt_sorting time')
plt.ylabel('Deltime_sqrt')
plt.title('Simple Linear Regression')
plt.plot(Delsqrtdf['SortingTime_sqrt'],sqrtlin_model.predict(Delsqrtdf[['SortingTime_sqrt']]),color='purple')
plt.show()


# In[121]:


sqrtlin_model.score(x_testsq,y_testsq)


# In[126]:


sqscore = r2_score(y_testsq, y_predsq)
print("The accuracy of our model is {}%".format(round(sqscore, 2) *100))


# ## Transforming the data to cuberoot values

# In[127]:


df['SortingTime_cbrt'] = np.cbrt(df['Sorting Time'])
df['DelTime_cbrt'] = np.cbrt(df['Delivery Time'])


# In[128]:


Delcbrtdf = df[['SortingTime_cbrt','DelTime_cbrt']]
Delcbrtdf


# In[130]:


Del_features_cb = Delcbrtdf.drop('DelTime_cbrt',axis=1)
Del_target_cb = Delcbrtdf['DelTime_cbrt']


# In[131]:


x_traincb, x_testcb, y_traincb, y_testcb = train_test_split(Del_features_cb, Del_target_cb, train_size=0.75, random_state = 10)


# In[132]:


print(x_traincb.shape)
print(x_testcb.shape)
print(y_traincb.shape)
print(y_testcb.shape)


# In[133]:


cbrtlin_model = LinearRegression()


# In[134]:


cbrtlin_model.fit(x_traincb,y_traincb)


# In[135]:


y_pred_cbrt = cbrtlin_model.predict(x_testcb)


# In[136]:


y_pred_cbrt


# In[137]:


plt.scatter(Delcbrtdf['SortingTime_cbrt'],Delcbrtdf['DelTime_cbrt'])
plt.xlabel('cbrt_sorting time')
plt.ylabel('Deltime_cbrt')
plt.title('Simple Linear Regression')
plt.plot(Delcbrtdf['SortingTime_cbrt'],cbrtlin_model.predict(Delcbrtdf[['SortingTime_cbrt']]),color='red')
plt.show()


# In[139]:


cbrtlin_model.score(x_testcb,y_testcb)


# In[140]:


cbscore = r2_score(y_testcb, y_pred_cbrt)
print("The accuracy of our model is {}%".format(round(cbscore, 2) *100))


# ### Therefore the scores of the models are as follows:
# #### 1. No transformation = 79.2%
# #### 2. Logarithm transformed = 58.2%
# #### 3. Cbrt = 55%
# #### 4. Sqrt = 53.5%

# ### Model with no transformation shows the highest accuracy score with score = 79.2%

# ## Salary Dataset

# In[32]:


df1 = pd.read_csv('Salary_Data.csv')
df1


# In[33]:


df1.duplicated()


# In[34]:


df1.isnull().sum()


# In[35]:


df1.boxplot('Salary')


# In[36]:


features1=df1.drop('Salary',axis=1)


# In[37]:


features1


# In[38]:


target1=df1['Salary']
target1


# In[39]:


x_train1, x_test1, y_train1, y_test1 = train_test_split(features1, target1, train_size=0.75, random_state = 50)


# In[40]:


print(x_train1.shape)
print(x_test1.shape)
print(y_train1.shape)
print(y_test1.shape)


# In[41]:


linear_model = LinearRegression()


# In[42]:


linear_model.fit(x_train1,y_train1)


# In[43]:


y_pred1=linear_model.predict(x_test1)


# In[44]:


y_pred1


# In[45]:


linear_model.intercept_


# In[46]:


plt.scatter(df1['YearsExperience'],df1['Salary'])
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Simple Linear Regression')
plt.plot(df1['YearsExperience'],linear_model.predict(df1[['YearsExperience']]),color='green')
plt.show()


# In[47]:


linear_model.score(x_test1,y_test1)


# In[50]:


score = r2_score(y_test1, y_pred1)
print("The accuracy of our model is {}%".format(round(score, 2) *100))


# ## Transforming the data to log values

# In[51]:


df1['log_base2'] = np.log2(df1['Salary'])


# In[52]:


df1['log_YOE'] = np.log2(df1['YearsExperience'])


# In[53]:


df1


# In[54]:


featureslog=df1[['log_YOE']]
targetlog=df1[['log_base2']]


# In[55]:


x_trainl, x_testl, y_trainl, y_testl = train_test_split(featureslog, targetlog, train_size=0.75, random_state = 50)


# In[56]:


print(x_trainl.shape)
print(x_testl.shape)
print(y_trainl.shape)
print(y_testl.shape)


# In[57]:


loglinear_model = LinearRegression()


# In[58]:


#x_trainl=np.reshape(-1,1)
#y_trainl=np.reshape(-1,1)
loglinear_model.fit(x_trainl,y_trainl)


# In[59]:


y_predlog=loglinear_model.predict(x_testl)


# In[60]:


y_predlog


# In[61]:


plt.scatter(df1['log_YOE'],df1['log_base2'])
plt.xlabel('log_YOE')
plt.ylabel('Transformed salary')
plt.title('Simple Linear Regression')
plt.plot(df1['log_YOE'],loglinear_model.predict(df1[['log_YOE']]),color='green')
plt.show()


# In[62]:


loglinear_model.score(x_testl,y_testl)


# In[63]:


from sklearn.metrics import r2_score, mean_absolute_error
score = r2_score(y_testl, y_predlog)
print("The accuracy of our model is {}%".format(round(score, 2) *100))


# In[64]:


mae = mean_absolute_error(y_test1,y_predlog)
mae


# ## Transforming the data to sqrt values

# In[65]:


df1['Salary_sqrt'] = np.sqrt(df1['Salary'])


# In[66]:


df1['sqrt_YOE'] = np.sqrt(df1['YearsExperience'])


# In[67]:


df1


# In[68]:


sqrtdf=df1[['sqrt_YOE','Salary_sqrt']]
sqrtdf


# In[69]:


features_sq = sqrtdf.drop('Salary_sqrt',axis=1)
features_sq


# In[70]:


targ_sq=sqrtdf['Salary_sqrt']
targ_sq


# In[71]:


x_trains, x_tests, y_trains, y_tests = train_test_split(features_sq, targ_sq, train_size=0.75, random_state = 50)


# In[72]:


print(x_trains.shape)
print(x_tests.shape)
print(y_trains.shape)
print(y_tests.shape)


# In[73]:


sqrtlinear_model = LinearRegression()


# In[74]:


sqrtlinear_model.fit(x_trains,y_trains)


# In[75]:


y_predsqrt=sqrtlinear_model.predict(x_tests)


# In[76]:


y_predsqrt


# In[77]:


plt.scatter(sqrtdf['sqrt_YOE'],sqrtdf['Salary_sqrt'])
plt.xlabel('sqrt_YOE')
plt.ylabel('Transformed salary')
plt.title('Simple Linear Regression')
plt.plot(sqrtdf['sqrt_YOE'],sqrtlinear_model.predict(sqrtdf[['sqrt_YOE']]),color='purple')
plt.show()


# In[78]:


sqrtlinear_model.score(x_tests,y_tests)


# In[79]:


score_sqrt = r2_score(y_tests, y_predsqrt)
print("The accuracy of our model is {}%".format(round(score_sqrt, 2) *100))


# ## Transforming the data to cuberoot values

# In[80]:


df1['Salary_cbrt'] = np.cbrt(df1['Salary'])


# In[81]:


df1['cbrt_YOE'] = np.cbrt(df1['YearsExperience'])


# In[82]:


cbrtdf=df1[['cbrt_YOE','Salary_cbrt']]
cbrtdf


# In[83]:


features_cb = cbrtdf.drop('Salary_cbrt',axis=1)
targ_cb = cbrtdf[['Salary_cbrt']]


# In[84]:


x_trainc, x_testc, y_trainc, y_testc = train_test_split(features_cb, targ_cb, train_size=0.75, random_state = 50)


# In[85]:


print(x_trainc.shape)
print(x_testc.shape)
print(y_trainc.shape)
print(y_testc.shape)


# In[86]:


cbrtlinear_model = LinearRegression()


# In[87]:


cbrtlinear_model.fit(x_trainc, y_trainc)


# In[88]:


y_predcbrt = cbrtlinear_model.predict(x_testc)


# In[89]:


y_predcbrt


# In[90]:


plt.scatter(cbrtdf['cbrt_YOE'],cbrtdf['Salary_cbrt'])
plt.xlabel('cbrt_YOE')
plt.ylabel('Transformed salary')
plt.title('Simple Linear Regression')
plt.plot(cbrtdf['cbrt_YOE'],cbrtlinear_model.predict(cbrtdf[['cbrt_YOE']]),color='red')
plt.show()


# In[91]:


cbrtlinear_model.score(x_testc,y_testc)


# In[92]:


score_cbrt = r2_score(y_testc, y_predcbrt)
print("The accuracy of our model is {}%".format(round(score_cbrt, 2) *100))


# ### Therefore from the above transformations, it is clear that data transformed to squareroot model gives the highest accuracy score of 89.9% rounded to 90%. 

# #### Order of accuracy scores:
# #### 1. sqrt = 89.9 ~ 90%
# #### 2. cbrt = 89.6 ~ 90%
# #### 3. no transformation = 89.1 ~ 89%
# #### 4. log transformation = 87.2 ~ 87%

# In[ ]:




