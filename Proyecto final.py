#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.compat import lzip
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from scipy.stats import bartlett
from statsmodels.stats import diagnostic as diag
from sklearn.linear_model import LinearRegression

df = pd.read_excel("C:/Users/brianz0r/Downloads/Startups.xlsx")
df.head()


# In[12]:


df.describe()


# In[4]:


y = df['loan']
x = df[['age','credit-rating','children']]
x_const = sm.add_constant(x)


# In[35]:


# define the multiple Linear regression model
linear_regress = LinearRegression()


# In[36]:


# Fit the multiple Linear regression model
linear_regress.fit(x,y)


# In[38]:


modelo = sm.OLS(y, x_const).fit()


# In[39]:


print(modelo.summary())


# In[60]:


## Esperanza de los residuos es 0

mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))


# In[40]:


## Normalidad

name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(modelo.resid)
lzip(name, test)


# In[59]:


## Homocedasticidad

regr = linear_model.LinearRegression()
regr.fit(x_const,y)
y_pred = regr.predict(x_const)

residuals = y-y_pred

name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(residuals, x_const)
lzip(name, test)


# In[69]:


## Autocorrelación

min(diag.acorr_ljungbox(residuals , lags = 25)[1])


# In[ ]:




