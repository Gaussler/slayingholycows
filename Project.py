#!/usr/bin/env python
# coding: utf-8

# # Project

# In[155]:


import numpy as np
import pandas as pd 
import math
from random import gauss
from math import sqrt
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from math import log, sqrt, exp
from scipy.stats import norm
from glob import glob
import warnings
import os
from functools import reduce
warnings.filterwarnings("ignore")


# # Data Acquisition and Cleaning - 10 years daily from NASDAQ 100 Technology Sector 

# In[37]:


# function for reading files
def read_file(f):
    df = pd.read_csv(f)
    df['ticker'] = f.split('.')[0].strip('^')
    return df

# function for getting log-returns
def getlogret(data):
    logret = np.log(data) - np.log(data.shift(1))
    return logret


# In[38]:


# create the dataframe for closing price, named as 'close'
close = pd.concat([read_file(f) for f in glob('*.csv')])
close = close.set_index(['Date','ticker'])[['Close']].unstack()
# extract the tickers for renaming purpose later
tickers = close.columns.get_level_values('ticker')
#print(close)


# In[39]:


# create the dataframe for logreturns, named as 'logret'
logret = close
for i in range(0,len(close.axes[1])):
    logret.iloc[:,i] = getlogret(close.iloc[:,i])
logret = logret.iloc[1:].rename(columns={'Close': 'Logret'})
#print(logret)


# # Data Analysis - use the historical data to predict the future

# In[40]:


# define functions for plotting histogram and the numerical results
# given the error functions and a fixed timewindow
def plothist(function, tw):
    myerr = function(logret, tw).iloc[1:]
    fig = plt.figure(figsize = (15,10))
    ax = fig.gca()
    errplot = myerr.hist(ax = ax)
    errvalue = print(myerr)
    return errvalue, errplot


# ## Volatility

# In[41]:


# Define functions for actual/predicted volatility and the absolute error between them

def volpred(df, n):
    # keep the index for non-overlapped rolling window. eg for a=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], a[2::3]=[2, 5, 8]
    return df.rolling(n).std()[n-1::n] 

def volerr(df, n):
    df_volpred = volpred(df, n) 
    df_volerr = (df_volpred - df_volpred.shift(1)).rename(columns={'Logret': 'Residual Error (Volatility)'})
    return df_volerr

def volabserr(df, n):
    df_volpred = volpred(df, n)
    df_volabserr = (df_volpred - df_volpred.shift(1)).abs().rename(columns={'Logret': 'Absolute Error (Volatility)'})
    return df_volabserr


# In[174]:


# Define error functions, relative errors, taking size of error into account

def volAPE(df,n): # use absolute percentage error
    df_volpred = volpred(df, n)
    df_volerr = (df_volpred - df_volpred.shift(1))
    df_volerr.columns = tickers
    df_volpred.columns = tickers
    df_volAPE = (df_volerr.iloc[1:].div(df_volpred.iloc[1:])).abs() # abs((actual-pred)/actual) 
    df_volAPE.columns=pd.MultiIndex.from_product([['APE (Volatility)'], df_volAPE.columns])
    return df_volAPE

def volAMPE(df,n): # use absolute mean percentage error
    df_volpred = volpred(df, n)
    df_volerr = (df_volpred - df_volpred.shift(1))
    df_volerr.columns = tickers
    df_volpred.columns = tickers
    avg = (df_volerr.iloc[1:] + df_volpred.iloc[1:]) / 2
    df_volAMPE = (df_volerr.iloc[1:].div(avg)).abs() # abs((actual-pred)/avg) 
    df_volAMPE.columns=pd.MultiIndex.from_product([['AMPE (Volatility)'], df_volAMPE.columns])
    return df_volAMPE

def volLAAR(df,n): # use absolute value of log of accuracy ratio as error
    df_volpred = volpred(df, n)
    df_volact = df_volpred.shift(1)
    df_volact.columns = tickers
    df_volpred.columns = tickers
    df_volLAAR = np.log(((df_volpred.iloc[1:].div(df_volact.iloc[1:]))).abs()) # log(abs(pred/act)) 
    df_volLAAR.columns=pd.MultiIndex.from_product([['LAAR (Volatility)'], df_volLAAR.columns])
    return df_volLAAR


# ### Iterate through the time window

# In[142]:


# # define functions for printing the numerical results and plotting the dot graphs
# given the error functions, starting&ending timewindow with step size, tail
def plotdot_vol(function, tw1, tw2, step, tail):
    
    # create a NaN dataframe with rows as the time window, columns as the securities
    col_names = [i for i in range(tw1, tw2+step, step)] # timewindow
    # generate NaN entries
    data = np.empty((close.shape[1], len(col_names))) # number of rows = number of securities
    # create dataframe
    data[:] = np.nan
    dfnan = pd.DataFrame(data, columns=col_names)
    
    # fill in the NaN dataframe with the percentage of error observations in tail
    percent = dfnan
    j=0
    for i in range(tw1, tw2+step, step):
        myerr = (function(logret, i).iloc[1:])*np.sqrt(260) # use annual vol
        # create df 'outlier' consists of T/F, T if abs err>tail
        outlier = myerr > tail
        # fill in values representing the percentage of abs error outside tail
        # outlier.mean() calculates percentage of T values 
        percent.iloc[:,j] = outlier.mean().reset_index(drop=True) 
        j = j+1
    
    percent_T = percent.transpose() # exchange x and y axis for plotting purpose
    percent_T.columns = tickers # rename the columns as the securities
    dotvalue = print(percent_T)

    # plot the dot chart
    dotplot = percent_T.plot(ls = '', marker = '.', figsize = (10,5), title = f"Volatility, Percentage of error observations outside tail={tail}, {str(function.__name__).strip('vol')}")
    dotplot.legend(bbox_to_anchor = (1.2, 1))
    return dotvalue, dotplot


# ## Mean

# In[49]:


# Define functions for actual/predicted mean and the absolute error between them

def meanpred(df, n):
    return df.rolling(n).mean()[n-1::n]

def meanerr(df, n):
    df_meanpred = meanpred(df, n)
    df_meanerr = (df_meanpred - df_meanpred.shift(1)).rename(columns={'Logret': 'Residual Error (Mean)'})
    return df_meanerr 

def meanabserr(df, n):
    df_meanpred = meanpred(df, n)
    df_meanabserr = (df_meanpred - df_meanpred.shift(1)).abs().rename(columns={'Logret': 'Absolute Error (Mean)'})
    return df_meanabserr


# In[173]:


# Define error functions, relative errors, taking size of error into account

def meanAPE(df,n): # use absolute percentage error
    df_meanpred = meanpred(df, n)
    df_meanerr = (df_meanpred - df_meanpred.shift(1))
    df_meanerr.columns = tickers
    df_meanpred.columns = tickers
    df_meanAPE = (df_meanerr.iloc[1:].div(df_meanpred.iloc[1:])).abs() # abs((actual-pred)/actual)
    df_meanAPE.columns=pd.MultiIndex.from_product([['APE (Mean)'], df_meanAPE.columns])
    return df_meanAPE

def meanAMPE(df,n): # use absolute mean percentage error
    df_meanpred = meanpred(df, n)
    df_meanerr = (df_meanpred - df_meanpred.shift(1))
    df_meanerr.columns = tickers
    df_meanpred.columns = tickers
    avg = (df_meanerr.iloc[1:] + df_meanpred.iloc[1:]) / 2
    df_meanAMPE = (df_meanerr.iloc[1:].div(avg)).abs() # abs((actual-pred)/avg) 
    df_meanAMPE.columns=pd.MultiIndex.from_product([['AMPE (Mean)'], df_meanAMPE.columns])
    return df_meanAMPE

def meanLAAR(df,n): # use absolute value of log of accuracy ratio as error
    df_meanpred = meanpred(df, n)
    df_meanact = df_meanpred.shift(1)
    df_meanact.columns = tickers
    df_meanpred.columns = tickers
    df_meanLAAR = np.log((df_meanpred.iloc[1:].div(df_meanact.iloc[1:])).abs()) # abs(log(pred/act))
    df_meanLAAR.columns=pd.MultiIndex.from_product([['LAAR (Mean)'], df_meanLAAR.columns])
    return df_meanLAAR


# ### Iterate through the time window

# In[105]:


# # define functions for printing the numerical results and plotting the dot graphs
# given the error functions, starting&ending timewindow with step size, tail

def plotdot_mean(function, tw1, tw2, step, tail):
    
    # create a NaN dataframe with rows as the time window, columns as the securities
    col_names = [i for i in range(tw1, tw2+step, step)] # timewindow
    # generate NaN entries
    data = np.empty((close.shape[1], len(col_names))) # number of rows = number of securities
    # create dataframe
    data[:] = np.nan
    dfnan = pd.DataFrame(data, columns=col_names)
    
    # fill in the NaN dataframe with the percentage of error observations in tail
    percent = dfnan
    j=0
    for i in range(tw1, tw2+step, step):
        myerr = (function(logret, i).iloc[1:])*260 # use annual mean
        # create df 'outlier' consists of T/F, T if abs err>tail
        outlier = myerr > tail
        # fill in values representing the percentage of abs error outside tail
        # outlier.mean() calculates percentage of T values 
        percent.iloc[:,j] = outlier.mean().reset_index(drop=True) 
        j = j+1
    
    percent_T = percent.transpose()
    percent_T.columns = tickers # rename the columns as the securities
    dotvalue = print(percent_T)

    # plot the dot chart
    dotplot = percent_T.plot(ls = '', marker = '.', figsize = (10,5), title = f"Mean, Percentage of error observations outside tail={tail}, {str(function.__name__).strip('mean')}")
    dotplot.legend(bbox_to_anchor = (1.2, 1))
    return dotvalue, dotplot


# ## Compute results for volatility & mean

# In[144]:


# Volatility, AMPE, histogram, time window=70
plothist(volAMPE, 70)


# In[126]:


# Mean, AMPE, histogram, time window=70
plothist(volAMPE, 70)


# In[102]:


# Volatility, AMPE, time window from 5 to 150, step size 5, tail 50%
plotdot_vol(volAMPE, 5, 150, 5, 0.5)


# In[103]:


# Volatility, AMPE, time window from 5 to 150, step size 5, tail 10%
plotdot_vol(volAMPE, 5, 150, 5, 0.1)


# In[104]:


# Volatility, AMPE, time window from 5 to 150, step size 5, tail 0
plotdot_vol(volAMPE, 5, 150, 5, 0)


# In[109]:


# Mean, AMPE, time window from 5 to 150, step size 5, tail 50%
plotdot_mean(meanAMPE, 5, 150, 5, 0.5)


# In[110]:


# Mean, AMPE, time window from 5 to 150, step size 5, tail 10%
plotdot_mean(meanAMPE, 5, 150, 5, 0.1)


# In[111]:


# Mean, AMPE, time window from 5 to 150, step size 5, tail 0%
plotdot_mean(meanAMPE, 5, 150, 5, 0)


# In[127]:


# Volatility, APE, histogram, time window=70
plothist(volAPE, 70)


# In[128]:


# Mean, APE, histogram, time window=70
plothist(meanAPE, 70)


# In[130]:


# Mean, APE, time window from 5 to 150, step size 5, tail 50%
plotdot_vol(volAPE, 5, 150, 5, 0.5)


# In[131]:


# Mean, APE, time window from 5 to 150, step size 5, tail 10%
plotdot_vol(volAPE, 5, 150, 5, 0.1)


# In[132]:


# Mean, APE, time window from 5 to 150, step size 5, tail 0
plotdot_vol(volAPE, 5, 150, 5, 0)


# In[133]:


# Mean, APE, time window from 5 to 150, step size 5, tail 50%
plotdot_mean(meanAPE, 5, 150, 5, 0.5)


# In[134]:


# Mean, APE, time window from 5 to 150, step size 5, tail 10%
plotdot_mean(meanAPE, 5, 150, 5, 0.1)


# In[138]:


# Mean, APE, time window from 5 to 150, step size 5, tail 0
plotdot_mean(meanAPE, 5, 150, 5, 0)


# In[175]:


# Volatility, LAAR, histogram, time window=70
plothist(volLAAR, 70)


# In[176]:


# mean, ALAR, histogram, time window=70
plothist(meanLAAR, 70)


# In[177]:


# Volatility, LAAR, time window from 5 to 150, step size 5, tail 1.65 (e^0.5)
plotdot_vol(volLAAR, 5, 150, 5, math.e**0.5)


# In[179]:


# Volatility, LAAR, time window from 5 to 150, step size 5, tail e^0.9
plotdot_vol(volLAAR, 5, 150, 5, math.e**0.9)


# In[186]:


# Volatility, LAAR, time window from 5 to 150, step size 5, tail e^100
plotdot_vol(volLAAR, 5, 150, 5, math.e**100)


# In[187]:


# Volatility, LAAR, time window from 5 to 150, step size 5, tail e^-1
plotdot_vol(volLAAR, 5, 150, 5, math.e**-1)


# In[188]:


# Mean, LAAR, time window from 5 to 150, step size 5, tail e^0.5
plotdot_mean(meanLAAR, 5, 150, 5, math.e**0.5)


# In[189]:


# Mean, LAAR, time window from 5 to 150, step size 5, tail e^0.9
plotdot_mean(meanLAAR, 5, 150, 5, math.e**0.9)


# In[190]:


# Volatility, LAAR, time window from 5 to 150, step size 5, tail e^100
plotdot_mean(meanLAAR, 5, 150, 5, math.e**100)


# In[191]:


# Volatility, LAAR, time window from 5 to 150, step size 5, tail e^100
plotdot_mean(meanLAAR, 5, 150, 5, math.e**-100)


# In[ ]:




