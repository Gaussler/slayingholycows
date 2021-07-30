#!/usr/bin/env python
# coding: utf-8

# # Project

# In[1]:


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

# In[2]:


# function for reading files
def read_file(f):
    df = pd.read_csv(f)
    df['ticker'] = f.split('.')[0].strip('^')
    return df

# function for getting log-returns
def getlogret(data):
    logret = np.log(data) - np.log(data.shift(1))
    return logret


# In[3]:


# create the dataframe for closing price, named as 'close'
close = pd.concat([read_file(f) for f in glob('*.csv')])
close = close.set_index(['Date','ticker'])[['Close']].unstack()
# extract the tickers for renaming purpose later
tickers = close.columns.get_level_values('ticker')
print(close)


# In[4]:


# create the dataframe for logreturns, named as 'logret'
logret = close
for i in range(0,len(close.axes[1])):
    logret.iloc[:,i] = getlogret(close.iloc[:,i])
logret = logret.iloc[1:].rename(columns={'Close': 'Logret'})
print(logret)


# # Data Analysis - use the historical data to predict the future

# ## Volatility

# In[5]:


# Define functions for actual/predicted volatility and the absolute error between them

def volpred(df, n):
    # keep the index for non-overlapped rolling window. eg for a=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], a[2::3]=[2, 5, 8]
    return (df.rolling(n).std()[n-1::n])*np.sqrt(260) # change to annualized volatility

def volerr(df, n):
    df_volpred = volpred(df, n) 
    df_volerr = (df_volpred.shift(-1) - df_volpred).rename(columns={'Logret': 'Residual Error (Volatility)'})
    df_volerr = df_volerr.iloc[:-1] # delete the last NaN row
    return df_volerr

def volabserr(df, n):
    df_volpred = volpred(df, n)
    df_volabserr = (df_volpred.shift(-1) - df_volpred).abs().rename(columns={'Logret': 'Absolute Error (Volatility)'})
    df_volabserr = df_volabserr.iloc[:-1] # delete the last NaN row
    return df_volabserr


# In[6]:


# Define error functions, relative errors, taking size of error into account

def volAPE(df,n): # use absolute percentage error
    df_volpred = volpred(df, n)
    df_volact = df_volpred.shift(-1)
    df_volerr = (df_volpred - df_volact).iloc[:-1]  # delete the last NaN row
    df_volpred.columns = tickers
    df_volact.columns = tickers
    df_volerr.columns = tickers
    df_volAPE = (df_volerr.div(df_volact.iloc[:-1])).abs() # abs((actual-pred)/actual) 
    df_volAPE.columns=pd.MultiIndex.from_product([['APE (Volatility)'], df_volAPE.columns])
    return df_volAPE

def volAMPE(df,n): # use absolute mean percentage error
    df_volpred = volpred(df, n)
    df_volact = df_volpred.shift(-1)
    df_volerr = (df_volpred - df_volact).iloc[:-1]  # delete the last NaN row
    df_volpred.columns = tickers
    df_volact.columns = tickers
    df_volerr.columns = tickers
    avg = (df_volact.iloc[:-1] + df_volpred.iloc[:-1]) / 2   # exclude the last NaN rows
    df_volAMPE = (df_volerr.div(avg)).abs() # abs((actual-pred)/avg)
    df_volAMPE.columns=pd.MultiIndex.from_product([['AMPE (Volatility)'], df_volAMPE.columns])
    return df_volAMPE

def volLAAR(df,n): # use log of absolute value of accuracy ratio as error
    df_volpred = volpred(df, n)
    df_volact = df_volpred.shift(-1)
    df_volpred.columns = tickers
    df_volact.columns = tickers
    df_volLAAR = np.log((df_volpred.iloc[:-1].div(df_volact.iloc[:-1])).abs()) # log(abs(pred/act)) 
    df_volLAAR.columns=pd.MultiIndex.from_product([['LAAR (Volatility)'], df_volLAAR.columns])
    return df_volLAAR


# ## Mean

# In[7]:


# Define functions for actual/predicted mean and the absolute error between them

def meanpred(df, n):
    return (df.rolling(n).mean()[n-1::n])*260 # change to annualized mean

def meanerr(df, n):
    df_meanpred = meanpred(df, n)
    df_meanerr = (df_meanpred.shift(-1) - df_meanpred).rename(columns={'Logret': 'Residual Error (Mean)'})
    df_meanerr = df_meanerr.iloc[:-1] # delete the last NaN row
    return df_meanerr 

def meanabserr(df, n):
    df_meanpred = meanpred(df, n)
    df_meanabserr = (df_meanpred.shift(-1) - df_meanpred).abs().rename(columns={'Logret': 'Absolute Error (Mean)'})
    df_meanabserr = df_meanabserr.iloc[:-1] # delete the last NaN row
    return df_meanabserr


# In[8]:


# Define error functions, relative errors, taking size of error into account

def meanAPE(df,n): # use absolute percentage error
    df_meanpred = meanpred(df, n)
    df_meanact = df_meanpred.shift(-1)
    df_meanerr = (df_meanpred - df_meanact).iloc[:-1]  # delete the last NaN row
    df_meanpred.columns = tickers
    df_meanact.columns = tickers
    df_meanerr.columns = tickers
    df_meanAPE = (df_meanerr.div(df_meanact.iloc[:-1])).abs() # abs((actual-pred)/actual) 
    df_meanAPE.columns=pd.MultiIndex.from_product([['APE (Mean)'], df_meanAPE.columns])
    return df_meanAPE

def meanAMPE(df,n): # use absolute mean percentage error
    df_meanpred = meanpred(df, n)
    df_meanact = df_meanpred.shift(-1)
    df_meanerr = (df_meanpred - df_meanact).iloc[:-1]  # delete the last NaN row
    df_meanpred.columns = tickers
    df_meanact.columns = tickers
    df_meanerr.columns = tickers
    avg = (df_meanact.iloc[:-1] + df_meanpred.iloc[:-1]) / 2   # exclude the last NaN rows
    df_meanAMPE = (df_meanerr.div(avg)).abs() # abs((actual-pred)/avg)
    df_meanAMPE.columns=pd.MultiIndex.from_product([['AMPE (Mean)'], df_meanAMPE.columns])
    return df_meanAMPE

def meanLAAR(df,n): # use log of absolute value of accuracy ratio as error
    df_meanpred = meanpred(df, n)
    df_meanact = df_meanpred.shift(-1)
    df_meanpred.columns = tickers
    df_meanact.columns = tickers
    df_meanLAAR = np.log((df_meanpred.iloc[:-1].div(df_meanact.iloc[:-1])).abs()) # log(abs(pred/act)) 
    df_meanLAAR.columns=pd.MultiIndex.from_product([['LAAR (Mean)'], df_meanLAAR.columns])
    return df_meanLAAR


# In[9]:


# define functions for plotting histogram and the numerical results
# given the error functions and a fixed timewindow
def plothist(function, tw):
    myerr = function(logret, tw)
    fig = plt.figure(figsize = (15,10))
    ax = fig.gca()
    errplot = myerr.hist(ax = ax)
    errvalue = print(myerr)
    return errvalue, errplot


# ### Iterate through the time window

# In[10]:


# # define functions for printing the numerical results and plotting the dot graphs
# given the error functions, starting&ending timewindow with step size, tail

# for volatility
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
    j = 0
    tw_inf = [] # mark the timewindows for which the errors include +/-inf 
    tw_null = [] # mark the timewindows for which the errors include NaN 
    
    for i in range(tw1, tw2+step, step):
        myerr = function(logret, i)
        
        # check infinity problem
        if myerr.isin([np.inf, -np.inf]).values.any() == True :
            tw_inf.append(i)
        else :
            tw_inf = tw_inf            
        # check NaN problem
        if myerr.isnull().values.any() == True :
            tw_null.append(i)
        else :
            tw_null = tw_null
            
        # create df 'outlier' consists of T/F, T if abs err>tail
        outlier = myerr > tail
        # fill in values representing the percentage of abs error outside tail
        # outlier.mean() calculates percentage of T values 
        percent.iloc[:,j] = outlier.mean().reset_index(drop=True) 
        j = j+1
    
    percent_T = percent.transpose() # exchange x and y axis for plotting purpose
    percent_T.columns = tickers # rename the columns as the securities
    dotvalue = print(percent_T)
    msg_inf = print(f"Inf error exists when time window = {tw_inf}")
    msg_null = print(f"NaN exists when time window = {tw_null}")
    
    # plot the dot chart
    dotplot = percent_T.plot(ls = '', marker = '.', figsize = (10,5), title = f"Volatility, Percentage of error observations outside tail={tail}, {str(function.__name__).strip('vol')}")
    dotplot.legend(bbox_to_anchor = (1.2, 1))
    return msg_inf, msg_null, dotvalue, dotplot

# for mean
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
    j = 0
    tw_inf = [] # mark the timewindows for which the errors include +/-inf
    tw_null = [] # mark the timewindows for which the errors include NaN
    
    for i in range(tw1, tw2+step, step):
        myerr = function(logret, i)
        
        # check infinity problem
        if myerr.isin([np.inf, -np.inf]).values.any() == True :
            tw_inf.append(i)
        else :
            tw_inf = tw_inf
        # check NaN problem
        if myerr.isnull().values.any() == True :
            tw_null.append(i)
        else :
            tw_null = tw_null

        # create df 'outlier' consists of T/F, T if abs err>tail
        outlier = myerr > tail
        # fill in values representing the percentage of abs error outside tail
        # outlier.mean() calculates percentage of T values 
        percent.iloc[:,j] = outlier.mean().reset_index(drop=True) 
        j = j+1
    
    percent_T = percent.transpose() # exchange x and y axis for plotting purpose
    percent_T.columns = tickers # rename the columns as the securities
    dotvalue = print(percent_T)
    msg_inf = print(f"Inf error exists when time window = {tw_inf}")
    msg_null = print(f"NaN exists when time window = {tw_null}")
    
    # plot the dot chart
    dotplot = percent_T.plot(ls = '', marker = '.', figsize = (10,5), title = f"Mean, Percentage of error observations outside tail={tail}, {str(function.__name__).strip('mean')}")
    dotplot.legend(bbox_to_anchor = (1.2, 1))
    return msg_inf, msg_null, dotvalue, dotplot


# ## Compute results for volatility & mean

# #### APE

# In[11]:


# histogram, tw = 70
plothist(volAPE, 70)
plothist(meanAPE, 70)


# In[12]:


# dot plot, volatility, tail = 10%
plotdot_vol(volAPE, 5, 150, 5, 0.1)


# In[13]:


# dot plot, volatility, tail = 10%
plotdot_vol(volAPE, 5, 150, 5, 0.5)


# In[14]:


# dot plot, volatility, tail = 0, 100 (for checking)
plotdot_vol(volAPE, 5, 150, 5, 0)
plotdot_vol(volAPE, 5, 150, 5, 100)


# In[15]:


# dot plot, mean, tail = 10%
plotdot_mean(meanAPE, 5, 150, 5, 0.1)


# In[16]:


# dot plot, mean, tail = 50%
plotdot_mean(meanAPE, 5, 150, 5, 0.5)


# In[17]:


# dot plot, mean, tail = 0, 100 (for checking)
plotdot_mean(meanAPE, 5, 150, 5, 0)
plotdot_mean(meanAPE, 5, 150, 5, 100)


# In[18]:


# dot plot, mean, tail = inf (for checking)
plotdot_mean(meanAPE, 5, 150, 5, math.inf)


# #### AMPE

# In[19]:


# histogram, tw = 70
plothist(volAMPE, 70)
plothist(meanAMPE, 70)


# In[20]:


# dot plot, volatility, tail = 10%
plotdot_vol(volAMPE, 5, 150, 5, 0.1)


# In[21]:


# dot plot, volatility, tail = 50%
plotdot_vol(volAMPE, 5, 150, 5, 0.5)


# In[22]:


# dot plot, volatility, tail = 0, 100 (for checking)
plotdot_vol(volAMPE, 5, 150, 5, 0)
plotdot_vol(volAMPE, 5, 150, 5, 100)


# In[23]:


# dot plot, mean, tail = 10%
plotdot_mean(meanAMPE, 5, 150, 5, 0.1)


# In[24]:


# dot plot, mean, tail = 50%
plotdot_mean(meanAMPE, 5, 150, 5, 0.5)


# In[25]:


# dot plot, mean, tail = 0, 100 (for checking)
plotdot_mean(meanAMPE, 5, 150, 5, 0)
plotdot_mean(meanAMPE, 5, 150, 5, 100)


# In[26]:


# dot plot, mean, tail = inf (for checking)
plotdot_mean(meanAMPE, 5, 150, 5, math.inf)


# #### LAAR 

# In[27]:


# histogram, tw = 70
plothist(volLAAR, 70)
plothist(meanLAAR, 70)


# In[28]:


# dot plot, volatility, tail = e^0.9
plotdot_vol(volLAAR, 5, 150, 5, math.e**0.9)


# In[29]:


# dot plot, volatility, tail = e^0.5
plotdot_vol(volLAAR, 5, 150, 5, math.e**0.5)


# In[30]:


# dot plot, volatility, tail = +/-inf (for checking)
plotdot_vol(volLAAR, 5, 150, 5, math.inf)
plotdot_vol(volLAAR, 5, 150, 5, -math.inf)


# In[31]:


# dot plot, mean, tail = e^0.9
plotdot_mean(meanLAAR, 5, 150, 5, math.e**0.9)


# In[32]:


# dot plot, volatility, tail = e^0.5
plotdot_mean(meanLAAR, 5, 150, 5, math.e**0.5)


# In[33]:


# dot plot, volatility, tail = +/-inf (for checking)
plotdot_mean(meanLAAR, 5, 150, 5, math.inf)
plotdot_mean(meanLAAR, 5, 150, 5, -math.inf)

