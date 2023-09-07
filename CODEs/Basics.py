"""
Created on Mon Sep  4 21:38:12 2023

Topic: Basic Codes that are needed again and again in many calculations
@author: R.Nandi
"""



## basic modules .............................................
import pandas as pd
import numpy as np
from collections import Counter
from numpy.random import choice,normal,multinomial,multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns




#### Data Type ================================================================
def Cat_or_Num(X):
    """ Determines whether a data column is Categorical or Numerical in a DataFrame
        Assuming the 1st entry of a column is not missing
        Output is an array of size=number_of_columns ; True = Categorical , False = Numerical
    """
    n_col=X.shape[1]
    numericType = np.vectorize(lambda pos: isinstance(X.iloc[0,pos], (np.integer,np.floating)))
    return np.invert(numericType(range(n_col)))


# *****************************************************************************
##
###
####
###
##
#
#### Scaling Data =============================================================
def Scale_Numeric(X,is_Cat):
    """ Scales the numerical columns of a data matrix , skips the categorical columns as it is.
        Do not return any new copy , but change the DataFrame in place.
    """
    x = X.iloc[:,np.invert(is_Cat)]
    X.iloc[:,np.invert(is_Cat)] = StandardScaler().fit_transform(x)


# *****************************************************************************
##
###
####
###
##
#
#### Simulate Samples =========================================================

def simulateIndependent(n_obs,col_type,NUM=lambda size: normal(0,1,size),CAT=lambda size: choice(['A','B','C','D'],size,replace=True)):
    """
    A function to simulate a DataMatrix , where each column is sampled from some population , independent of each other.

    Parameters
    ----------
    n_obs : int
        Number of observations per column.
    col_type : tuple as (N,C) or array as [True,False,True,True,True,...]
        * If tuple, 1st element N denotes number of Numerical columns , 2nd element C denotes number of Categorical columns. When only one kind of columns are needed, input the other element as 0
        * If array as [True,False,...] , True denotes which position is for Numerical column, False denotes which position is for Categorical column.
    NUM : a function ;  default is lambda size: normal(0,1,size)
        Mentions from where to sample Numerical columns.
    CAT : a function ; default is lambda size: choice(['A','B','C','D'],size,replace=True))
        Mentions from where to sample Categorical columns.

    Returns
    -------
    DataFrame ; where each row is an observation , each column is a feature

    """
    if isinstance(col_type,tuple) :
        N,C = col_type
        col_type = [True]*N+[False]*C
    else :
        N,C = sum(col_type),len(col_type)-sum(col_type)

    NUM_Block = [pd.Series(NUM(n_obs)) for _ in range(N)]
    CAT_Block = [pd.Series(CAT(n_obs)) for _ in range(C)]
    BLOCKs = pd.concat(NUM_Block+CAT_Block,axis=1)
    BLOCKs.rename(columns={**{i:('Num'+str(i)) for i in range(N)},**{(N+j):('Cat'+str(j)) for j in range(C)}},inplace=True)
    col_order = []
    i,j,k = 0,0,0
    tot = N+C
    while k<tot:
        if col_type[k]:
            col_order += ['Num'+str(i)]
            i += 1
        else :
            col_order += ['Cat'+str(j)]
            j += 1
        k = i+j

    return BLOCKs[col_order]



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def simulateJoint(n_obs,popln=lambda size: multivariate_normal([0,0], [[1, 0], [0, 1]],size)):
    """
    A function to simulate a DataMatrix from a joint-distribution specified.

    Parameters
    ----------
    n_obs : int
        Number of observations per column.
    popln : a function mentioning the joint distribution of the columns ; default is lambda size: multivariate_normal([0,0], [[1, 0], [0, 1]],size)

    Returns
    -------
    DataFrame ; where each row is an observation , each column is a feature

    """

    OUT = pd.DataFrame(popln(n_obs))
    cols = OUT.columns
    OUT.rename(columns={i:'col_'+str(i) for i in cols},inplace=True)

    return OUT



# *****************************************************************************
