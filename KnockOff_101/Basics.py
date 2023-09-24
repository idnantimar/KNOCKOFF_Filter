"""
Created on Mon Sep  4 21:38:12 2023

Topic: Basic Codes that are needed again and again in many calculations
@author: R.Nandi
"""



## basic modules .............................................
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns




#### Data Type ================================================================
def Cat_or_Num(X):
    """ Determines whether a data column is Categorical or Numerical in a DataFrame
        ( Assuming the 1st entry of a column is not missing )
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
