"""
Created on Tue Sep 19 23:16:49 2023

Topic: Checking the quality of knockoff
@author: R.Nandi
"""

from Basics import *


#### Diagnostic of the quality of knockoff ====================================
'''
 ** NOTE: For Numerical data only
'''

from sklearn.metrics.pairwise import pairwise_kernels
import random


def kernel_Z1Z2(Z1,Z2=None,metric_type='rbf'):
    #  estimates E[k(Z1,Z2)] = <E[Phi(Z1)],E[Phi(Z2)]>
    # Inputs are two DataMatrix corresponding to Z1 & Z2 observations respectively
    D= pairwise_kernels(Z1,Z2,metric_type)
    np.fill_diagonal(D,0)
    return np.mean(D)


MMD_score = lambda P1,P2,kernel_type : kernel_Z1Z2(P1,metric_type=kernel_type) + kernel_Z1Z2(P2,metric_type=kernel_type) - 2*kernel_Z1Z2(P1,P2,metric_type=kernel_type)



def MMD_checkQuality(X,X_knockoff,n_partialSwap=20,set_seed=None,kernel_type='rbf'):
    """
    let LHS = [X,X_knockoff]
        RHS = anySwap(LHS)
        kernel_Z1Z2 = estimated_<E[Phi(Z1)],E[Phi(Z2)]> , where Z1 & Z2 are two independent random variables

    If the knockoff copy is of good quality, LHS & RHS should be identically distributed.
        MMD_score = kernel_Z1Z2(LHS,LHS) + kernel_Z1Z2(RHS,RHS) - 2*kernel_Z1Z2(LHS,RHS)  , should be near 0 in such case. The further this score from 0 , it says the knockoff fails to mimic the original data.

    NOTE: i-th row of LHS and j-th row of RHS should be independent, unless i=j

    """
    n,p = X.shape
    LHS = pd.concat([X,X_knockoff],axis=1)
    fullSwap = lambda : pd.concat([X_knockoff,X],axis=1)
    def partialSwap():
        col_ix = np.array(range(2*p))
        swappable = np.random.choice(range(p),size=random.randint(1,p),replace=False)
        col_ix[swappable] += p
        col_ix[(swappable+p)] -= p
        return LHS.iloc[:,col_ix]

    score = [MMD_score(LHS,fullSwap(),kernel_type)]
    random.seed(set_seed)
    np.random.seed(set_seed)
    for _ in range(n_partialSwap):
        score += [MMD_score(LHS,partialSwap(),kernel_type)]

    return np.mean(score)




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def pairwise_absCorrelation(X,X_knockoff):
    """
    Computes correlation coefficient between Xj & Xj_knockoff
    Ideally it should be near 0
    """
    n,p = X.shape
    names = X.columns
    pairCor = np.corrcoef(X,X_knockoff,rowvar=False)
    pairCor = np.diag(pairCor[p:,:p])

    return pd.Series(np.abs(pairCor),index=names)



# *****************************************************************************
