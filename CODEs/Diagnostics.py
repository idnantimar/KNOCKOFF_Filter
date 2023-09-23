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
    if Z2 is None :
        np.fill_diagonal(D,0)
        n = D.shape[0]
        out = np.mean(D)*n/(n-1)
    else : out = np.mean(D)
    return out


MMD_score = lambda P1,P2,kernel_type : kernel_Z1Z2(P1,metric_type=kernel_type) + kernel_Z1Z2(P2,metric_type=kernel_type) - 2*kernel_Z1Z2(P1,P2,metric_type=kernel_type)



def MMD_checkQuality(X1st,X1st_knockoff,X2nd,X2nd_knockoff,n_partialSwap=20,set_seed=None,kernel_type='rbf'):
    """
    let 
       * LHS = [X1st,X1st_knockoff]
       * RHS = anySwap([X2nd,X2nd_knockoff])
       * kernel_Z1Z2 = estimated_<E[Phi(Z1)],E[Phi(Z2)]> , where Z1 & Z2 are two independent random variables , Phi() some feature map

    If the knockoff copy is of good quality, LHS & RHS should be identically distributed.
        MMD_score = kernel_Z1Z2(LHS,LHS) + kernel_Z1Z2(RHS,RHS) - 2*kernel_Z1Z2(LHS,RHS)  , should be near 0 in such case. The further this score exceeds 0 , it says the knockoff fails to mimic the original data.
    (If we could actually compute the expectations E[Phi(Zi)] , we must get MMD_score>=0 . But due to sample estimates , this score can be -ve sometimes.)

    NOTE: To make the observations in LHS & RHS independent , we need -
                * X1st & X2nd two independent DataMatrix , though the observations are identically distributed, i.e. X1st & X2nd two disjoint partition of the entire DataMatrix X
                * X1st_knockoff is not based on X2nd , similarly X2nd_knockoff is not based on X1st

    """
    p = X1st.shape[1]
    LHS = pd.concat([X1st,X1st_knockoff],axis=1)
    fullSwap_RHS = pd.concat([X2nd_knockoff,X2nd],axis=1)
    def partialSwap():
        col_ix = np.array(range(2*p))
        swappable = np.random.choice(range(p),size=random.randint(1,p),replace=False)
        col_ix[swappable] += p
        col_ix[(swappable+p)] -= p
        return fullSwap_RHS.iloc[:,col_ix]

    score = [MMD_score(LHS,fullSwap_RHS,kernel_type)]
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
