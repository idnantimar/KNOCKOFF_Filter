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


from sklearn.metrics import pairwise_distances
import random


def Dist_Z1Z2(Z1,Z2=None):
    #  estimates E|Z1-Z2| , where Z1 & Z2 are two independent random variables
    # Inputs are two DataMatrix corresponding to Z1 & Z2 observations respectively
    dist = pairwise_distances(Z1,Z2)
    np.fill_diagonal(dist ,0)
    return np.mean(dist)


MMD_score = lambda P1,P2 : Dist_Z1Z2(P1) + Dist_Z1Z2(P2) - 2*Dist_Z1Z2(P1,P2)




def MMD_checkQuality(X,X_knockoff, n_partialSwap = 20, set_seed=None):
    """
    let LHS = [X,X_knockoff]
        RHS = anySwap(LHS)
        Dist_Z1Z2 = estimated_E|Z1-Z2| , where Z1 & Z2 are two independent random variables

    If the knockoff copy is of good quality, LHS & RHS should be identically distributed.
        MMD_score = Dist_Z1Z2(LHS,LHS) + Dist_Z1Z2(RHS,RHS) - 2*Dist_Z1Z2(LHS,RHS)  , should be near 0 in such case. The further this score from 0 , it says the knockoff fails to mimic the original data.
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

    score = [MMD_score(LHS,fullSwap())]
    random.seed(set_seed)
    np.random.seed(set_seed)
    for _ in range(n_partialSwap):
        score += [MMD_score(LHS,partialSwap())]

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
