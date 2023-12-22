"""
Created on Tue Sep 19 23:16:49 2023

Topic: Checking the quality of knockoff and Comparisons with various other methods.
@author: R.Nandi
"""

from ..Basics import *


#### Diagnostic of the quality of knockoff (NUMERICAL DATA)====================

from dcor.homogeneity import energy_test,energy_distance


def EnergyDistance_test(X1st,X1st_knockoff,X2nd,X2nd_knockoff,n_partialSwap=10,set_seed=None,num_resamples=100):
    """
    let
       * LHS = [X1st,X1st_knockoff]
       * RHS = anySwap([X2nd,X2nd_knockoff])
    If the knockoff copy is of good quality, LHS & RHS should be identically distributed.

    returns average test statistics and p-values over fullSwap & n_partialSwap.
    """
    _,p = X1st.shape
    LHS = pd.concat([X1st,X1st_knockoff],axis=1)
    fullSwap_RHS = pd.concat([X2nd_knockoff,X2nd],axis=1)
    def partialSwap(rng):
        col_ix = np.array(range(2*p))
        swappable = rng.choice(range(p),size=rng.integers(1,p,endpoint=True),replace=False)
        col_ix[swappable] += p
        col_ix[(swappable+p)] -= p
        return fullSwap_RHS.iloc[:,col_ix]

    def score_currentRHS(RHS,rng):
        p_val,score = energy_test(LHS,RHS,num_resamples=num_resamples,random_state=rng)
        return (p_val,score)
    generator0 = RNG(set_seed)
    result = [score_currentRHS(fullSwap_RHS,generator0)]
    result += list(map(lambda j: score_currentRHS(partialSwap(generator0),generator0),range(n_partialSwap)))
    result = np.array(result)
    return (result[:,1].mean(),np.median(result[:,0]))




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
