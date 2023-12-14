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

    p_val_,score_ = energy_test(LHS,fullSwap_RHS,num_resamples=num_resamples)
    p_val = [p_val_]
    score = [score_]
    generator0 = RNG(set_seed)
    for itr in range(n_partialSwap):
        partialSwap_RHS = partialSwap(generator0)
        p_val_,score_ = energy_test(LHS,partialSwap_RHS,num_resamples=num_resamples)
        p_val += [p_val_]
        score += [score_]

    return (np.mean(score),np.median(p_val))




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

