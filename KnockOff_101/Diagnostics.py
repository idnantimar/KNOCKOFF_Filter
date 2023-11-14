"""
Created on Tue Sep 19 23:16:49 2023
Topic: Checking the quality of knockoff (For Numerical data only)
@author: R.Nandi
"""

from .Basics import *
#### Diagnostic of the quality of knockoff ====================================


def _E_kernel_Z1Z2(D):
    # estimates E[k(Z1,Z2)] = <E[Phi(Z1)],E[Phi(Z2)]>
    # Input is the gram-matrix
    if (D==D.T).all():
        np.fill_diagonal(D,0)
        n = D.shape[0]
        out = np.mean(D)*n/(n-1)
    else : out = np.mean(D)
    return out



from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import euclidean,pdist,squareform
from scipy.stats import norm as normal_distribution

def _RBF_median_heuristic(Z1,Z2=None,k=1):
    # RBF kernel with scale=k*(median of observed distances)
    if Z2 is None : D = pdist(Z1)
    else : D = pdist(np.vstack((Z1,Z2)))
    sd = k*np.median(D)
    D = squareform(D)
    if Z2 is not None :
        n = len(Z1)
        D11 = D[:n,:n]
        D12 = D[:n,n:]
        D22 = D[n:,n:]
        K = (normal_distribution.pdf(D11,scale=sd),
             normal_distribution.pdf(D12,scale=sd),
             normal_distribution.pdf(D22,scale=sd))
    else : K = normal_distribution.pdf(D,scale=sd)
    kern_map = lambda z2,z1 : normal_distribution.pdf(euclidean(z1,z2),scale=sd)
    return [K,kern_map,sd]



import random
from sklearn.metrics.pairwise import pairwise_kernels

def MMD_checkQuality(X1st,X1st_knockoff,X2nd,X2nd_knockoff,n_partialSwap=10,set_seed=None):
    """
    let
       * LHS = [X1st,X1st_knockoff]
       * RHS = anySwap([X2nd,X2nd_knockoff])
       * E_kernel_Z1Z2 = estimated_<E[Phi(Z1)],E[Phi(Z2)]> , where Z1 & Z2 are two independent random variables , Phi() some feature map
    If the knockoff copy is of good quality, LHS & RHS should be identically distributed.
        MMD_score = E_kernel_Z1Z2(LHS,LHS) + E_kernel_Z1Z2(RHS,RHS) - 2*E_kernel_Z1Z2(LHS,RHS)  , should be near 0 in such case. The further this score exceeds 0 , it says the knockoff fails to mimic the original data.
    (If we could actually compute the expectations E[Phi(Zi)] , we must get MMD_score>=0 . But due to sample estimates , this score can be -ve sometimes.)
    NOTE: To make the observations in LHS & RHS independent , we need -
                * X1st & X2nd two independent DataMatrix , though the observations are identically distributed
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
    K11,K12,K22 = _RBF_median_heuristic(LHS,fullSwap_RHS)[0]
    score = [_E_kernel_Z1Z2(K11)
             + _E_kernel_Z1Z2(K22)
             - 2*_E_kernel_Z1Z2(K12)]
    random.seed(set_seed)
    np.random.seed(set_seed)
    for _ in range(n_partialSwap):
        partialSwap_RHS = partialSwap()
        K11,K12,K22 = _RBF_median_heuristic(LHS,partialSwap_RHS)[0]
        score += [_E_kernel_Z1Z2(K11)
                 + _E_kernel_Z1Z2(K22)
                 - 2*_E_kernel_Z1Z2(K12)]
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
