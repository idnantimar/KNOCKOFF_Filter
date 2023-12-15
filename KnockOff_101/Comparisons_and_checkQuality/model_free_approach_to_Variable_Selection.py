"""
Created on Wed Dec 13 23:40:34 2023

Topic: @author: R.Nandi
"""

from ..Basics import *

#### Model-Free approach to variable selection (for Categorical Response) =====
'''
  **  Usually variable selection methods like- LASSO,Multiple_Testing etc
      rely on the coefficients in some parametric model of Y|X=x.
  ** But for Categorical Response case , a variable is consedered as unimportant when
         X_j|Y=y identically distributed for all y.
     Hence, Nonparametric tests for homogeneity can be applied for variable selection.
  ** Specially works well for misspecified models.
  ** Though it does mot take into account the inter-relation between predictors.

  NOTE: Due to Bonferroni correction for multiple(>2) categories, can lead to very small power.
        But it does not require any CrossValidation or MCMC, so it performs very fast.


'''

from itertools import combinations
from scipy.stats import ks_2samp,contingency
from scipy.stats import entropy


#> ............................................................

def _testHomogeneity_Categorical(x,y,num_resamples=200,seed_resamples=None):
    # based on permutation test of Jensen-Shannon divergence
    n1,n2 = len(x),len(y)
    n = n1+n2
    pooled_data = pd.concat([x,y])
    categories_ = pooled_data.cat.categories
    def JS_Div(a,b):
        a = pd.get_dummies(a).sum(axis=0)
        b = pd.get_dummies(b).sum(axis=0)
        a /= sum(a)
        b /= sum(b)
        m = (a+b)/2
        JSD = (entropy(a,m)+entropy(b,m))/2
        return JSD

    T_obs = JS_Div(x,y)
    generator = RNG(seed_resamples)
    def T_i(i):
        resampled_data = generator.choice(pooled_data,n,replace=False)
        resampled_data = pd.Categorical(resampled_data,categories=categories_)
        x_,y_ = resampled_data[:n1],resampled_data[n1:]
        return JS_Div(x_,y_)
    T_simulated = map(T_i,range(num_resamples))
    T_simulated = np.array(list(T_simulated))

    return (T_simulated>=T_obs).mean()
# .............................................................




def modelfree_SelectXj(y,X,is_Cat,level=0.05,
                       case_continuous_=lambda a,b: ks_2samp(a,b).pvalue,
                       case_categorical_=lambda a,b: _testHomogeneity_Categorical(a,b,num_resamples=200,seed_resamples=None)):
    """
    A model-free approach for variable selection, when Response is Categorical.

    Parameters
    ----------
    y : Series or 1D-array ; length=n
        The Categorical Response.
    X : DataFrame or 2D-array ; shape=(n,p)
        The DataMatrix. Each column represents a feature, can be Numerical or Categorical as well.
    is_Cat : list or array of True/False values ; length=p
        Each element determines whether the corresponding column of X is of Categorical(True) or Numerical(False) type.
    level : float ; default 0.05
        The level of significance for testing each column.
    case_continuous_,case_categorical_ : a function that returns p-value
        Functions to be used to test the importance of each column.
    generator : random number generator ; default np.random.default_rng(seed=None).

    Returns
    -------
    Indicator of Selection (True= selected,False= rejected).

    """
    y = pd.get_dummies(y)
    y = np.array(y,dtype=bool)
    _,k = y.shape
    all_pairs = list(combinations(range(k),2))
    X = pd.DataFrame(X).copy()
    n,p = X.shape
    col_ix = pd.Series(range(p))
    names = X.columns
    is_Cat = np.array(is_Cat)
    level /= k*(k-1)/2
    X[names[is_Cat]] = X[names[is_Cat]].astype('category')

    def for_Xj(j):
        Xj = X.iloc[:,j]
        for Group1,Group2 in all_pairs :
            Group1 = Xj[y[:,Group1]]
            Group2 = Xj[y[:,Group2]]
            p_val = case_categorical_(Group1,Group2) if is_Cat[j] else case_continuous_(Group1,Group2)
            if (p_val<level) : break
        return (p_val<level)
    DECISION = list(map(for_Xj,range(p)))

    return np.array(DECISION)




# *****************************************************************************



