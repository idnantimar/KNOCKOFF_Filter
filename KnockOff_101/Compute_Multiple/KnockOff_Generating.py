"""
Created on Sun Sep  3 23:47:24 2023

Topic: Generating KnockOff copy of DataMatrix
@author: R.Nandi
"""

from ..Basics import *
from sklearnex import patch_sklearn
patch_sklearn(verbose=0)


#### Sequential KnockOff generation ===========================================
'''
 ** Generates KnockOff copy of data matrix X ,
    treating X as a random observtion from an underlying population
 ** For j=1,2,...,p this method sequentially generates
        Xj_knockoff ~ Xj|(X[-j],X_knockoff[1:j-1])
 ** For Continuous Xj this conditional distn. is fitted as Normal
      Categorical Xj this conditional distn. is fitte as Multinomial
    to estimate the underlying parameters
 ** The restriction 'Xj & Xj_knockoff should be as uncorrelated as possible' has not implemented explicitly in this method

'''


from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, RepeatedKFold
import warnings
from sklearn.exceptions import ConvergenceWarning



def sKnockOff(X, is_Cat, scaling=False, seed_for_sample=None) :
    """
    Generates KnockOff copy of DataMatrix by 'sequential knockoff' method.

    Parameters
    ----------
    X : DataFrame or 2D-array ; size=(n,p)
        The DataMatrix whose KnockOff copy is required to be generated.

    is_Cat : list or array of True/False values ; length=p
        Each element determines whether the corresponding column of X is of Categorical(True) or Numerical(False) type.

    scaling : bool ; default False
        Whether the numerical columns of X will be standardized before further calculation.

    seed_for_sample : int ; default None
        Seeds of various pseudo-random number generation steps, to be specified for reproducable Output.

    Returns
    -------
    tuple in the form (X,X_knockoff)
        1st element is the DataMatrix (after scaling, if any) ;
        2nd element is the corresponding KnockOff copy

    """
    X = pd.DataFrame(X).copy()
    n,p = X.shape
    idx = X.index
    X.columns = X.columns.astype(str)
    names = X.columns # making sure the col names are string , not int
    is_Cat = np.array(is_Cat)
    X[names[is_Cat]] = X[names[is_Cat]].astype('category')

   ## standardizing continuous columns ------------------------
    if scaling : Scale_Numeric(X,is_Cat)

   ## initialize KnockOff copy --------------------------------
    X_knockoff = pd.DataFrame(index=idx,columns=np.vectorize(lambda name: (name+'_kn.off'))(names))
    X_knock_current = pd.DataFrame(index=idx) # without initializing the shape, the final knockoff copy will be too much fragmented

   ## sequencing over columns ---------------------------------
    generator0 = RNG(seed_for_sample)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="n_components > n_samples", category=UserWarning)
    for j in range(p) :
        name = names[j]
        Xj = X[name] #> response , in the regression model of the conditional distribution   Xj|(X[-j],X_knockoff[1:j-1])
        Xj_type = is_Cat[j]
        Xcombined_j = pd.concat([X.drop(name,axis=1),X_knock_current],axis=1) #> predictors
        Xcombined_j = pd.get_dummies(Xcombined_j,drop_first=True)

        if Xj_type :
             Xcombined_j = Xcombined_j.to_numpy()
            #> fit ........................................
             Model = LogisticRegression(penalty=None)
             Model.fit(Xcombined_j,Xj)
             categories = Model.classes_
             probabilities = Model.predict_proba(Xcombined_j)
            #> new sample .................................
             Xj_copy = generator0.multinomial(1, probabilities).argmax(axis=1)
             Xj_copy = categories[Xj_copy]
        else :
            #> fit ........................................
             Model = LinearRegression()
             Model.fit(Xcombined_j,Xj)
             Xj_copy = Model.predict(Xcombined_j)
             s = np.std(Xj-Xj_copy)
            #> new sample ..................................
             Xj_copy = generator0.normal(Xj_copy,s)

        X_knockoff[name+'_kn.off'] = Xj_copy
        X_knock_current = X_knockoff.iloc[:,:(j+1)]

    warnings.filterwarnings("default", category=ConvergenceWarning)
    warnings.filterwarnings("default", message="n_components > n_samples", category=UserWarning)


   ## KnockOff copy --------------------------------------------
    return tuple([X,X_knockoff])




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#> ...............................................................
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
# ...............................................................


from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import Nystroem



def sKnockOff_KernelTrick(X, is_Cat, scaling=False, seed_for_sample=None, seed_for_CVfolds=None,
                          Kernel_Trick=_RBF_median_heuristic, Nystroem_nComp=100) :
    """
    Generates KnockOff copy of DataMatrix by 'sequential knockoff' method.
    Can use any kind of feature map , by specifying corresponding kernel matrix.

    Parameters
    ----------
    X : DataFrame or 2D-array ; size=(n,p)
        The DataMatrix whose KnockOff copy is required to be generated.

    is_Cat : list or array of True/False values ; length=p
        Each element determines whether the corresponding column of X is of Categorical(True) or Numerical(False) type.

    scaling : bool ; default False
        Whether the numerical columns of X will be standardized before further calculation.

    seed_for_sample, seed_for_CVfolds : int ; default None
        Seeds of various pseudo-random number generation steps, to be specified for reproducable Output.

    Kernel_Trick : a function ; default _RBF_median_heuristic
        This function -
                       * takes the DataMatrix as input
                       * returns a list of the form [K,kern_map] , where K is the gram-matrix , kern_map is the function to compute corresponding kernel map k(u,v) based on two observations u & v

    Nystroem_nComp : int ; default 100
        Dimensionality of the feature space(approximate kernel feature map) used in Logistic regression.

    Returns
    -------
    tuple in the form (X,X_knockoff)
        1st element is the DataMatrix (after scaling, if any) ;
        2nd element is the corresponding KnockOff copy

    """
    X = pd.DataFrame(X).copy()
    n,p = X.shape
    idx = X.index
    X.columns = X.columns.astype(str)
    names = X.columns
    is_Cat = np.array(is_Cat)
    X[names[is_Cat]] = X[names[is_Cat]].astype('category')

   ## standardizing continuous columns ------------------------
    if scaling : Scale_Numeric(X,is_Cat)

   ## initialize KnockOff copy --------------------------------
    X_knockoff = pd.DataFrame(index=idx,columns=np.vectorize(lambda name: (name+'_kn.off'))(names))
    X_knock_current = pd.DataFrame(index=idx)


   ## sequencing over columns ---------------------------------
    generator0 = RNG(seed_for_sample)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="n_components > n_samples", category=UserWarning)
    for j in range(p) :
        name = names[j]
        Xj = X[name]
        Xj_type = is_Cat[j]
        Xcombined_j = pd.concat([X.drop(name,axis=1),X_knock_current],axis=1)
        Xcombined_j = pd.get_dummies(Xcombined_j,drop_first=True)

        K = (Kernel_Trick(Xcombined_j))[int(Xj_type)]

        if Xj_type :
            #> fit ........................................
             Phi = Nystroem(kernel=K,random_state=seed_for_sample,n_components=Nystroem_nComp)
             K = Phi.fit_transform(Xcombined_j)
             if min(Counter(Xj).values())>=3: # crossvalidation is preferred , unless very few observation from a category
                 Model = LogisticRegressionCV(Cs=np.logspace(-4,2,num=5),cv=RepeatedStratifiedKFold(n_repeats=5,n_splits=3,random_state=seed_for_CVfolds))
             else: Model = LogisticRegression(C=0.1)
             Model.fit(K,Xj)
             categories = Model.classes_
             probabilities = Model.predict_proba(K)
            #> new sample .................................
             Xj_copy = generator0.multinomial(1, probabilities).argmax(axis=1)
             Xj_copy = categories[Xj_copy]
        else :
            #> fit ........................................
             Model = GridSearchCV(KernelRidge(kernel='precomputed'),param_grid={'alpha':np.logspace(-1,4,num=5)},cv=RepeatedKFold(n_repeats=5,n_splits=5,random_state=seed_for_CVfolds))
             Model.fit(K,Xj)
             Xj_copy = Model.predict(K)
             s = np.std(Xj-Xj_copy)
            #> new sample ..................................
             Xj_copy = generator0.normal(Xj_copy,s)

        X_knockoff[name+'_kn.off'] = Xj_copy
        X_knock_current = X_knockoff.iloc[:,:(j+1)]

    warnings.filterwarnings("default", category=ConvergenceWarning)
    warnings.filterwarnings("default", message="n_components > n_samples", category=UserWarning)


   ## KnockOff copy --------------------------------------------
    return tuple([X,X_knockoff])




# *****************************************************************************
##
###
####
###
##
# modified Sequential Knockoff ================================================
'''
 ** It appears that as we move from first feature to last feature , we are modelling the conditional distribution Xj|(X[-j],X_knockoff[1:j-1]) based on larger data available. So there can be systematic bias in quality.
 ** To address this problem , split the data in a few blocks , shuffle the order of columns in each block , generate Sequential Knockoff as usual , then reshuffle them back to original order. Finally stack them together as the beginning

'''

from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from ..Basics import _seed_sum



def KnockOff_Reshuffled(X, is_Cat, scaling=False,
                        n_Blocks=3, n_parallel=1, seed_for_randomizing=None,
                        method=lambda Z,z_type,seed: sKnockOff(Z,z_type,seed_for_sample=seed), seed_for_BaseMethod=None) :
    """
    This function splits the data in a few blocks , shuffles the order of columns in each block , generates KnockOff as usual (default method sKnockOff) in each block, then reshuffle them back to original order.

    WARNING: takes much more time than ogiginal KnockOff generating method, but easily parallelizable.

    """
    X = pd.DataFrame(X).copy()
    n,p = X.shape
    idx = X.index
    X.columns = X.columns.astype(str)
    names = X.columns
    is_Cat = np.array(is_Cat)
    X[names[is_Cat]] = X[names[is_Cat]].astype('category')

   ## standardizing continuous columns ------------------------
    if scaling : Scale_Numeric(X,is_Cat)

   ## splitting blocks ----------------------------------------
    Blocks = list(KFold(n_Blocks,shuffle=True,random_state=seed_for_randomizing).split(X))

   ## random shuffle ------------------------------------------
    def Shuffle(Z,rng):
        col_ix = rng.choice(range(p),size=p,replace=False)
        shuffled_Data = Z.iloc[:,col_ix]
        is_Cat_similarly = is_Cat[col_ix]
        return (shuffled_Data, is_Cat_similarly,col_ix)
    def ShuffleBack(Z,Z_knockoff,col_ix):
        inverse_ix = pd.Series(col_ix).sort_values().index
        actualZ = Z.iloc[:,inverse_ix]
        actualZ_knockoff = Z_knockoff.iloc[:,inverse_ix]
        return (actualZ,actualZ_knockoff)


   ## blockwise knockoff generation ---------------------------
    ORIGINALs = []
    KNOCKOFFs = []
    def blockwise_KnockOff(i):
        generator_i = RNG(_seed_sum(seed_for_randomizing,i))
        ix = Blocks[i][1]
        Block = X.iloc[ix]
        Block,is_Cat_i,col_ix = Shuffle(Block,generator_i)
        Block,Block_knockoff = method(Block,is_Cat_i,_seed_sum(seed_for_BaseMethod,i))
        return ShuffleBack(Block,Block_knockoff,col_ix)

    if n_parallel>1 : OUT = (Parallel(n_jobs=n_parallel,backend='loky')(delayed(blockwise_KnockOff)(i) for i in range(n_Blocks)))
    else : OUT = list(map(blockwise_KnockOff,range(n_Blocks)))

    for Block,Block_knockoff in OUT :
        ORIGINALs += [Block]
        KNOCKOFFs += [Block_knockoff]

   ## combining blocks -----------------------------------------
    X = pd.DataFrame(pd.concat(ORIGINALs,axis=0),index=idx)
    X_knockoff = pd.DataFrame(pd.concat(KNOCKOFFs,axis=0),index=idx)
        # we want to recover both row order and column order


   ## KnockOff copy --------------------------------------------
    return tuple([X,X_knockoff])




# *****************************************************************************
