"""
Created on Sun Sep  3 23:47:24 2023

Topic: Generating KnockOff copy of data matrix
@author: R.Nandi
"""

from Basics import *


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

from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import GridSearchCV ,RepeatedStratifiedKFold,RepeatedKFold
import warnings
from sklearn.exceptions import ConvergenceWarning


def sKnockOff(X, is_Cat, scaling=False, seed_for_sample=None, seed_for_KernelTrick=None, seed_for_CV=None, Kernel_nComp=100) :
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

    seed_for_sample, seed_for_KernelTrick, seed_for_CV : int ; default None
        Seeds of various pseudo-random number generation steps, to be specified for reproducable Output.

    Kernel_nComp : int ; default 100
        Dimensionality of the feature space(approximate RBF kernel feature map) used in regression.

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

   ## standardizing continuous columns ------------------------
    if scaling : Scale_Numeric(X,is_Cat)

   ## initialize KnockOff copy --------------------------------
    X_knockoff = pd.DataFrame(index=idx)

   ## kernel trick --------------------------------------------
    rbf_sampler = RBFSampler(gamma='scale',random_state=seed_for_KernelTrick,n_components=Kernel_nComp)

   ## sequencing over columns ---------------------------------
    np.random.seed(seed_for_sample)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for j in range(p) :
        name = names[j]
        Xj = X[name] # response , in the regression model of the conditional distribution   Xj|(X[-j],X_knockoff[1:j-1])
        Xcombined_j = pd.concat([X.drop(name,axis=1),X_knockoff],axis=1) # predictors
        current_isCat = Cat_or_Num(Xcombined_j)
        Xcombined_jKernel = rbf_sampler.fit_transform(Xcombined_j.iloc[:,np.invert(current_isCat)]) if (not current_isCat.all()) else Xcombined_j.iloc[:,np.invert(current_isCat)]
        # kernel trick on numerical columns
        Xcombined_jCat = Xcombined_j.iloc[:,current_isCat] # categorical columns

        Xcombined_j = pd.get_dummies(pd.concat([pd.DataFrame(Xcombined_jKernel,index=idx),Xcombined_jCat],axis=1),drop_first=True)
        Xcombined_j.columns = Xcombined_j.columns.astype(str)

        if is_Cat[j] :
            #> fit ........................................
             Model = LogisticRegression()
             CV_type = RepeatedStratifiedKFold(n_repeats=5,n_splits=3,random_state=seed_for_CV) if min(Counter(Xj).values())>=3 else RepeatedKFold(n_repeats=5,n_splits=3,random_state=seed_for_CV) # stratified K-fold is preferred unless not enough observations available per class
             Model = (GridSearchCV( Model, param_grid={'C':[0.1,0.4,1,2.5,10]}, scoring='accuracy',cv=CV_type)).fit(Xcombined_j,Xj)
             categories = Model.classes_
             probabilities = pd.DataFrame(Model.predict_proba(Xcombined_j),index=idx)
            #> new sample .................................
             Xj_copy = probabilities.apply(lambda x : np.random.multinomial(1,x), axis=1,result_type='expand').idxmax(axis=1)
             Xj_copy = categories[Xj_copy]

        else :
            #> fit ........................................
             Model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],random_state=seed_for_CV)
             Model.fit(Xcombined_j,Xj)
             Xj_copy = Model.predict(Xcombined_j)
             s = np.std(Xj-Xj_copy)
            #> new sample ..................................
             Xj_copy = np.random.normal(Xj_copy,s)

        X_knockoff[name+'_kn.off'] = Xj_copy

    warnings.filterwarnings("default", category=ConvergenceWarning)

   ## KnockOff copy --------------------------------------------
    return tuple([X,X_knockoff])



# modified Sequential Knockoff ++++++++++++++++++++++++++++++++++++++++++++++++
'''
 ** It appears that as we move from first feature to last feature , we are modelling the conditional distribution Xj|(X[-j],X_knockoff[1:j-1]) based on larger data available. So there can be systematic bias in quality.
 ** To address this problem , split the data in a few blocks , shuffle the order of columns in each block , generate Sequential Knockoff as usual , then reshuffle them back to original order. Finally stack them together as the beginning

'''

from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from multiprocessing import cpu_count


def sKnockOff_Modified(X, is_Cat, scaling=False, n_Blocks=3, compute_parallel=False, seed_for_randomizing=None, seed_for_sample=None, seed_for_KernelTrick=None, seed_for_CV=None, Kernel_nComp=100) :
    """
    This function splits the data in a few blocks , shuffle the order of columns in each block , generate Sequential KnockOff as usual in each block, then reshuffle them back to original order.

    WARNING: takes too much time than ogiginal sKnockOff method.

    """
    X = pd.DataFrame(X).copy()
    n,p = X.shape
    idx = X.index
    X.columns = X.columns.astype(str)
    names = X.columns
    names_knockoff = np.vectorize(lambda name: (name+'_kn.off'))(names)

   ## standardizing continuous columns ------------------------
    if scaling : Scale_Numeric(X,is_Cat)

   ## splitting blocks ----------------------------------------
    Blocks = list(KFold(n_Blocks,shuffle=True,random_state=seed_for_randomizing).split(X))

   ## random shuffle ------------------------------------------
    np.random.seed(seed_for_randomizing)
    def Shuffle(Z):
        S = np.random.choice(range(p),size=p,replace=False)
        shuffled_Data = Z.iloc[:,S]
        is_Cat_similarly = list(pd.Series(is_Cat)[S])
        return (shuffled_Data, is_Cat_similarly)
    def ShuffleBack(Z,Z_knockoff):
        actualZ = Z[names]
        actualZ_knockoff = Z_knockoff[names_knockoff]
        return (actualZ,actualZ_knockoff)

   ## blockwise knockoff generation ---------------------------
    ORIGINALs = []
    KNOCKOFFs = []
    def blockwise_KnockOff(i):
        ix = Blocks[i][1]
        Block = X.iloc[ix]
        Block,is_Cat_i = Shuffle(Block)
        Block,Block_knockoff = sKnockOff(Block, is_Cat_i, False, seed_for_sample, seed_for_KernelTrick, seed_for_CV, Kernel_nComp)
        return ShuffleBack(Block, Block_knockoff)


    if compute_parallel : OUT = (Parallel(n_jobs=cpu_count())(delayed(blockwise_KnockOff)(i) for i in range(n_Blocks)))
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
