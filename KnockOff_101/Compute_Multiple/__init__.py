"""
Created on Sun Sep 24 13:33:51 2023

Topic: Generates multiple knockoff copies at a time , or, Computes feature importance based on multiple knockoff copies at atime
       Contains two submodules -
            i. KnockOff_Generating
            ii. Feature_Importance

@author: R.Nandi
"""

from ..Basics import *
from . import KnockOff_Generating,Feature_Importance


#### for parallel computation .....................................
from joblib import Parallel, delayed
from multiprocessing import cpu_count
#### for progress bar .............................................
import contextlib,joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# .................................................................


#### Generating Multiple KnockOff copies at a time ============================

from ..Basics import _seed_sum


def genMulti(X, n_copy, is_Cat,
             method= lambda Z,z_type,seed: KnockOff_Generating.sKnockOff(Z,z_type,seed_for_sample=seed), seed_for_sample=None,
             scaling=True, n_parallel=cpu_count(), shuffle_columns=True, seed_for_shuffle=None):
    """
    Generates multiple KnockOff copies of a same DataMatrix.

    Parameters
    ----------
    X : DataFrame or 2d-array
        The DataMatrix.

    n_copy : int
        Number of copies to be generated.

    is_Cat : list or array of True/False values
        Each element determines whether the corresponding column of X is of Categorical(True) or Numerical(False) type.

    method : A function that creates X_knockoff ; default sKnockOff
        This function should take input -
            * Z : DataMatrix
            * z_type : an array indicating which column is Categorical(True) , which one is Numerical(False)
        & produce output (X,X_knockoff) tuple
            * seed : for reproducible output.

    seed_for_sample : seed to be used in base method for each KnockOff copy.

    scaling : bool ; default True
        Whether the DataMatrix should be standardized before calculations.

    n_parallel : int ; default cpu_count()
        Number of cores used for parallel computing.

    shuffle_columns : bool ; default True
        Whether the columns of DataMatrix should be shuffled before each iteration
        (useful to mitigate some systematic bias due to sequential nature of KnockOff generating algorithm).

    seed_for_shuffle : seed for controling the shuffle of the columns , when shuffle_columns=True.


    Returns
    -------
    list of tuples in the form
        [(X,X_knockoff.1),(X,X_knockoff.2),...,(X,X_knockoff.n_copy)]

    """
    X = pd.DataFrame(X).copy()
    n,p = X.shape
    names = X.columns
    is_Cat = np.array(is_Cat)

    if scaling : Scale_Numeric(X, is_Cat)


    if shuffle_columns :
        def Shuffle(Z):
            col_ix = np.random.choice(range(p),size=p,replace=False)
            shuffled_Data = Z.iloc[:,col_ix]
            is_Cat_similarly = is_Cat[col_ix]
            return (shuffled_Data, is_Cat_similarly,col_ix)
        def ShuffleBack(Z,Z_knockoff,col_ix):
            inverse_ix = pd.Series(col_ix).sort_values().index
            actualZ = Z.iloc[:,inverse_ix]
            actualZ_knockoff = Z_knockoff.iloc[:,inverse_ix]
            return (actualZ,actualZ_knockoff)

        def one_copy(i) :
            np.random.seed(_seed_sum(seed_for_shuffle,i))
            X_,is_Cat_i,col_ix = Shuffle(X)
            X_,X_knockoff_ = method(X_,is_Cat_i,_seed_sum(seed_for_sample,i))
            return ShuffleBack(X_,X_knockoff_,col_ix)
    else :
        def one_copy(i) : return method(X,is_Cat,_seed_sum(seed_for_sample,i))


    with tqdm_joblib(tqdm(desc="Progress_Bar(Generating KnockOff copies...) & expected remaining time", total=n_copy,bar_format="{n}/{total}{unit}|{bar}|{desc}|[{remaining}]")) :
        OUT = Parallel(n_jobs=n_parallel)(delayed(one_copy)(i) for i in range(n_copy))

    return OUT




# *****************************************************************************
##
###
####
###
##
#
#### Scores based on multilple KnockOff copies ================================

def scoreMulti(combinedData, y, FDR=0.1, impStat=Feature_Importance._basicImp_ContinuousResponse, n_parallel=cpu_count()):
    """
    When we have multiple KnockOff copies corresponding to same DataMatrix, based on each of them we can compute one possible feature importance. Goal is to make one overall decision based on them.

    Parameters
    ----------
    combinedData : list of tuples in the form
        [(X,X_knockoff.1),(X,X_knockoff.2),...,(X,X_knockoff.n_copy)]

    y : Series or 1D-array ; for Series index=index_of_data , for array length=number_of_index_in_data
        The response variable. Can be continuous or categorical anything , but impStat should be chosen accordingly.e.g. - for continuous case use impStat=_basicImp_ContinuousResponse , for binary case use impStat=_basicImp_BinaryResponse , for multiple catogory case use impStat=LOFO_ImpCategorical etc.

    FDR : float between [0,1] or list of such float values ; default 0.1
        The False Discovery Rate upperbound to be specified.

    impStat : any function that computes feature importance & threshold for selection ; default _basicImp_ContinuousResponse
        This function should -
            * take input X, X_knockoff , y , FDR
            * produce output as importance statistics corresponding to features , followed by threshold values

    n_parallel : int ; default cpu_count()
        Number of cores used for parallel computing.


    Returns
    -------
    DataFrame of size (number_of_knockoff_copy, number_of_features+number_of_FDR_input)
    where -
        * each row corresponds to one copy,
        * initial columns are feature importance scores corresponding to various features
        * last columns are cut-off values corresponding to FDR control

    """

    lenKnockOff = len(combinedData)

    def score_for_one_copy(i) :
        X_,X_knockoff = combinedData[i]
        return impStat(X_,X_knockoff,y,FDR)

    with tqdm_joblib(tqdm(desc="Progress_Bar(Feature Importance Ordering...) & expected remaining time", total=lenKnockOff,bar_format="{n}/{total}{unit}|{bar}|{desc}|[{remaining}]")) :
        OUT = pd.DataFrame(Parallel(n_jobs=n_parallel)(delayed(score_for_one_copy)(i) for i in range(lenKnockOff)),index=range(lenKnockOff))

    return OUT




# *****************************************************************************

"""
Created on Sun Sep 24 13:33:51 2023

Topic: Generates multiple knockoff copies at a time , or, Computes feature importance based on multiple knockoff copies at atime
       Contains two submodules -
            i. KnockOff_Generating
            ii. Feature_Importance

@author: R.Nandi
"""

from ..Basics import *
from . import KnockOff_Generating,Feature_Importance


#### for parallel computation .....................................
from joblib import Parallel, delayed
from multiprocessing import cpu_count
#### for progress bar .............................................
import contextlib,joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# .................................................................

"""
NOTE: It is hard to obtain reproducable output at this level , instead try to set seed on individual functions , if needed.
"""

#### Generating Multiple KnockOff copies at a time ============================

def genMulti(X, n_copy, is_Cat, method=KnockOff_Generating.sKnockOff, scaling=True, n_parallel=cpu_count()):
    """
    Generates multiple KnockOff copies of a same DataMatrix.

    Parameters
    ----------
    X : DataFrame or 2d-array
        The DataMatrix.

    n_copy : int
        Number of copies to be generated.
        
    is_Cat : list or array of True/False values 
        Each element determines whether the corresponding column of X is of Categorical(True) or Numerical(False) type.

    method : A function that creates X_knockoff ; default sKnockOff
        This function should take input -
            * X : DataMatrix
            * is_Cat : an array indicating which column is Categorical(True) , which one is Numerical(False)
        & produce output (X,X_knockoff) tuple.

    scaling : bool ; default True
        Whether the DataMatrix should be standardized before calculations.

    n_parallel : int ; default cpu_count()
        Number of cores used for parallel computing.

    Returns
    -------
    list of tuples in the form
        [(X,X_knockoff.1),(X,X_knockoff.2),...,(X,X_knockoff.n_copy)]

    """
    X = pd.DataFrame(X).copy()
    n,p = X.shape
    names = X.columns

    if scaling : Scale_Numeric(X, is_Cat)

    def one_copy(i) : return method(X,is_Cat)
 
    with tqdm_joblib(tqdm(desc="Progress_Bar(Generating KnockOff copies...) & expected remaining time", total=n_copy,bar_format="{n}/{total}{unit}|{bar}|{desc}|[{remaining}]")) :
        OUT = Parallel(n_jobs=n_parallel)(delayed(one_copy)(i) for i in range(n_copy))
        
    return OUT




# *****************************************************************************
##
###
####
###
##
#
#### Scores based on multilple KnockOff copies ================================

def scoreMulti(combinedData, y, FDR=0.1, impStat=Feature_Importance._basicImp_ContinuousResponse, n_parallel=cpu_count()):
    """
    When we have multiple KnockOff copies corresponding to same DataMatrix, based on each of them we can compute one possible feature importance. Goal is to make one overall decision based on them.

    Parameters
    ----------
    combinedData : list of tuples in the form
        [(X,X_knockoff.1),(X,X_knockoff.2),...,(X,X_knockoff.n_copy)]

    y : Series or 1D-array ; for Series index=index_of_data , for array length=number_of_index_in_data
        The response variable. Can be continuous or categorical anything , but impStat should be chosen accordingly.e.g. - for continuous case use impStat=_basicImp_ContinuousResponse , for binary case use impStat=_basicImp_BinaryResponse , for multiple catogory case use impStat=LOFO_ImpCategorical etc.

    FDR : float between [0,1] or list of such float values ; default 0.1
        The False Discovery Rate upperbound to be specified.

    impStat : any function that computes feature importance & threshold for selection ; default _basicImp_ContinuousResponse
        This function should -
            * take input X, X_knockoff , y , FDR
            * produce output as importance statistics corresponding to features , followed by threshold values

    n_parallel : int ; default cpu_count()
        Number of cores used for parallel computing.


    Returns
    -------
    DataFrame of size (number_of_knockoff_copy, number_of_features+number_of_FDR_input)
    where -
        * each row corresponds to one copy,
        * initial columns are feature importance scores corresponding to various features
        * last columns are cut-off values corresponding to FDR control

    """

    lenKnockOff = len(combinedData)

    def score_for_one_copy(i) :
        X_,X_knockoff = combinedData[i]
        return impStat(X_,X_knockoff,y,FDR)
        
    with tqdm_joblib(tqdm(desc="Progress_Bar(Feature Importance Ordering...) & expected remaining time", total=lenKnockOff,bar_format="{n}/{total}{unit}|{bar}|{desc}|[{remaining}]")) :
        OUT = pd.DataFrame(Parallel(n_jobs=n_parallel)(delayed(score_for_one_copy)(i) for i in range(lenKnockOff)),index=range(lenKnockOff))

    return OUT




# *****************************************************************************

