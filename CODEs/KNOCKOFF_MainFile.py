"""
Created on Mon Sep  4 00:09:01 2023

Topic: The Main Code File
NOTE: Appending the "path/to/this/folder" to the sys.path() & Importing this 'KNOCKOFF_MainFile' is enough to access all codes in this folder

@author: R.Nandi
"""


#### importing codes from modules created earlier .................
from Basics import *
from KnockOff_Generating import *      # First step of KnockOff
from Feature_Importance import *     # Intermediate step of KnockOff
from Derandomized_Decision import *    # Final step of KnockOff
from Diagnostics import *


#### for parallel computation .....................................
from joblib import Parallel, delayed
from multiprocessing import cpu_count



#### Generating Multiple KnockOff copies at a time ============================

def genMulti(X, n_copy, method=sKnockOff, scaling=True, n_parallel=cpu_count()):
    """
    Generates multiple KnockOff copies of a same DataMatrix.

    Parameters
    ----------
    X : DataFrame or 2d-array
        The DataMatrix.

    n_copy : int
        Number of copies to be generated.

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
    is_Cat = Cat_or_Num(X)

    if scaling : Scale_Numeric(X, is_Cat)

    def one_copy(i) : return method(X,is_Cat)
    return list(Parallel(n_jobs=n_parallel)(delayed(one_copy)(i) for i in range(n_copy)))




# *****************************************************************************
##
###
####
###
##
#
#### KnockOff Filter ==========================================================

def KnockOff_Filter(X, y, FDR = 0.1, method = sKnockOff, Xs_Xknockoffs = False, impStat = basicImp_ContinuousResponse, n_aggregate = 20, acceptance_rate = 0.6, plotting = True, plot_Threshold = True, plot_Legend = True, trueBeta_for_FDP = None, appendTitle = '', n_parallel=cpu_count()):
    """
    A function to select important features on a dataset , based on FDR control

    Parameters
    ----------
    X : The Main data part

        * if KnockOff is already generated - list of tuples [(Data,Data_knockoff.1),(Data,Data_knockoff.2),...]
        * if KnockOff is not generated yet - DataFrame or 2D-array where each row is an observation , each column is a feature
        ( NOTE - For categorical features with multiple(>2) levels , pass it in original form, not in '0-1 encoded' form. Otherwise there will be more than one dummy column related to a single feature , but the model treat them as independent and can return knockoff copy with more than one 1 in a dummy row )

    y : Series or 1D-array ; for Series index=index_of_data , for array length=number_of_index_in_data
        The response variable. Can be continuous or categorical anything , but impStat should be chosen accordingly.e.g. - for continuous case use impStat=basicImp_ContinuousResponse , for binary case use impStat=basicImp_BinaryResponse , for multiple catogory case use impStat=LOFO_ImpCategorical etc.

    FDR : float between [0,1] or list of such float values ; default 0.1
        The False Discovery Rate upperbound to be specified.

    method : any function that creates X_knockoff ; default sKnockOff ; not needed if Xs_Xknockoffs=True
        This function should take input-
            * X : DataMatrix
            * is_Cat : an array indicating which column is Categorical(True) , which one is Numerical(False)
        & produce output (X,X_knockoff) tuple.

    Xs_Xknockoffs : bool ; default False
        Whether in the data KnockOff copies are already inputted(True) or they are yet to be generated(False).

    impStat : any function that computes feature importance & threshold for selection ; default basicImp_ContinuousResponse.
        This function should -
            * take input X, X_knockoff , y , FDR
            * produce output as importance statistics corresponding to features , followed by threshold values

    n_aggregate : int ; default 20 ; not needed if Xs_Xknockoffs=True
        Number of KnockOff copies to be generated from same data for derandomized decision.

    acceptance_rate : float between [0,1] ; default 0.60
        In derandomization , a feature will be accepted if it is accepted in >n_aggregate*acceptance_rate times individually.

    plotting, plot_Threshold, plot_Legend : bool ; default True

    trueBeta_for_FDP : array of bool ; default None
        If we know which features are actually important(True) and which ones are null feature(False) , we can input it to compute empirical FDR.

    appendTitle : string ; default ''

    n_parallel : int ; default cpu_count()
        Number of cores used for parallel computing.


    Returns
    -------
    a dict of the form
        {
            'SelectedFeature_Names' : Names of the selected variables.

            'SelectedFeature_Indicators' : Indicator of Selection (True/False).

            'FDP' : empirical FDR (if true coefficients are known).
        }

    """

   ## generating Feature Importance Stats ------------------
    if not Xs_Xknockoffs :
        X = pd.DataFrame(X).copy()
        n,p = X.shape
        names = X.columns
        is_Cat = Cat_or_Num(X)
        Scale_Numeric(X, is_Cat)

        def this_is_the_main_job(i) :
            X_,X_knockoff = method(X,is_Cat)
            return impStat(X_,X_knockoff,y,FDR)
        DATA = pd.DataFrame(Parallel(n_jobs=n_parallel)(delayed(this_is_the_main_job)(i) for i in range(n_aggregate)),index=range(n_aggregate))

    else :
        lenKnockOff = len(X)

        def this_is_the_main_job(i) :
            X_,X_knockoff = X[i]
            return impStat(X_,X_knockoff,y,FDR)
        DATA = pd.DataFrame(Parallel(n_jobs=n_parallel)(delayed(this_is_the_main_job)(i) for i in range(lenKnockOff)),index=range(lenKnockOff))

   ## Filtering ---------------------------------------------
    returnValue = applyFilter(DATA, FDR,acceptance_rate,trueBeta_for_FDP,plotting,plot_Threshold,plot_Legend,appendTitle)
    return returnValue




# *****************************************************************************
