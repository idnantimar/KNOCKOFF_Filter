"""
Created on Mon Sep  4 00:09:01 2023

Topic: The Main Code File

NOTE: Appending the "path/to/this/folder" to the sys.path() & Importing this 'KnockOff_101' is enough to access all codes in this project
        It contains modules (check their documentations for use)-
                1)  Basics
                2)  Compute_Multiple
                        *  KnockOff_Generating
                        *  Feature_Importance
                3)  Derandomized_Decision
                4)  Diagnostics
                5)  Simulation_and_Visualization

@author: R.Nandi
"""


#### importing codes from modules created earlier .................
from .Basics import *
from . import Compute_Multiple
from . import Derandomized_Decision



#### KnockOff Filter ==========================================================

from multiprocessing import cpu_count

def KnockOff_Filter(X, y, FDR=0.1, method=Compute_Multiple.KnockOff_Generating.sKnockOff, Xs_Xknockoffs=False, impStat=Compute_Multiple.Feature_Importance.basicImp_ContinuousResponse, n_aggregate=20, acceptance_rate=0.6, plotting=True, plot_Threshold=True, plot_Legend=True, trueBeta_for_FDP=None, appendTitle='', n_parallel=cpu_count()):
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

    impStat : any function that computes feature importance & threshold for selection ; default basicImp_ContinuousResponse
        This function should -
            * take input X, X_knockoff , y , FDR
            * produce output as importance statistics corresponding to features , followed by threshold values

    n_aggregate : int ; default 20 ; not needed if Xs_Xknockoffs=True
        Number of KnockOff copies to be generated from same data for derandomized decision.

    acceptance_rate : float between [0,1] ; default 0.60
        In derandomization , a feature will be accepted if it is accepted in >=n_aggregate*acceptance_rate times individually.

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
                'FDR_level':{

                    'SelectedFeature_Names' : Names of the selected variables.

                    'SelectedFeature_Indicators' : Indicator of Selection (True= selected,False= rejected).

                    'FDP' : empirical FDR (if true coefficients are known).}


                'ImportanceScores_&_Thresholds' : Feature Importance scores and the Threshold values for selection/rejection
            }

    """

   ## generating Feature Importance Stats ------------------
    if not Xs_Xknockoffs :
        Xs_Xknockoffs = Compute_Multiple.genMulti(X,n_aggregate,method,True,n_parallel)
        DATA = Compute_Multiple.scoreMulti(Xs_Xknockoffs,y,FDR,impStat,n_parallel)
    else : DATA = Compute_Multiple.scoreMulti(X,y,FDR,impStat,n_parallel)

   ## Filtering ---------------------------------------------
    returnValue = Derandomized_Decision.applyFilter(DATA, FDR,acceptance_rate,trueBeta_for_FDP,plotting,plot_Threshold,plot_Legend,appendTitle)
    returnValue['ImportanceScores_&_Thresholds'] = DATA
    return returnValue




# *****************************************************************************
