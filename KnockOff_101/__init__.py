"""
Created on Mon Sep  4 00:09:01 2023

Topic: The Main Code File

NOTE: Appending the "path/to/this/folder" to the sys.path() & Importing this 'KnockOff_101' is enough to access all codes in this project
        It contains modules (check their documentations for use)-
                1)  Basics
                2)  Compute_Multiple
                        *  KnockOff_Generating
                        *  Feature_Importance
                3)  Comparisons_and_checkQuality
                        *
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

def KnockOff_Filter(X, y, is_Cat, FDR=0.1,
                    method= lambda Z,z_type,seed: Compute_Multiple.KnockOff_Generating.sKnockOff(Z,z_type,seed_for_sample=seed), seed_for_BaseMethod=None,
                    knockoff_copy_done_already=False,
                    impStat=Compute_Multiple.Feature_Importance._basicImp_ContinuousResponse,
                    n_aggregate=20, shuffle_columns=True, seed_for_shuffle=None, acceptance_rate=0.6, n_parallel=cpu_count(),
                    plotting=True, plot_Threshold=True, plot_Legend=True, trueBeta_for_FDP=None, appendTitle='', plot_Scale_width_and_height=(1,1)):
    """
    A function to select important features on a dataset , based on FDR control

    Parameters
    ----------
    X : The Main data part

        * if KnockOff is already generated - list of tuples [(Data,Data_knockoff.1),(Data,Data_knockoff.2),...]
        * if KnockOff is not generated yet - DataFrame or 2D-array where each row is an observation , each column is a feature
        ( NOTE - For categorical features with multiple(>2) levels , pass it in original form, not in '0-1 encoded' form. Otherwise there will be more than one dummy column related to a single feature , but the model treat them as independent and can return knockoff copy with more than one 1 in a dummy row )

    y : Series or 1D-array ; for Series index=index_of_data , for array length=number_of_index_in_data
        The response variable. Can be continuous or categorical anything , but impStat should be chosen accordingly.e.g. - for continuous case use impStat=_basicImp_ContinuousResponse , for binary case use impStat=_basicImp_BinaryResponse , for multiple catogory case use impStat=LOFO_ImpCategorical etc.

    is_Cat : list or array of True/False values ; not needed for Xs_Xknockoffs=True
        Each element determines whether the corresponding column of X is of Categorical(True) or Numerical(False) type.

    FDR : float between [0,1] or list of such float values ; default 0.1
        The False Discovery Rate upperbound to be specified.

    method : A function that creates X_knockoff ; default sKnockOff
        This function should take input -
            * Z : DataMatrix
            * z_type : an array indicating which column is Categorical(True) , which one is Numerical(False)
        & produce output (X,X_knockoff) tuple
            * seed : for reproducible output.

    seed_for_BaseMethod : int ; default None
        seed to be used in base method for each KnockOff copy.

    knockoff_copy_done_already : bool ; default False
        Whether in the data KnockOff copies are already inputted(True) or they are yet to be generated(False).

    impStat : any function that computes feature importance & threshold for selection ; default _basicImp_ContinuousResponse
        This function should -
            * take input X, X_knockoff , y , FDR
            * produce output as importance statistics corresponding to features , followed by threshold values

    n_aggregate : int ; default 20 ; not needed if Xs_Xknockoffs=True
        Number of KnockOff copies to be generated from same data for derandomized decision.

    shuffle_columns : bool ; default True
        Whether the columns of DataMatrix should be shuffled before each iteration
        (useful to mitigate some systematic bias due to sequential nature of KnockOff generating algorithm).

    seed_for_shuffle : int ; default None
        seed for controling the shuffle of the columns , when shuffle_columns=True.

    acceptance_rate : float between [0,1] ; default 0.60
        In derandomization , a feature will be accepted if it is accepted in >=n_aggregate*acceptance_rate times individually.

    n_parallel : int ; default cpu_count()
        Number of cores used for parallel computing.

    plotting, plot_Threshold, plot_Legend : bool ; default True

    trueBeta_for_FDP : array of bool ; default None
        If we know which features are actually important(True) and which ones are null feature(False) , we can input it to compute empirical FDR.

    appendTitle : string ; default ''

    plot_Scale_width_and_height : a tuple of the form (w,h) ; default (1,1)


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
    if not knockoff_copy_done_already :
        Xs_Xknockoffs = Compute_Multiple.genMulti(X,n_aggregate,is_Cat,method,seed_for_BaseMethod,True,n_parallel,shuffle_columns,seed_for_shuffle)
        DATA = Compute_Multiple.scoreMulti(Xs_Xknockoffs,y,FDR,impStat,n_parallel)
    else : DATA = Compute_Multiple.scoreMulti(X,y,FDR,impStat,n_parallel)

   ## Filtering ---------------------------------------------
    returnValue = Derandomized_Decision.applyFilter(DATA,FDR,acceptance_rate,trueBeta_for_FDP,plotting,plot_Threshold,plot_Legend,appendTitle,plot_Scale_width_and_height)
    returnValue['ImportanceScores_&_Thresholds'] = DATA
    return returnValue




# *****************************************************************************
