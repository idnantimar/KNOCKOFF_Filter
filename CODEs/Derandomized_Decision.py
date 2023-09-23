"""
Created on Mon Sep  4 13:26:14 2023

Topic: Based on feature importance & threshold , determines which features to be selected/rejected
@author: R.Nandi
"""

from Basics import *
from joblib import Parallel, delayed



#### Scores based on multilple KnockOff copies ================================

def scoreMulti(combinedData,y,FDR,impStat,n_parallel):
    """
    When we have multiple KnockOff copies corresponding to same DataMatrix, based on each of them we can compute one possible feature importance. Goal is to make one overall decision based on them.

    Parameters
    ----------
    combinedData : list of tuples in the form
        [(X,X_knockoff.1),(X,X_knockoff.2),...,(X,X_knockoff.n_copy)]

    y : Series or 1D-array ; for Series index=index_of_data , for array length=number_of_index_in_data
        The response variable. Can be continuous or categorical anything , but impStat should be chosen accordingly.e.g. - for continuous case use impStat=basicImp_ContinuousResponse , for binary case use impStat=basicImp_BinaryResponse , for multiple catogory case use impStat=LOFO_ImpCategorical etc.

    FDR : float between [0,1] or list of such float values
        The False Discovery Rate upperbound to be specified.

    impStat : any function that computes feature importance & threshold for selection
        This function should -
            * take input X, X_knockoff , y , FDR
            * produce output as importance statistics corresponding to features , followed by threshold values

    n_parallel : int
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

    return pd.DataFrame(Parallel(n_jobs=n_parallel)(delayed(score_for_one_copy)(i) for i in range(lenKnockOff)),index=range(lenKnockOff))



# *****************************************************************************
##
###
####
###
##
#
#### Filtering ================================================================

def applyFilter(DATA,FDR,acceptance_rate=0.6,trueBeta_for_FDP=None,plotting=True,plot_Threshold=True,plot_Legend=True,appendTitle=''):
    """
    When we have multiple KnockOff copies corresponding to same DataMatrix, based on each of them we can compute one possible feature importance. Goal is to make one overall decision based on them.

    If we have a DataFrame of size (number_of_knockoff_copy, number_of_features+number_of_FDR_input) , where -
        * each row corresponds to one copy,
        * initial columns are feature importance scores corresponding to various features,
        * last columns are cut-off values corresponding to FDR control
    this function creates derandomized decision based on available iterations regarding "which feature should be accepted/rejected for FDR control" , i.e. only those features are finally selected which are selected >=n_aggregate*acceptance_rate times individually.

    """
    FDRs = np.array(FDR).ravel()
    lenFDR = len(FDRs)
    CutOffs = DATA.iloc[:,-lenFDR:]
    Scores = DATA.iloc[:,:-lenFDR]
    names = Scores.columns

    OUTPUT = {}
    for t in range(lenFDR):
        W = pd.concat([Scores,CutOffs.iloc[:,t]],axis=1)
        FDR = FDRs[t]

       ## selecting features -------------------------------
        SelectFeature = W.apply(lambda x: x[:-1]>=x[-1], axis=1).apply(np.mean,axis=0)
            # in first step, corresponding to each iteration- gives True for selected feature , False for rejected
            # in next step , computes the proportion of times each feature was selected out of total iterations
        SelectFeature = (SelectFeature>=acceptance_rate)
        Selected = names[SelectFeature]

       ## FDP ----------------------------------------------
        FDP = "True coefficients are Not known"
        if trueBeta_for_FDP is not None:
            trueBeta_for_FDP = np.array(trueBeta_for_FDP,dtype=bool).astype(int)
            countFD = lambda x: np.sum((x-trueBeta_for_FDP)==1)
                # counts where selected = True but true coefficient = 0
            FDP = countFD(SelectFeature)
            FDP /= max(1,sum(SelectFeature))

       ## Plotting -----------------------------------------
        if plotting:
           plt.figure(figsize=(10,0.25*len(names)))
           plt.boxplot(W.iloc[:,:-1],vert=False,labels=names,
                       patch_artist=True,boxprops=dict(facecolor='grey'),medianprops = dict(color="blue",linewidth=2))
           if plot_Threshold : plt.axvline(x=np.median(W.iloc[:,-1]),linestyle='--',color='r',linewidth=1.5)
           plt.grid()
           plt.xlabel("Feature Importance")
           plt.title("FDR= "+str(FDR*100)+"% | No. of selected features = "+str(len(Selected))+'/'+str(len(names))+appendTitle)
           if plot_Legend :
               for feature in names[::-1] :
                   plt.axvline(x=0,ymin=0,ymax=0,label=feature,color='lightgreen' if (feature in Selected) else 'red')
               plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper left')
           plt.show()

       ## Storing ------------------------------------------
        OUTPUT['FDR_'+str(FDR*100)] = {'SelectedFeature_Names' : Selected,
                'SelectedFeature_Indicators' : SelectFeature,
                'FDP' : FDP}

   ## return value -----------------------------------------
    return OUTPUT



# *****************************************************************************
