"""
Created on Mon Sep  4 13:26:14 2023

Topic: Based on feature importance & threshold , determines which features to be selected/rejected
@author: R.Nandi
"""

from Basics import *


#### Filtering ================================================================

def applyFilter(DATA,FDR,acceptance_rate=0.6,trueBeta_for_FDP=None,plotting=True,plot_Threshold=True,plot_Legend=True,appendTitle=''):
    """
    When we have a DataFrame , where -
        * each row is one iteration ,
        * initial columns are feature importance scores corresponding to various features,
        * last columns are cut-off values corresponding to FDR ,
    this function creates derandomized decision based on available iterations
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
        selectFeature = W.apply(lambda x: x[:-1]>=x[-1],axis=1 )
            # corresponding to each iteration, gives 1 for selected feature , 0 for not selected
        SelectFeature = selectFeature.apply(np.mean,axis=0)
        SelectFeature = (SelectFeature>=acceptance_rate)
        Selected = names[SelectFeature]

       ## FDP ----------------------------------------------
        FDP = "True coefficients are Not known"
        if trueBeta_for_FDP is not None:
            trueBeta_for_FDP = np.array(np.array(trueBeta_for_FDP,dtype=bool),dtype=int)
            countFD = lambda x: np.sum((x-trueBeta_for_FDP)==1)
                # counts where selected = True but true coefficient = 0
            FDP = countFD(np.array(SelectFeature,dtype=int))
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
