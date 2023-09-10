"""
Created on Fri Sep  8 01:06:10 2023

Topic: Checking whether randomization of columns actually improves sKnockOff or not.
        All the columns in the DataMatrix are independent here.
@author: R.Nandi
"""

# working dir & useful Modules ..........................
from os import chdir
chdir("G:\PROJECTWORK_M.Stat_FINAL\SIMULATIONs\SIMULATION-1")

from sys import path
path.append("G:\PROJECTWORK_M.Stat_FINAL\CODEs")
from KNOCKOFF_MainFile import *



#### KNN based diagnostic of the quality of Knockoff ==========================

from sklearn.neighbors import KNeighborsClassifier


def KNN_checkQuality(XandX_knockoff):
    """
    Check the proportion that in the combined data of X & X_knockoff , the nearest neighbour of an observation is from the same group (i.e. if original obs is from X , nearest neighbour is also from X).
    * Ideally this proportion should be 1/2 .
    * A proportion very much >1/2 indicates the generated knockoff copy does not mimic the original data well.
    * A proportion very much <1/2 indicates the generated knockoff copy may overfit the original data. In extreme case when X=X_knockoff , this proportion = 0 .
    """
    X,X_knockoff = XandX_knockoff
    n,p = X.shape
    combinedData = pd.DataFrame(np.vstack((X.to_numpy(),X_knockoff.to_numpy())))
    target = np.array([0]*n + [1]*n)    # label of the target class

    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(combinedData, target)
    ix = knn_classifier.kneighbors()[1]    # index on nearest neighbour
    ix = np.ravel(ix//n)     # label of the target class of nearest neighbour

    return np.mean(ix==target)





# *****************************************************************************
##
###
####
###
##
#
#### SIMULATION ===============================================================
from time import time

Scores = lambda data, method : list(map(method,data))
n_itr = 10
n_knockoff = 12
DATA = {"without_randomization":{'p_'+str(p_): {} for p_ in [20,50,80,150]},
        "with_randomization":{'p_'+str(p_): {} for p_ in [20,50,80,150]}}
SCORESMed_KNN = {'p_'+str(p_): pd.DataFrame(index=["without_randomization","with_randomization"]) for p_ in [20,50,80,150]}
TIME_KNN = {'p_'+str(p_): pd.DataFrame(columns=["without_randomization","with_randomization"],index=range(n_itr)) for p_ in [20,50,80,150]}


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for itr in range(n_itr):
    X_Full = simulateIndependent(200,(150,0))
    for p_ in [20,50,80,150]:
        ## Data | p_  ...........................................
        t = time()
        Data_withoutModification = genMulti(X_Full.iloc[:,:p_],n_knockoff,method=sKnockOff)
        TIME_KNN['p_'+str(p_)].loc[itr,'without_randomization'] = time() - t
        t = time()
        Data_withModification = genMulti(X_Full.iloc[:,:p_],n_knockoff,method=sKnockOff_Modified)
        TIME_KNN['p_'+str(p_)].loc[itr,'with_randomization'] = time() - t
        DATA['without_randomization']['p_'+str(p_)]['itr'+str(itr)] = Data_withoutModification
        DATA['with_randomization']['p_'+str(p_)]['itr'+str(itr)] = Data_withModification

        scores_KNN = pd.DataFrame({"without_randomization":Scores(Data_withoutModification,KNN_checkQuality),
                      "with_randomization":Scores(Data_withModification,KNN_checkQuality)})
        SCORESMed_KNN['p_'+str(p_)] = SCORESMed_KNN['p_'+str(p_)].join(pd.Series(scores_KNN.median(),name=('itr'+str(itr))))
        sns.boxplot(data=scores_KNN,palette=['red','green'])
        plt.axhline(y=0.5,color='black',linestyle='dashed')
        plt.ylabel('KNN scores')
        plt.title("Itr_"+str(itr)+" | n_cols="+str(p_))
        plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from scipy.stats import wilcoxon
P_Vals = pd.Series(name='Signed-Rank Test',dtype=float,index=['p_20','p_50','p_80','p_150'])

p_ = ['20','50','80','150']
for col in p_ :
    D = SCORESMed_KNN['p_'+col].T
    Red = D.without_randomization
    Green = D.with_randomization
    stat, p_value = wilcoxon(Red, Green,zero_method='pratt')
    P_Vals['p_'+col] = p_value
    D = pd.melt(D,ignore_index=False)
    D.rename(columns={'variable':'sKnockOff'},inplace=True)
    sns.lineplot(x=D.index,y='value',hue='sKnockOff',data=D,palette=['red','green'],marker='o')
    plt.ylabel('avg. KNN_scores')
    plt.title('n_cols='+col+' | signed-rank test: p val '+str(np.round(p_value,5)))
    plt.axhline(y=0.5,color='black',linestyle='dashed')
    plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def pairwise_absCorrelation(XandX_knockoff):
    """
    Computes correlation coefficient between Xj & Xj_knockoff
    """
    X,X_knockoff = XandX_knockoff
    n,p = X.shape
    names = X.columns
    pairCor = np.corrcoef(X,X_knockoff,rowvar=False)
    pairCor = np.diag(pairCor[p:,:p])

    return pd.Series(np.abs(pairCor))



absCORR = {'p_'+str(p_):{"without_randomization": pd.DataFrame(index=range(p_)) ,"with_randomization": pd.DataFrame(index=range(p_)) } for p_ in [20,50,80,150]}

for p_ in [20,50,80,150]:
    for itr in range(n_itr):
        Data = DATA['with_randomization']['p_'+str(p_)]['itr'+str(itr)]
        absCORR['p_'+str(p_)]['with_randomization']['itr'+str(itr)] = pd.DataFrame(list(map(pairwise_absCorrelation,Data))).mean(axis=0)

        Data = DATA['without_randomization']['p_'+str(p_)]['itr'+str(itr)]
        absCORR['p_'+str(p_)]['without_randomization']['itr'+str(itr)] = pd.DataFrame(list(map(pairwise_absCorrelation,Data))).mean(axis=0)


for p_ in [20,50,80,150]:
    D = pd.DataFrame({'with_randomization': absCORR['p_'+str(p_)]['with_randomization'].mean(axis=1),
    'without_randomization': absCORR['p_'+str(p_)]['without_randomization'].mean(axis=1)})

    D = pd.melt(D,ignore_index=False)
    D.rename(columns={'variable':'sKnockOff'},inplace=True)
    sns.lineplot(x=D.index,y='value',hue='sKnockOff',data=D,palette=['green','red'])
    plt.ylabel('avg. pairwise corr')
    plt.xlabel('column index ')
    plt.title('n_cols='+str(p_))
    plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plot_Time = pd.DataFrame(index=["without_randomization","with_randomization"])
for p_ in [20,50,80,150]:
    plot_Time[p_] = TIME_KNN['p_'+str(p_)].median(axis=0)

D = pd.melt((plot_Time/n_knockoff).T,ignore_index=False)
D.rename(columns={'variable':'sKnockOff'},inplace=True)
sns.lineplot(x=D.index,y='value',hue='sKnockOff',data=D,palette=['red','green'],marker='o')
plt.ylabel('seconds/knockoff_copy')
plt.xlabel('number of columns ')
plt.title("Computation Time")
plt.grid()
plt.show()




# *****************************************************************************
