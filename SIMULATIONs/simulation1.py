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



#### MMD based diagnostic of the quality of Knockoff ==========================

from sklearn.metrics import pairwise_distances
from random import randint


def Dist_Z1Z2(Z1,Z2=None):
    #  estimates E|Z1-Z2| , where Z1 & Z2 are two independent random variables
    # Inputs are two DataMatrix corresponding to Z1 & Z2 observations respectively
    dist = pairwise_distances(Z1,Z2)
    np.fill_diagonal(dist ,0)
    return np.mean(dist)


MMD_score = lambda P1,P2 : Dist_Z1Z2(P1) + Dist_Z1Z2(P2) - 2*Dist_Z1Z2(P1,P2)




def MMD_checkQuality(XandX_knockoff , n_partialSwap = 10):
    """
    let LHS = [X,X_knockoff]
        RHS = anySwap(LHS)
        Dist_Z1Z2 = estimated_E|Z1-Z2| , where Z1 & Z2 are two independent random variables

    If the knockoff copy is of good quality, LHS & RHS should be identically distributed.
        MMD_score = Dist_Z1Z2(LHS,LHS) + Dist_Z1Z2(RHS,RHS) - 2*Dist_Z1Z2(LHS,RHS)  , should be near 0 in such case. The further this score from 0 , it says the knockoff fails to mimic the original data.
    """
    X,X_knockoff = XandX_knockoff
    n,p = X.shape
    LHS = pd.concat([X,X_knockoff],axis=1)
    fullSwap = lambda : pd.concat([X_knockoff,X],axis=1)
    def partialSwap():
        col_ix = np.array(range(2*p))
        swappable = choice(range(p),size=randint(1,p),replace=False)
        col_ix[swappable] += p
        col_ix[(swappable+p)] -= p
        return LHS.iloc[:,col_ix]

    score = [MMD_score(LHS,fullSwap())]
    for _ in range(n_partialSwap):
        score += [MMD_score(LHS,partialSwap())]

    return np.mean(score)




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
n_obs = 200
DATA = {"without_randomization":{'p_'+str(p_): {} for p_ in [20,50,80,150]},
        "with_randomization":{'p_'+str(p_): {} for p_ in [20,50,80,150]}}
SCORESMed = {'p_'+str(p_): pd.DataFrame(index=["without_randomization","with_randomization"]) for p_ in [20,50,80,150]}
TIME = {'p_'+str(p_): pd.DataFrame(columns=["without_randomization","with_randomization"],index=range(n_itr)) for p_ in [20,50,80,150]}


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for itr in range(n_itr):
    X_Full = simulateIndependent(n_obs,(150,0))
    for p_ in [20,50,80,150]:
        ## Data | p_  ...........................................
        t = time()
        Data_withoutModification = genMulti(X_Full.iloc[:,:p_],n_knockoff,method=sKnockOff)
        TIME['p_'+str(p_)].loc[itr,'without_randomization'] = time() - t
        t = time()
        Data_withModification = genMulti(X_Full.iloc[:,:p_],n_knockoff,method=sKnockOff_Modified)
        TIME['p_'+str(p_)].loc[itr,'with_randomization'] = time() - t
        DATA['without_randomization']['p_'+str(p_)]['itr'+str(itr)] = Data_withoutModification
        DATA['with_randomization']['p_'+str(p_)]['itr'+str(itr)] = Data_withModification


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for itr in range(n_itr):
    for p_ in [20,50,80,150]:
        ## Data | p_  ...........................................
        Data_withoutModification = DATA['without_randomization']['p_'+str(p_)]['itr'+str(itr)]
        Data_withModification = DATA['with_randomization']['p_'+str(p_)]['itr'+str(itr)]
        scores = pd.DataFrame({"without_randomization":Scores(Data_withoutModification,MMD_checkQuality),
                      "with_randomization":Scores(Data_withModification,MMD_checkQuality)})
        SCORESMed['p_'+str(p_)]['itr'+str(itr)] = scores.median()
        sns.boxplot(data=scores,palette=['red','green'])
        plt.axhline(color='black',linestyle='dashed')
        plt.ylabel('MMD scores')
        plt.title("Itr_"+str(itr)+" | n_cols="+str(p_))
        plt.show()


from scipy.stats import wilcoxon
P_Vals = pd.Series(name='Signed-Rank Test',dtype=float,index=['p_20','p_50','p_80','p_150'])

p_ = ['20','50','80','150']
for col in p_ :
    D = SCORESMed['p_'+col].T
    Red = D.without_randomization
    Green = D.with_randomization
    stat, p_value = wilcoxon(Red, Green,zero_method='pratt',alternative='greater')
        # perform one-sided Wilcoxon Signed-Rank test for "modififed method performs better"
    P_Vals['p_'+col] = p_value
    D = pd.melt(D,ignore_index=False)
    D.rename(columns={'variable':'sKnockOff'},inplace=True)
    sns.lineplot(x=D.index,y='value',hue='sKnockOff',data=D,palette=['red','green'],marker='o')
    plt.ylabel('avg. MMD_scores')
    plt.title('n_cols='+col+' | signed-rank test: p val '+str(np.round(p_value,5)))
    plt.axhline(color='black',linestyle='dashed')
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
    sns.lineplot(x=D.index,y='value',hue='sKnockOff',data=D,palette=['green','red']).set_xticks(range(0,p_,p_//10))
    plt.ylabel('absolute pairwise corr')
    plt.xlabel('column index ')
    plt.title('n_cols='+str(p_))
    plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plot_Time = pd.DataFrame(index=["without_randomization","with_randomization"])
for p_ in [20,50,80,150]:
    plot_Time[p_] = TIME['p_'+str(p_)].median(axis=0)

D = pd.melt((plot_Time/n_knockoff).T,ignore_index=False)
D.rename(columns={'variable':'sKnockOff'},inplace=True)
sns.lineplot(x=D.index,y='value',hue='sKnockOff',data=D,palette=['red','green'],marker='o')
plt.ylabel('seconds/knockoff_copy')
plt.xlabel('number of columns ')
plt.title("Computation Time")
plt.grid()
plt.show()




# *****************************************************************************
