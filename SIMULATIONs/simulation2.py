"""
Created on Fri Sep  8 01:06:10 2023

Topic: Checking whether randomization of columns actually improves sKnockOff or not.
        All the columns in the DataMatrix are not independent here, but covariances are nearly zero.
@author: R.Nandi
"""

# working dir & useful Modules ..........................
from os import chdir
chdir("G:\PROJECTWORK_M.Stat_FINAL\SIMULATIONs\SIMULATION-2")

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
DATA = {"without_randomization":{'p_'+str(p_): {} for p_ in [10,20,40,60,80]},
        "with_randomization":{'p_'+str(p_): {} for p_ in [10,20,40,60,80]}}
SCORESMed = {'p_'+str(p_): pd.DataFrame(index=["without_randomization","with_randomization"]) for p_ in [10,20,40,60,80]}
TIME = {'p_'+str(p_): pd.DataFrame(columns=["without_randomization","with_randomization"],index=range(n_itr)) for p_ in [10,20,40,60,80]}

mu = [0]*80
Sigma = np.corrcoef(np.random.rand(80,1000))
plt.figure(figsize=(6,6))
sns.heatmap(Sigma,cmap="YlGnBu", annot=False,square=True)
plt.title("correlation matrix : Full")
plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for itr in range(n_itr):
    X_Full = simulateJoint(n_obs,popln=lambda size: multivariate_normal(mu, Sigma,size))
    for p_ in [10,20,40,60,80]:
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
    for p_ in [10,20,40,60,80]:
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
P_Vals = pd.Series(name='Signed-Rank Test',dtype=float,index=['p_10','p_20','p_40','p_60','p_80'])

p_ = ['10','20','40','60','80']
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



absCORR = {'p_'+str(p_):{"without_randomization": pd.DataFrame(index=range(p_)) ,"with_randomization": pd.DataFrame(index=range(p_)) } for p_ in [10,20,40,60,80]}

for p_ in [10,20,40,60,80]:
    for itr in range(n_itr):
        Data = DATA['with_randomization']['p_'+str(p_)]['itr'+str(itr)]
        absCORR['p_'+str(p_)]['with_randomization']['itr'+str(itr)] = pd.DataFrame(list(map(pairwise_absCorrelation,Data))).mean(axis=0)

        Data = DATA['without_randomization']['p_'+str(p_)]['itr'+str(itr)]
        absCORR['p_'+str(p_)]['without_randomization']['itr'+str(itr)] = pd.DataFrame(list(map(pairwise_absCorrelation,Data))).mean(axis=0)


for p_ in [10,20,40,60,80]:
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
for p_ in [10,20,40,60,80]:
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

"""
Created on Mon Sep 11 19:30:52 2023

Topic: Checking whether randomization of columns actually improves sKnockOff or not.
        All the columns in the DataMatrix are not independent here. Each column is correlated with its two immediate neighbours.
@author: R.Nandi
"""


# working dir & useful Modules ..........................
from os import chdir
chdir("G:\PROJECTWORK_M.Stat_FINAL\SIMULATIONs\SIMULATION-2")

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
DATA = {"without_randomization":{'p_'+str(p_): {} for p_ in [10,20,40,60]},
        "without_randomization_Reverse":{'p_'+str(p_): {} for p_ in [10,20,40,60]},
        "with_randomization":{'p_'+str(p_): {} for p_ in [10,20,40,60]}}
SCORESMed = {'p_'+str(p_): pd.DataFrame(index=["without_randomization","without_randomization_Reverse","with_randomization"]) for p_ in [10,20,40,60]}
TIME = {'p_'+str(p_): pd.DataFrame(columns=["without_randomization","without_randomization_Reverse","with_randomization"],index=range(n_itr)) for p_ in [10,20,40,60]}

mu = [0]*60
Sigma = np.eye(60) + np.diag([0.5]*59,k=1) + np.diag([0.5]*59,k=-1) + np.diag([0.25]*58,k=2) + np.diag([0.25]*58,k=-2)
plt.figure(figsize=(6,6))
sns.heatmap(Sigma,cmap="YlGnBu", annot=False,square=True)
plt.title("correlation matrix : Full")
plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for itr in range(n_itr):
    X_Full = simulateJoint(n_obs,popln=lambda size: multivariate_normal(mu, Sigma,size))
    for p_ in [10,20,40,60]:
        ## Data | p_  ...........................................
        t = time()
        Data_withoutModification = genMulti(X_Full.iloc[:,:p_],n_knockoff,method=sKnockOff)
        TIME['p_'+str(p_)].loc[itr,'without_randomization'] = time() - t
        t = time()
        Data_withoutModification_Reverse = genMulti(X_Full.iloc[:,np.arange(p_)[::-1]],n_knockoff,method=sKnockOff)
        TIME['p_'+str(p_)].loc[itr,'without_randomization_Reverse'] = time() - t
        t = time()
        Data_withModification = genMulti(X_Full.iloc[:,:p_],n_knockoff,method=sKnockOff_Modified)
        TIME['p_'+str(p_)].loc[itr,'with_randomization'] = time() - t
        DATA['without_randomization']['p_'+str(p_)]['itr'+str(itr)] = Data_withoutModification
        DATA['without_randomization_Reverse']['p_'+str(p_)]['itr'+str(itr)] = Data_withoutModification_Reverse
        DATA['with_randomization']['p_'+str(p_)]['itr'+str(itr)] = Data_withModification


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for itr in range(n_itr):
    for p_ in [10,20,40,60]:
        ## Data | p_  ...........................................
        Data_withoutModification = DATA['without_randomization']['p_'+str(p_)]['itr'+str(itr)]
        Data_withoutModification_Reverse = DATA['without_randomization_Reverse']['p_'+str(p_)]['itr'+str(itr)]
        Data_withModification = DATA['with_randomization']['p_'+str(p_)]['itr'+str(itr)]
        scores = pd.DataFrame({"without_randomization":Scores(Data_withoutModification,MMD_checkQuality),
                "without_randomization_Reverse":Scores(Data_withoutModification_Reverse,MMD_checkQuality),
                      "with_randomization":Scores(Data_withModification,MMD_checkQuality)})
        SCORESMed['p_'+str(p_)]['itr'+str(itr)] = scores.median()
        plt.figure(figsize=(8,6))
        sns.boxplot(data=scores,palette=['red','darkgrey','green'])
        plt.axhline(color='black',linestyle='dashed')
        plt.ylabel('MMD scores')
        plt.title("Itr_"+str(itr)+" | n_cols="+str(p_))
        plt.show()


from scipy.stats import wilcoxon
P_Vals = pd.DataFrame(columns=["Forward_vs_Reverse","Forward_vs_Randomization","Reverse_vs_Randomization"],index=['p_10','p_20','p_40','p_60'])
'''
 **  Using Wilcoxon Signed-Rank test to check whether on average MMD_scores reduce by introducing randomization

'''
for col in ['10','20','40','60'] :
    D = SCORESMed['p_'+col].T
    Red = D.without_randomization
    Grey = D.without_randomization_Reverse
    Green = D.with_randomization

    stat, p_value = wilcoxon(Red, Green,zero_method='pratt',alternative='greater')
    P_Vals.loc['p_'+col,"Forward_vs_Randomization"] = p_value
    stat, p_value = wilcoxon(Grey, Green,zero_method='pratt',alternative='greater')
    P_Vals.loc['p_'+col,"Reverse_vs_Randomization"] = p_value
    stat, p_value = wilcoxon(Red, Grey,zero_method='pratt',alternative='two-sided')
    P_Vals.loc['p_'+col,"Forward_vs_Reverse"] = p_value


    D = pd.melt(D,ignore_index=False)
    D.rename(columns={'variable':'sKnockOff'},inplace=True)
    sns.lineplot(x=D.index,y='value',hue='sKnockOff',data=D,palette=['red','darkgrey','green'],marker='o')
    plt.ylabel('avg. MMD_scores')
    plt.title('n_cols='+col)
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



absCORR = {'p_'+str(p_):{"without_randomization": pd.DataFrame(index=range(p_)),
                         "without_randomization_Reverse": pd.DataFrame(index=range(p_)),
                         "with_randomization": pd.DataFrame(index=range(p_)) } for p_ in [10,20,40,60,80]}

for p_ in [10,20,40,60]:
    for itr in range(n_itr):
        Data = DATA['with_randomization']['p_'+str(p_)]['itr'+str(itr)]
        absCORR['p_'+str(p_)]['with_randomization']['itr'+str(itr)] = pd.DataFrame(list(map(pairwise_absCorrelation,Data))).mean(axis=0)

        Data = DATA['without_randomization']['p_'+str(p_)]['itr'+str(itr)]
        absCORR['p_'+str(p_)]['without_randomization']['itr'+str(itr)] = pd.DataFrame(list(map(pairwise_absCorrelation,Data))).mean(axis=0)

        Data = DATA['without_randomization_Reverse']['p_'+str(p_)]['itr'+str(itr)]
        absCORR['p_'+str(p_)]['without_randomization_Reverse']['itr'+str(itr)] = (pd.DataFrame(list(map(pairwise_absCorrelation,Data))).mean(axis=0).to_numpy())[::-1]




from scipy.stats import spearmanr
checkTREND_ = pd.DataFrame(columns=['without_randomization','without_randomization_Reverse','with_randomization'],index=['p_10','p_20','p_40','p_60'])
checkTREND_pvalues = pd.DataFrame(columns=['without_randomization','without_randomization_Reverse','with_randomization'],index=['p_10','p_20','p_40','p_60'])
'''
 ** Checking whether the order(Forward or Reversed or Randomized) of columns in KnockOff matters
'''
for p_ in [10,20,40,60]:
    D = pd.DataFrame({'without_randomization': absCORR['p_'+str(p_)]['without_randomization'].mean(axis=1),
    'without_randomization_Reverse': absCORR['p_'+str(p_)]['without_randomization_Reverse'].mean(axis=1),
    'with_randomization': absCORR['p_'+str(p_)]['with_randomization'].mean(axis=1)})

    Red = D['without_randomization']
    stat,p_value = spearmanr(Red,Red.index,alternative='two-sided')
    checkTREND_.loc['p_'+str(p_),'without_randomization'] = stat
    checkTREND_pvalues.loc['p_'+str(p_),'without_randomization'] = p_value

    Grey = D['without_randomization_Reverse']
    stat,p_value = spearmanr(Grey,Grey.index,alternative='two-sided')
    checkTREND_.loc['p_'+str(p_),'without_randomization_Reverse'] = stat
    checkTREND_pvalues.loc['p_'+str(p_),'without_randomization_Reverse'] = p_value

    Green = D['with_randomization']
    stat,p_value = spearmanr(Green,Green.index,alternative='two-sided')
    checkTREND_.loc['p_'+str(p_),'with_randomization'] = stat
    checkTREND_pvalues.loc['p_'+str(p_),'with_randomization'] = p_value


    D = pd.melt(D,ignore_index=False)
    D.rename(columns={'variable':'sKnockOff'},inplace=True)
    sns.lineplot(x=D.index,y='value',hue='sKnockOff',data=D,palette=['red','darkgrey','green']).set_xticks(range(0,p_,p_//10))
    plt.ylabel('absolute pairwise corr')
    plt.xlabel('column index ')
    plt.title('n_cols='+str(p_))
    plt.show()




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plot_Time = pd.DataFrame(index=["without_randomization","without_randomization_Reverse","with_randomization"])
for p_ in [10,20,40,60]:
    plot_Time[p_] = TIME['p_'+str(p_)].median(axis=0)

D = pd.melt((plot_Time/n_knockoff).T,ignore_index=False)
D.rename(columns={'variable':'sKnockOff'},inplace=True)
sns.lineplot(x=D.index,y='value',hue='sKnockOff',data=D,palette=['red','darkgrey','green'],marker='o')
plt.ylabel('seconds/knockoff_copy')
plt.xlabel('number of columns ')
plt.title("Computation Time")
plt.grid()
plt.show()




# *****************************************************************************


"""
Created on Mon Sep 11 19:30:52 2023

Topic: Checking whether randomization of columns actually improves sKnockOff or not.
        All the columns in the DataMatrix are not independent here. The correlation matrix is based on some real data.
@author: R.Nandi
"""


# working dir & useful Modules ..........................
from os import chdir
chdir("G:\PROJECTWORK_M.Stat_FINAL\SIMULATIONs\SIMULATION-4")

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
#### SIMULATION-1 =============================================================
X = pd.read_excel("Concrete_Data.xls")

n_knockoff = 12
Data_withoutModification = genMulti(X,n_knockoff,method=sKnockOff)
Data_withoutModification_Reverse = genMulti(X.iloc[:,::-1],n_knockoff,method=sKnockOff)
Data_withModification = genMulti(X,n_knockoff,method=sKnockOff_Modified)

DATA = {"without_randomization":Data_withoutModification,
        "without_randomization_Reverse":Data_withoutModification_Reverse,
        "with_randomization":Data_withModification}

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Scores = lambda data, method : list(map(method,data))

SCORES = pd.DataFrame({"without_randomization":Scores(Data_withoutModification,MMD_checkQuality),
                "without_randomization_Reverse":Scores(Data_withoutModification_Reverse,MMD_checkQuality),
                "with_randomization":Scores(Data_withModification,MMD_checkQuality)})
plt.figure(figsize=(8,6))
sns.boxplot(data=SCORES,palette=['red','darkgrey','green'])
plt.axhline(color='black',linestyle='dashed')
plt.ylabel('MMD scores')
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

    return pd.Series(np.abs(pairCor),index=names)



absCORR = {}
absCORR['without_randomization'] = pd.DataFrame(list(map(pairwise_absCorrelation,Data_withoutModification))).mean(axis=0)
ix = absCORR['without_randomization'].index
absCORR['without_randomization_Reverse'] = pd.DataFrame(list(map(pairwise_absCorrelation,Data_withoutModification_Reverse))).mean(axis=0)
absCORR['with_randomization'] = pd.DataFrame(list(map(pairwise_absCorrelation,Data_withModification))).mean(axis=0)
absCORR = pd.DataFrame(absCORR,index=ix)


D = pd.melt(absCORR,ignore_index=False)
D.rename(columns={'variable':'sKnockOff'},inplace=True)
plt.figure(figsize=(7,6))
sns.lineplot(x=D.index,y='value',hue='sKnockOff',data=D,palette=['red','darkgrey','green'])
plt.xticks(rotation=45)
plt.ylabel('absolute pairwise corr')
plt.xlabel('column index ')
plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Data,Data_knockoff = Data_withModification[np.random.randint(0,n_knockoff)]
Visual(Data,Data_knockoff,Cat_or_Num(Data),appendTitle=' | with_randomization',scale_the_corrplot=1,KDE=True)

Data,Data_knockoff = Data_withoutModification[np.random.randint(0,n_knockoff)]
Visual(Data,Data_knockoff,Cat_or_Num(Data),appendTitle=' | without_randomization',scale_the_corrplot=1,KDE=True)

Data,Data_knockoff = Data_withoutModification_Reverse[np.random.randint(0,n_knockoff)]
Data = Data.iloc[:,::-1]
Data_knockoff = Data_knockoff.iloc[:,::-1]
Visual(Data,Data_knockoff,Cat_or_Num(Data),appendTitle=' | without_randomization_Reverse',scale_the_corrplot=1,KDE=True)


# *****************************************************************************
##
###
####
###
##
#
#### SIMULATION-2 =============================================================
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X2 = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
X2 = pd.DataFrame(X2)

n_knockoff = 12
Data_withoutModification = genMulti(X2,n_knockoff,method=sKnockOff)
Data_withoutModification_Reverse = genMulti(X2.iloc[:,::-1],n_knockoff,method=sKnockOff)
Data_withModification = genMulti(X2,n_knockoff,method=sKnockOff_Modified)

DATA2 = {"without_randomization":Data_withoutModification,
        "without_randomization_Reverse":Data_withoutModification_Reverse,
        "with_randomization":Data_withModification}

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Scores = lambda data, method : list(map(method,data))

SCORES2 = pd.DataFrame({"without_randomization":Scores(Data_withoutModification,MMD_checkQuality),
                "without_randomization_Reverse":Scores(Data_withoutModification_Reverse,MMD_checkQuality),
                "with_randomization":Scores(Data_withModification,MMD_checkQuality)})
plt.figure(figsize=(8,6))
sns.boxplot(data=SCORES2,palette=['red','darkgrey','green'])
plt.axhline(color='black',linestyle='dashed')
plt.ylabel('MMD scores')
plt.show()



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

absCORR2 = {}
absCORR2['without_randomization'] = pd.DataFrame(list(map(pairwise_absCorrelation,Data_withoutModification))).mean(axis=0)
ix = absCORR2['without_randomization'].index
absCORR2['without_randomization_Reverse'] = pd.DataFrame(list(map(pairwise_absCorrelation,Data_withoutModification_Reverse))).mean(axis=0)
absCORR2['with_randomization'] = pd.DataFrame(list(map(pairwise_absCorrelation,Data_withModification))).mean(axis=0)
absCORR2 = pd.DataFrame(absCORR2,index=ix)


D = pd.melt(absCORR2,ignore_index=False)
D.rename(columns={'variable':'sKnockOff'},inplace=True)
plt.figure(figsize=(7,6))
sns.lineplot(x=D.index,y='value',hue='sKnockOff',data=D,palette=['red','darkgrey','green'])
plt.xticks(rotation=45)
plt.ylabel('absolute pairwise corr')
plt.xlabel('column index ')
plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Data,Data_knockoff = Data_withModification[np.random.randint(0,n_knockoff)]
Visual(Data,Data_knockoff,Cat_or_Num(Data),appendTitle=' | with_randomization',scale_the_corrplot=0.8,KDE=True)

Data,Data_knockoff = Data_withoutModification[np.random.randint(0,n_knockoff)]
Visual(Data,Data_knockoff,Cat_or_Num(Data),appendTitle=' | without_randomization',scale_the_corrplot=0.8,KDE=True)

Data,Data_knockoff = Data_withoutModification_Reverse[np.random.randint(0,n_knockoff)]
Data = Data.iloc[:,::-1]
Data_knockoff = Data_knockoff.iloc[:,::-1]
Visual(Data,Data_knockoff,Cat_or_Num(Data),appendTitle=' | without_randomization_Reverse',scale_the_corrplot=0.8,KDE=True)



# *****************************************************************************
##
###
####
###
##
#
#### SIMULATION-3 =============================================================
X3= pd.read_csv("my_data.csv",header = None)

n_knockoff = 12
Data_withoutModification = genMulti(X3,n_knockoff,method=sKnockOff)
Data_withoutModification_Reverse = genMulti(X3.iloc[:,::-1],n_knockoff,method=sKnockOff)
Data_withModification = genMulti(X3,n_knockoff,method=sKnockOff_Modified)

DATA3 = {"without_randomization":Data_withoutModification,
        "without_randomization_Reverse":Data_withoutModification_Reverse,
        "with_randomization":Data_withModification}

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Scores = lambda data, method : list(map(method,data))

SCORES3 = pd.DataFrame({"without_randomization":Scores(Data_withoutModification,MMD_checkQuality),
                "without_randomization_Reverse":Scores(Data_withoutModification_Reverse,MMD_checkQuality),
                "with_randomization":Scores(Data_withModification,MMD_checkQuality)})
plt.figure(figsize=(8,6))
sns.boxplot(data=SCORES3,palette=['red','darkgrey','green'])
plt.axhline(color='black',linestyle='dashed')
plt.ylabel('MMD scores')
plt.show()



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


absCORR3 = {}
absCORR3['without_randomization'] = pd.DataFrame(list(map(pairwise_absCorrelation,Data_withoutModification))).mean(axis=0)
ix = absCORR3['without_randomization'].index
absCORR3['without_randomization_Reverse'] = pd.DataFrame(list(map(pairwise_absCorrelation,Data_withoutModification_Reverse))).mean(axis=0)
absCORR3['with_randomization'] = pd.DataFrame(list(map(pairwise_absCorrelation,Data_withModification))).mean(axis=0)
absCORR3 = pd.DataFrame(absCORR3,index=ix)


D = pd.melt(absCORR3,ignore_index=False)
D.rename(columns={'variable':'sKnockOff'},inplace=True)
plt.figure(figsize=(7,6))
sns.lineplot(x=D.index,y='value',hue='sKnockOff',data=D,palette=['red','darkgrey','green'])
plt.xticks(rotation=60)
plt.ylabel('absolute pairwise corr')
plt.xlabel('column index ')
plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Data,Data_knockoff = Data_withModification[np.random.randint(0,n_knockoff)]
Visual(Data,Data_knockoff,Cat_or_Num(Data),appendTitle=' | with_randomization',scale_the_corrplot=0.5)

Data,Data_knockoff = Data_withoutModification[np.random.randint(0,n_knockoff)]
Visual(Data,Data_knockoff,Cat_or_Num(Data),appendTitle=' | without_randomization',scale_the_corrplot=0.5)

Data,Data_knockoff = Data_withoutModification_Reverse[np.random.randint(0,n_knockoff)]
Data = Data.iloc[:,::-1]
Data_knockoff = Data_knockoff.iloc[:,::-1]
Visual(Data,Data_knockoff,Cat_or_Num(Data),appendTitle=' | without_randomization_Reverse',scale_the_corrplot=0.5)


# *****************************************************************************

