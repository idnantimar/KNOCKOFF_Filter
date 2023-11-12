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
path.append("G:\PROJECTWORK_M.Stat_FINAL")
from KnockOff_101.Basics import *
from KnockOff_101 import Simulation_and_Visualization
from KnockOff_101.Compute_Multiple.KnockOff_Generating import sKnockOff,sKnockOff_Modified
from KnockOff_101.Diagnostics import MMD_checkQuality,pairwise_absCorrelation

#### SIMULATION ===============================================================
n_itr = 25
n_obs = 200
n_Cols = [20]
m = max(n_Cols)
DATA = {"without_randomization":{'p_'+str(p_): {} for p_ in n_Cols},
        "with_randomization":{'p_'+str(p_): {} for p_ in n_Cols}}

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for itr in range(n_itr):
    X_Full = Simulation_and_Visualization.simulateIndependent(n_obs,(m,0))
    for p_ in n_Cols:
        ## Data | p_  ...........................................
        X_current = X_Full.iloc[:,:p_]
        Data_withoutModification = sKnockOff(X_current,Cat_or_Num(X_current),True)
        Data_withModification = sKnockOff_Modified(X_current,Cat_or_Num(X_current),True)
        DATA['without_randomization']['p_'+str(p_)]['itr'+str(itr)] = Data_withoutModification
        DATA['with_randomization']['p_'+str(p_)]['itr'+str(itr)] = Data_withModification


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

n_itrPair = int((n_itr)*(n_itr-1)/2)
SCORES = {'p_'+str(p_): pd.DataFrame(columns=["without_randomization","with_randomization"],index=range(n_itrPair)) for p_ in n_Cols}

itr = 0
for i in range(n_itr-1):
    for j in range(i+1,n_itr):
        for p_ in n_Cols:
            ## Data | p_  ...........................................
            D1wo,D1wo_knockoff = DATA['without_randomization']['p_'+str(p_)]['itr'+str(i)]
            D2wo,D2wo_knockoff = DATA['without_randomization']['p_'+str(p_)]['itr'+str(j)]
            D1w,D1w_knockoff = DATA['with_randomization']['p_'+str(p_)]['itr'+str(i)]
            D2w,D2w_knockoff = DATA['with_randomization']['p_'+str(p_)]['itr'+str(j)]
            SCORES['p_'+str(p_)].loc[itr,"without_randomization"] = MMD_checkQuality(D1wo,D1wo_knockoff,D2wo,D2wo_knockoff,set_seed=itr)
            SCORES['p_'+str(p_)].loc[itr,"with_randomization"] = MMD_checkQuality(D1w,D1w_knockoff,D2w,D2w_knockoff,set_seed=itr)
        itr += 1

from scipy.stats import wilcoxon
P_Vals = pd.Series(name='Signed-Rank Test',dtype=float,index=np.vectorize(lambda s:'p_'+str(s))(n_Cols))

for p_ in n_Cols :
    D = SCORES['p_'+str(p_)]
    Red = D.without_randomization
    Green = D.with_randomization
    stat, p_value = wilcoxon(Red,Green,zero_method='pratt',alternative='two-sided')
        # perform one-sided Wilcoxon Signed-Rank test for "modififed method performs better"
    P_Vals['p_'+str(p_)] = p_value
    sns.boxplot(data=D,palette=['red','green'])
    plt.axhline(color='black',linestyle='dashed')
    plt.ylabel('MMD_scores')
    plt.title('n_cols='+str(p_)+' | signed-rank test: p val '+str(np.round(p_value,5)))
    plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


absCORR = {'p_'+str(p_):{"without_randomization": pd.DataFrame(index=np.vectorize(lambda s:'Num'+str(s))(range(p_)),columns=np.vectorize(lambda s:'itr'+str(s))(range(n_itr))) ,
                         "with_randomization": pd.DataFrame(index=np.vectorize(lambda s:'Num'+str(s))(range(p_)),columns=np.vectorize(lambda s:'itr'+str(s))(range(n_itr))) } for p_ in n_Cols}

for p_ in n_Cols:
    for itr in range(n_itr):
        D1w,D1w_knockoff = DATA['with_randomization']['p_'+str(p_)]['itr'+str(itr)]
        absCORR['p_'+str(p_)]['with_randomization']['itr'+str(itr)] = pairwise_absCorrelation(D1w,D1w_knockoff)

        D1wo,D1wo_knockoff = DATA['without_randomization']['p_'+str(p_)]['itr'+str(itr)]
        absCORR['p_'+str(p_)]['without_randomization']['itr'+str(itr)] = pairwise_absCorrelation(D1wo,D1wo_knockoff)


for p_ in n_Cols:
    D = pd.DataFrame({'with_randomization': absCORR['p_'+str(p_)]['with_randomization'].mean(axis=1),
    'without_randomization': absCORR['p_'+str(p_)]['without_randomization'].mean(axis=1)})
    D = pd.melt(D,ignore_index=False)
    D.rename(columns={'variable':'sKnockOff'},inplace=True)
    sns.lineplot(x=D.index,y='value',hue='sKnockOff',data=D,palette=['green','red']).set_xticks(range(0,p_,p_//10))
    plt.ylabel('absolute pairwise corr')
    plt.xlabel('column index ')
    plt.xticks(rotation=45)
    plt.title('n_cols='+str(p_))
    plt.show()
    D = pd.DataFrame({'without_randomization': absCORR['p_'+str(p_)]['without_randomization'].mean(axis=1)})
    D = pd.melt(D,ignore_index=False)
    D.rename(columns={'variable':'sKnockOff'},inplace=True)
    sns.lineplot(x=D.index,y='value',hue='sKnockOff',data=D,palette=['red']).set_xticks(range(0,p_,p_//10))
    plt.ylabel('absolute pairwise corr')
    plt.xlabel('column index ')
    plt.xticks(rotation=45)
    plt.title('n_cols='+str(p_))
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
path.append("G:\PROJECTWORK_M.Stat_FINAL")
from KnockOff_101.Basics import *
from KnockOff_101 import Simulation_and_Visualization
from KnockOff_101.Compute_Multiple.KnockOff_Generating import sKnockOff,sKnockOff_Modified
from KnockOff_101.Diagnostics import MMD_checkQuality,pairwise_absCorrelation


#### SIMULATION ===============================================================
n_itr = 25
n_obs = 200
n_Cols = [50,40,30,20,10,5]
m = max(n_Cols)
Data_Format = lambda : {"without_randomization":{'p_'+str(p_): {} for p_ in n_Cols},
        "without_randomization_Reverse":{'p_'+str(p_): {} for p_ in n_Cols},
        "with_randomization":{'p_'+str(p_): {} for p_ in n_Cols}}
DATA = []

mu = [0]*m
Sigma = np.eye(m) + np.diag([0.5]*(m-1),k=1) + np.diag([0.5]*(m-1),k=-1) + np.diag([0.25]*(m-2),k=2) + np.diag([0.25]*(m-2),k=-2)
plt.figure(figsize=(6,6))
sns.heatmap(Sigma,cmap="YlGnBu", annot=False,square=True)
plt.title("correlation matrix : Full")
plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def one_copy(i):
    X_Full = Simulation_and_Visualization.simulateJoint(n_obs,popln=lambda size: np.random.multivariate_normal(mu, Sigma,size))
    Data = Data_Format()
    for p_ in n_Cols:
        X_current = X_Full.iloc[:,:p_]
        Data['without_randomization']['p_'+str(p_)] = sKnockOff(X_current,Cat_or_Num(X_current),True)
        Data['without_randomization_Reverse']['p_'+str(p_)] = sKnockOff(X_current.iloc[:,::-1],Cat_or_Num(X_current.iloc[:,::-1]),True)
        Data['with_randomization']['p_'+str(p_)] = sKnockOff_Modified(X_current,Cat_or_Num(X_current),True)
    return Data

from joblib import Parallel, delayed
from multiprocessing import cpu_count
DATA += Parallel(n_jobs=cpu_count())(delayed(one_copy)(i) for i in range(n_itr))



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

n_itrPair = int((n_itr)*(n_itr-1)/2)
SCORES = {'p_'+str(p_): pd.DataFrame(columns=["without_randomization","without_randomization_Reverse","with_randomization"],index=range(n_itrPair)) for p_ in n_Cols}

itr = 0
for i in range(n_itr-1):
    for j in range(i+1,n_itr):
        for p_ in n_Cols:
            ## Data | p_  ...........................................
            D1wo,D1wo_knockoff = DATA[i]['without_randomization']['p_'+str(p_)]
            D2wo,D2wo_knockoff = DATA[j]['without_randomization']['p_'+str(p_)]
            D1wor,D1wor_knockoff = DATA[i]['without_randomization_Reverse']['p_'+str(p_)]
            D2wor,D2wor_knockoff = DATA[j]['without_randomization_Reverse']['p_'+str(p_)]
            D1w,D1w_knockoff = DATA[i]['with_randomization']['p_'+str(p_)]
            D2w,D2w_knockoff = DATA[j]['with_randomization']['p_'+str(p_)]
            SCORES['p_'+str(p_)].loc[itr,"without_randomization"] = MMD_checkQuality(D1wo,D1wo_knockoff,D2wo,D2wo_knockoff,set_seed=itr)
            SCORES['p_'+str(p_)].loc[itr,"without_randomization_Reverse"] = MMD_checkQuality(D1wor,D1wor_knockoff,D2wor,D2wor_knockoff,set_seed=itr)
            SCORES['p_'+str(p_)].loc[itr,"with_randomization"] = MMD_checkQuality(D1w,D1w_knockoff,D2w,D2w_knockoff,set_seed=itr)
        itr += 1



from scipy.stats import wilcoxon
P_Vals = pd.DataFrame(columns=["Forward_vs_Reverse","Forward_vs_Randomization","Reverse_vs_Randomization"],index=np.vectorize(lambda s:'p_'+str(s))(n_Cols))
'''
 **  Using Wilcoxon Signed-Rank test to check whether on average MMD_scores reduce by introducing randomization
'''
for p_ in n_Cols :
    D = SCORES['p_'+str(p_)]
    Red = D.without_randomization
    Grey = D.without_randomization_Reverse
    Green = D.with_randomization
    stat, p_value = wilcoxon(Red, Green,zero_method='pratt',alternative='two-sided')
    P_Vals.loc['p_'+str(p_),"Forward_vs_Randomization"] = p_value
    stat, p_value = wilcoxon(Grey, Green,zero_method='pratt',alternative='two-sided')
    P_Vals.loc['p_'+str(p_),"Reverse_vs_Randomization"] = p_value
    stat, p_value = wilcoxon(Red, Grey,zero_method='pratt',alternative='two-sided')
    P_Vals.loc['p_'+str(p_),"Forward_vs_Reverse"] = p_value
    sns.boxplot(data=D,palette=['red','darkgrey','green'])
    plt.axhline(color='black',linestyle='dashed')
    plt.ylabel('MMD_scores')
    plt.title('n_cols='+str(p_))
    plt.xticks(rotation=10)
    plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

absCORR = {'p_'+str(p_):{"without_randomization": pd.DataFrame(index=np.vectorize(lambda s:'col_'+str(s))(range(p_)),columns=np.vectorize(lambda s:'itr'+str(s))(range(n_itr))),
                         "without_randomization_Reverse": pd.DataFrame(index=np.vectorize(lambda s:'col_'+str(s))(range(p_)),columns=np.vectorize(lambda s:'itr'+str(s))(range(n_itr))),
                         "with_randomization": pd.DataFrame(index=np.vectorize(lambda s:'col_'+str(s))(range(p_)),columns=np.vectorize(lambda s:'itr'+str(s))(range(n_itr))) } for p_ in n_Cols}

for p_ in n_Cols:
    for itr in range(n_itr):
        D1w,D1w_knockoff = DATA[itr]['with_randomization']['p_'+str(p_)]
        absCORR['p_'+str(p_)]['with_randomization']['itr'+str(itr)] = pairwise_absCorrelation(D1w,D1w_knockoff)
        D1wo,D1wo_knockoff = DATA[itr]['without_randomization']['p_'+str(p_)]
        absCORR['p_'+str(p_)]['without_randomization']['itr'+str(itr)] = pairwise_absCorrelation(D1wo,D1wo_knockoff)
        D1wor,D1wor_knockoff = DATA[itr]['without_randomization_Reverse']['p_'+str(p_)]
        absCORR['p_'+str(p_)]['without_randomization_Reverse']['itr'+str(itr)] = pairwise_absCorrelation(D1wor,D1wor_knockoff)


from scipy.stats import spearmanr
checkTREND_ = pd.DataFrame(columns=['without_randomization','without_randomization_Reverse','with_randomization'],index=np.vectorize(lambda s:'p_'+str(s))(n_Cols))
checkTREND_pvalues = pd.DataFrame(columns=['without_randomization','without_randomization_Reverse','with_randomization'],index=np.vectorize(lambda s:'p_'+str(s))(n_Cols))
'''
 ** Checking whether the order(Forward or Reversed or Randomized) of columns in KnockOff matters
'''
for p_ in n_Cols:
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
    D0 = pd.melt(D,ignore_index=False)
    D0.rename(columns={'variable':'sKnockOff'},inplace=True)
    sns.lineplot(x=D0.index,y='value',hue='sKnockOff',data=D0,palette=['red','darkgrey','green']).set_xticks(range(0,p_,p_//10 if p_>10 else 1))
    plt.ylabel('absolute pairwise corr')
    plt.xlabel('column index ')
    plt.title('n_cols='+str(p_))
    plt.xticks(rotation=40)
    plt.show()
    D0 = pd.melt(D.loc[:,['without_randomization','without_randomization_Reverse']],ignore_index=False)
    D0.rename(columns={'variable':'sKnockOff'},inplace=True)
    sns.lineplot(x=D0.index,y='value',hue='sKnockOff',data=D0,palette=['red','darkgrey']).set_xticks(range(0,p_,p_//10 if p_>10 else 1))
    plt.ylabel('absolute pairwise corr')
    plt.xlabel('column index ')
    plt.title('n_cols='+str(p_))
    plt.xticks(rotation=40)
    plt.show()


# *****************************************************************************


"""
Created on Sun Nov 12 23:02:11 2023

Topic: Checking whether randomization of columns actually improves sKnockOff or not.
        All the columns in the DataMatrix are not independent here. Each column is correlated with its ten immediate neighbours.
@author: R.Nandi
"""


# working dir & useful Modules ..........................
from os import chdir
chdir("G:\PROJECTWORK_M.Stat_FINAL\SIMULATIONs\SIMULATION-3")

from sys import path
path.append("G:\PROJECTWORK_M.Stat_FINAL")
from KnockOff_101.Basics import *
from KnockOff_101 import Simulation_and_Visualization
from KnockOff_101.Compute_Multiple.KnockOff_Generating import sKnockOff,sKnockOff_Modified
from KnockOff_101.Diagnostics import MMD_checkQuality,pairwise_absCorrelation


#### SIMULATION ===============================================================
n_itr = 25
n_obs = 200
n_Cols = [40,30,20,10]
m = max(n_Cols)
Data_Format = lambda : {"without_randomization":{'p_'+str(p_): {} for p_ in n_Cols},
        "without_randomization_Reverse":{'p_'+str(p_): {} for p_ in n_Cols},
        "with_randomization":{'p_'+str(p_): {} for p_ in n_Cols}}
DATA = []

mu = [0]*m
Sigma = np.eye(m)
for k in range(1,11): Sigma += np.diag([0.5**k]*(m-k),k) + np.diag([0.5**k]*(m-k),-k)
plt.figure(figsize=(6,6))
sns.heatmap(Sigma,cmap="YlGnBu", annot=False,square=True)
plt.title("correlation matrix : Full")
plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def one_copy(i):
    X_Full = Simulation_and_Visualization.simulateJoint(n_obs,popln=lambda size: np.random.multivariate_normal(mu, Sigma,size))
    Data = Data_Format()
    for p_ in n_Cols:
        X_current = X_Full.iloc[:,:p_]
        Data['without_randomization']['p_'+str(p_)] = sKnockOff(X_current,Cat_or_Num(X_current),True)
        Data['without_randomization_Reverse']['p_'+str(p_)] = sKnockOff(X_current.iloc[:,::-1],Cat_or_Num(X_current.iloc[:,::-1]),True)
        Data['with_randomization']['p_'+str(p_)] = sKnockOff_Modified(X_current,Cat_or_Num(X_current),True)
    return Data

from joblib import Parallel, delayed
from multiprocessing import cpu_count
DATA += Parallel(n_jobs=cpu_count())(delayed(one_copy)(i) for i in range(n_itr))



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

n_itrPair = int((n_itr)*(n_itr-1)/2)
SCORES = {'p_'+str(p_): pd.DataFrame(columns=["without_randomization","without_randomization_Reverse","with_randomization"],index=range(n_itrPair)) for p_ in n_Cols}

itr = 0
for i in range(n_itr-1):
    for j in range(i+1,n_itr):
        for p_ in n_Cols:
            ## Data | p_  ...........................................
            D1wo,D1wo_knockoff = DATA[i]['without_randomization']['p_'+str(p_)]
            D2wo,D2wo_knockoff = DATA[j]['without_randomization']['p_'+str(p_)]
            D1wor,D1wor_knockoff = DATA[i]['without_randomization_Reverse']['p_'+str(p_)]
            D2wor,D2wor_knockoff = DATA[j]['without_randomization_Reverse']['p_'+str(p_)]
            D1w,D1w_knockoff = DATA[i]['with_randomization']['p_'+str(p_)]
            D2w,D2w_knockoff = DATA[j]['with_randomization']['p_'+str(p_)]
            SCORES['p_'+str(p_)].loc[itr,"without_randomization"] = MMD_checkQuality(D1wo,D1wo_knockoff,D2wo,D2wo_knockoff,set_seed=itr)
            SCORES['p_'+str(p_)].loc[itr,"without_randomization_Reverse"] = MMD_checkQuality(D1wor,D1wor_knockoff,D2wor,D2wor_knockoff,set_seed=itr)
            SCORES['p_'+str(p_)].loc[itr,"with_randomization"] = MMD_checkQuality(D1w,D1w_knockoff,D2w,D2w_knockoff,set_seed=itr)
        itr += 1



from scipy.stats import wilcoxon
P_Vals = pd.DataFrame(columns=["Forward_vs_Reverse","Forward_vs_Randomization","Reverse_vs_Randomization"],index=np.vectorize(lambda s:'p_'+str(s))(n_Cols))
'''
 **  Using Wilcoxon Signed-Rank test to check whether on average MMD_scores reduce by introducing randomization
'''
for p_ in n_Cols :
    D = SCORES['p_'+str(p_)]
    Red = D.without_randomization
    Grey = D.without_randomization_Reverse
    Green = D.with_randomization
    stat, p_value = wilcoxon(Red, Green,zero_method='pratt',alternative='two-sided')
    P_Vals.loc['p_'+str(p_),"Forward_vs_Randomization"] = p_value
    stat, p_value = wilcoxon(Grey, Green,zero_method='pratt',alternative='two-sided')
    P_Vals.loc['p_'+str(p_),"Reverse_vs_Randomization"] = p_value
    stat, p_value = wilcoxon(Red, Grey,zero_method='pratt',alternative='two-sided')
    P_Vals.loc['p_'+str(p_),"Forward_vs_Reverse"] = p_value
    sns.boxplot(data=D,palette=['red','darkgrey','green'])
    plt.axhline(color='black',linestyle='dashed')
    plt.ylabel('MMD_scores')
    plt.title('n_cols='+str(p_))
    plt.xticks(rotation=10)
    plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

absCORR = {'p_'+str(p_):{"without_randomization": pd.DataFrame(index=np.vectorize(lambda s:'col_'+str(s))(range(p_)),columns=np.vectorize(lambda s:'itr'+str(s))(range(n_itr))),
                         "without_randomization_Reverse": pd.DataFrame(index=np.vectorize(lambda s:'col_'+str(s))(range(p_)),columns=np.vectorize(lambda s:'itr'+str(s))(range(n_itr))),
                         "with_randomization": pd.DataFrame(index=np.vectorize(lambda s:'col_'+str(s))(range(p_)),columns=np.vectorize(lambda s:'itr'+str(s))(range(n_itr))) } for p_ in n_Cols}

for p_ in n_Cols:
    for itr in range(n_itr):
        D1w,D1w_knockoff = DATA[itr]['with_randomization']['p_'+str(p_)]
        absCORR['p_'+str(p_)]['with_randomization']['itr'+str(itr)] = pairwise_absCorrelation(D1w,D1w_knockoff)
        D1wo,D1wo_knockoff = DATA[itr]['without_randomization']['p_'+str(p_)]
        absCORR['p_'+str(p_)]['without_randomization']['itr'+str(itr)] = pairwise_absCorrelation(D1wo,D1wo_knockoff)
        D1wor,D1wor_knockoff = DATA[itr]['without_randomization_Reverse']['p_'+str(p_)]
        absCORR['p_'+str(p_)]['without_randomization_Reverse']['itr'+str(itr)] = pairwise_absCorrelation(D1wor,D1wor_knockoff)


from scipy.stats import spearmanr
checkTREND_ = pd.DataFrame(columns=['without_randomization','without_randomization_Reverse','with_randomization'],index=np.vectorize(lambda s:'p_'+str(s))(n_Cols))
checkTREND_pvalues = pd.DataFrame(columns=['without_randomization','without_randomization_Reverse','with_randomization'],index=np.vectorize(lambda s:'p_'+str(s))(n_Cols))
'''
 ** Checking whether the order(Forward or Reversed or Randomized) of columns in KnockOff matters
'''
for p_ in n_Cols:
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
    D0 = pd.melt(D,ignore_index=False)
    D0.rename(columns={'variable':'sKnockOff'},inplace=True)
    sns.lineplot(x=D0.index,y='value',hue='sKnockOff',data=D0,palette=['red','darkgrey','green']).set_xticks(range(0,p_,p_//10 if p_>10 else 1))
    plt.ylabel('absolute pairwise corr')
    plt.xlabel('column index ')
    plt.title('n_cols='+str(p_))
    plt.xticks(rotation=40)
    plt.show()
    D0 = pd.melt(D.loc[:,['without_randomization','without_randomization_Reverse']],ignore_index=False)
    D0.rename(columns={'variable':'sKnockOff'},inplace=True)
    sns.lineplot(x=D0.index,y='value',hue='sKnockOff',data=D0,palette=['red','darkgrey']).set_xticks(range(0,p_,p_//10 if p_>10 else 1))
    plt.ylabel('absolute pairwise corr')
    plt.xlabel('column index ')
    plt.title('n_cols='+str(p_))
    plt.xticks(rotation=40)
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
path.append("G:\PROJECTWORK_M.Stat_FINAL")
from KnockOff_101.Basics import *
from KnockOff_101 import Simulation_and_Visualization
from KnockOff_101.Compute_Multiple.KnockOff_Generating import sKnockOff,sKnockOff_Modified
from KnockOff_101.Diagnostics import MMD_checkQuality,pairwise_absCorrelation


#### SIMULATION-1 =============================================================
X = pd.read_excel("Concrete_Data.xls")

Data_withoutModification = sKnockOff(X,Cat_or_Num(X),True)
Data_withoutModification_Reverse = sKnockOff(X.iloc[:,::-1],Cat_or_Num(X.iloc[:,::-1]),True)
Data_withModification = sKnockOff_Modified(X,Cat_or_Num(X),True)

DATA1 = {"without_randomization":Data_withoutModification,
        "without_randomization_Reverse":Data_withoutModification_Reverse,
        "with_randomization":Data_withModification}


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

absCORR = pd.DataFrame(index=X.columns, columns=['without_randomization','without_randomization_Reverse','with_randomization'])
absCORR['without_randomization'] = pairwise_absCorrelation(Data_withoutModification[0],Data_withoutModification[1])
absCORR['without_randomization_Reverse'] = pairwise_absCorrelation(Data_withoutModification_Reverse[0],Data_withoutModification_Reverse[1])
absCORR['with_randomization'] = pairwise_absCorrelation(Data_withModification[0],Data_withModification[1])
D = pd.melt(absCORR,ignore_index=False)
D.rename(columns={'variable':'sKnockOff'},inplace=True)
plt.figure(figsize=(7,6))
sns.lineplot(x=D.index,y='value',hue='sKnockOff',data=D,palette=['red','darkgrey','green'])
plt.xticks(rotation=45)
plt.ylabel('absolute pairwise corr')
plt.xlabel('column index ')
plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Data,Data_knockoff = Data_withModification
Simulation_and_Visualization.Visual(Data,Data_knockoff,appendTitle=' | with_randomization',scale_the_corrplot=1)

Data,Data_knockoff = Data_withoutModification
Simulation_and_Visualization.Visual(Data,Data_knockoff,appendTitle=' | without_randomization',scale_the_corrplot=1)

Data,Data_knockoff = Data_withoutModification_Reverse
Data = Data.iloc[:,::-1]
Data_knockoff = Data_knockoff.iloc[:,::-1]
Simulation_and_Visualization.Visual(Data,Data_knockoff,appendTitle=' | without_randomization_Reverse',scale_the_corrplot=1)


# *****************************************************************************
##
###
####
###
##
#
#### SIMULATION-2 =============================================================

X2 = pd.read_csv("my_data.csv",header = None)
_,m = X2.shape

n_itr = 25
n_obs = 200
n_Cols = [m,25,10,5]
Data_Format = lambda : {"without_randomization":{'p_'+str(p_): {} for p_ in n_Cols},
        "without_randomization_Reverse":{'p_'+str(p_): {} for p_ in n_Cols},
        "with_randomization":{'p_'+str(p_): {} for p_ in n_Cols}}
DATA2 = []

mu = [0]*m
Sigma = X2.corr()
plt.figure(figsize=(6,6))
sns.heatmap(Sigma,cmap="YlGnBu", annot=False,square=True)
plt.title("correlation matrix : Full")
plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def one_copy(i):
    X_Full = Simulation_and_Visualization.simulateJoint(n_obs,popln=lambda size: np.random.multivariate_normal(mu, Sigma,size))
    Data = Data_Format()
    for p_ in n_Cols:
        X_current = X_Full.iloc[:,:p_]
        Data['without_randomization']['p_'+str(p_)] = sKnockOff(X_current,Cat_or_Num(X_current),True)
        Data['without_randomization_Reverse']['p_'+str(p_)] = sKnockOff(X_current.iloc[:,::-1],Cat_or_Num(X_current.iloc[:,::-1]),True)
        Data['with_randomization']['p_'+str(p_)] = sKnockOff_Modified(X_current,Cat_or_Num(X_current),True)
    return Data

from joblib import Parallel, delayed
from multiprocessing import cpu_count
DATA2 += Parallel(n_jobs=cpu_count())(delayed(one_copy)(i) for i in range(n_itr))


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

n_itrPair = int((n_itr)*(n_itr-1)/2)
SCORES = {'p_'+str(p_): pd.DataFrame(columns=["without_randomization","without_randomization_Reverse","with_randomization"],index=range(n_itrPair)) for p_ in n_Cols}

itr = 0
for i in range(n_itr-1):
    for j in range(i+1,n_itr):
        for p_ in n_Cols:
            ## Data | p_  ...........................................
            D1wo,D1wo_knockoff = DATA2[i]['without_randomization']['p_'+str(p_)]
            D2wo,D2wo_knockoff = DATA2[j]['without_randomization']['p_'+str(p_)]
            D1wor,D1wor_knockoff = DATA2[i]['without_randomization_Reverse']['p_'+str(p_)]
            D2wor,D2wor_knockoff = DATA2[j]['without_randomization_Reverse']['p_'+str(p_)]
            D1w,D1w_knockoff = DATA2[i]['with_randomization']['p_'+str(p_)]
            D2w,D2w_knockoff = DATA2[j]['with_randomization']['p_'+str(p_)]
            SCORES['p_'+str(p_)].loc[itr,"without_randomization"] = MMD_checkQuality(D1wo,D1wo_knockoff,D2wo,D2wo_knockoff,set_seed=itr)
            SCORES['p_'+str(p_)].loc[itr,"without_randomization_Reverse"] = MMD_checkQuality(D1wor,D1wor_knockoff,D2wor,D2wor_knockoff,set_seed=itr)
            SCORES['p_'+str(p_)].loc[itr,"with_randomization"] = MMD_checkQuality(D1w,D1w_knockoff,D2w,D2w_knockoff,set_seed=itr)
        itr += 1



from scipy.stats import wilcoxon
P_Vals = pd.DataFrame(columns=["Forward_vs_Reverse","Forward_vs_Randomization","Reverse_vs_Randomization"],index=np.vectorize(lambda s:'p_'+str(s))(n_Cols))
'''
 **  Using Wilcoxon Signed-Rank test to check whether on average MMD_scores reduce by introducing randomization
'''
for p_ in n_Cols :
    D = SCORES['p_'+str(p_)]
    Red = D.without_randomization
    Grey = D.without_randomization_Reverse
    Green = D.with_randomization
    stat, p_value = wilcoxon(Red, Green,zero_method='pratt',alternative='two-sided')
    P_Vals.loc['p_'+str(p_),"Forward_vs_Randomization"] = p_value
    stat, p_value = wilcoxon(Grey, Green,zero_method='pratt',alternative='two-sided')
    P_Vals.loc['p_'+str(p_),"Reverse_vs_Randomization"] = p_value
    stat, p_value = wilcoxon(Red, Grey,zero_method='pratt',alternative='two-sided')
    P_Vals.loc['p_'+str(p_),"Forward_vs_Reverse"] = p_value
    sns.boxplot(data=D,palette=['red','darkgrey','green'])
    plt.axhline(color='black',linestyle='dashed')
    plt.ylabel('MMD_scores')
    plt.title('n_cols='+str(p_))
    plt.xticks(rotation=10)
    plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

absCORR2 = {'p_'+str(p_):{"without_randomization": pd.DataFrame(index=np.vectorize(lambda s:'col_'+str(s))(range(p_)),columns=np.vectorize(lambda s:'itr'+str(s))(range(n_itr))),
                         "without_randomization_Reverse": pd.DataFrame(index=np.vectorize(lambda s:'col_'+str(s))(range(p_)),columns=np.vectorize(lambda s:'itr'+str(s))(range(n_itr))),
                         "with_randomization": pd.DataFrame(index=np.vectorize(lambda s:'col_'+str(s))(range(p_)),columns=np.vectorize(lambda s:'itr'+str(s))(range(n_itr))) } for p_ in n_Cols}

for p_ in n_Cols:
    for itr in range(n_itr):
        D1w,D1w_knockoff = DATA2[itr]['with_randomization']['p_'+str(p_)]
        absCORR2['p_'+str(p_)]['with_randomization']['itr'+str(itr)] = pairwise_absCorrelation(D1w,D1w_knockoff)
        D1wo,D1wo_knockoff = DATA2[itr]['without_randomization']['p_'+str(p_)]
        absCORR2['p_'+str(p_)]['without_randomization']['itr'+str(itr)] = pairwise_absCorrelation(D1wo,D1wo_knockoff)
        D1wor,D1wor_knockoff = DATA2[itr]['without_randomization_Reverse']['p_'+str(p_)]
        absCORR2['p_'+str(p_)]['without_randomization_Reverse']['itr'+str(itr)] = pairwise_absCorrelation(D1wor,D1wor_knockoff)


from scipy.stats import spearmanr
checkTREND_ = pd.DataFrame(columns=['without_randomization','without_randomization_Reverse','with_randomization'],index=np.vectorize(lambda s:'p_'+str(s))(n_Cols))
checkTREND_pvalues = pd.DataFrame(columns=['without_randomization','without_randomization_Reverse','with_randomization'],index=np.vectorize(lambda s:'p_'+str(s))(n_Cols))
'''
 ** Checking whether the order(Forward or Reversed or Randomized) of columns in KnockOff matters
'''
for p_ in n_Cols:
    D = pd.DataFrame({'without_randomization': absCORR2['p_'+str(p_)]['without_randomization'].mean(axis=1),
    'without_randomization_Reverse': absCORR2['p_'+str(p_)]['without_randomization_Reverse'].mean(axis=1),
    'with_randomization': absCORR2['p_'+str(p_)]['with_randomization'].mean(axis=1)})
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
    D0 = pd.melt(D,ignore_index=False)
    D0.rename(columns={'variable':'sKnockOff'},inplace=True)
    sns.lineplot(x=D0.index,y='value',hue='sKnockOff',data=D0,palette=['red','darkgrey','green']).set_xticks(range(0,p_,p_//10 if p_>10 else 1))
    plt.ylabel('absolute pairwise corr')
    plt.xlabel('column index ')
    plt.title('n_cols='+str(p_))
    plt.xticks(rotation=40)
    plt.show()
    D0 = pd.melt(D.loc[:,['without_randomization','without_randomization_Reverse']],ignore_index=False)
    D0.rename(columns={'variable':'sKnockOff'},inplace=True)
    sns.lineplot(x=D0.index,y='value',hue='sKnockOff',data=D0,palette=['red','darkgrey']).set_xticks(range(0,p_,p_//10 if p_>10 else 1))
    plt.ylabel('absolute pairwise corr')
    plt.xlabel('column index ')
    plt.title('n_cols='+str(p_))
    plt.xticks(rotation=40)
    plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Data,Data_knockoff = DATA2[np.random.randint(0,n_itr)]['with_randomization']['p_'+str(m)]
Simulation_and_Visualization.Visual(Data,Data_knockoff,appendTitle=' | with_randomization',scale_the_corrplot=0.5)

Data,Data_knockoff = DATA2[np.random.randint(0,n_itr)]['without_randomization']['p_'+str(m)]
Simulation_and_Visualization.Visual(Data,Data_knockoff,appendTitle=' | without_randomization',scale_the_corrplot=0.5)

Data,Data_knockoff = DATA2[np.random.randint(0,n_itr)]['without_randomization_Reverse']['p_'+str(m)]
Data = Data.iloc[:,::-1]
Data_knockoff = Data_knockoff.iloc[:,::-1]
Simulation_and_Visualization.Visual(Data,Data_knockoff,appendTitle=' | without_randomization_Reverse',scale_the_corrplot=0.5)


# *****************************************************************************
