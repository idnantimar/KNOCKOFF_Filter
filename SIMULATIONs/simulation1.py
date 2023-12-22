"""
Created on Fri Sep  8 01:06:10 2023

Topic: Checking whether randomization of columns actually improves sKnockOff or not.
        All the columns in the DataMatrix are independent here.
@author: R.Nandi
"""

# working dir & useful Modules ..........................
from os import chdir
chdir("E:\PROJECTWORK_M.Stat_FINAL\FINAL_PRESENTATION\SIMULATIONs\SIMULATION-1")

from sys import path
path.append("E:\PROJECTWORK_M.Stat_FINAL")
from KnockOff_101.Basics import *
from KnockOff_101 import Simulation_and_Visualization
from KnockOff_101.Compute_Multiple.KnockOff_Generating import sKnockOff_KernelTrick,KnockOff_Reshuffled
from KnockOff_101.Comparisons_and_checkQuality import pairwise_absCorrelation

#### SIMULATION ===============================================================
n_itr = 10
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
        Data_withoutModification = sKnockOff_KernelTrick(X_current,Cat_or_Num(X_current),True)
        Data_withModification = KnockOff_Reshuffled(X_current,Cat_or_Num(X_current),True,method=sKnockOff_KernelTrick)
        DATA['without_randomization']['p_'+str(p_)]['itr'+str(itr)] = Data_withoutModification
        DATA['with_randomization']['p_'+str(p_)]['itr'+str(itr)] = Data_withModification


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
chdir("E:\PROJECTWORK_M.Stat_FINAL\FINAL_PRESENTATION\SIMULATIONs\SIMULATION-2")

from sys import path
path.append("E:\PROJECTWORK_M.Stat_FINAL")
from KnockOff_101.Basics import *
from KnockOff_101 import Simulation_and_Visualization
from KnockOff_101.Compute_Multiple.KnockOff_Generating import sKnockOff_KernelTrick,KnockOff_Reshuffled
from KnockOff_101.Comparisons_and_checkQuality import EnergyDistance_test,energy_distance,energy_test,pairwise_absCorrelation


#### SIMULATION ===============================================================
n_itr = 25
n_obs = 200
n_Cols = [45,35,25,15,5]
m = max(n_Cols)
Data_Format = lambda : {"without_randomization":{'p_'+str(p_): {} for p_ in n_Cols},
        "without_randomization_Reverse":{'p_'+str(p_): {} for p_ in n_Cols},
        "with_randomization":{'p_'+str(p_): {} for p_ in n_Cols}}
DATA = []

mu = [0]*m
Sigma = np.eye(m)
for k in range(1,3): Sigma += np.diag([(0.5)**k]*(m-k),k) + np.diag([(0.5)**k]*(m-k),-k)
plt.figure(figsize=(6,6))
sns.heatmap(Sigma,cmap="YlGnBu", annot=False,square=True)
plt.title("correlation matrix : Full")
plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def one_copy(i):
    X_Full = Simulation_and_Visualization.simulateJoint(n_obs,popln=lambda size,rng: rng.multivariate_normal(mu, Sigma,size))
    Data = Data_Format()
    for p_ in n_Cols:
        X_current = X_Full.iloc[:,:p_]
        Data['without_randomization']['p_'+str(p_)] = sKnockOff_KernelTrick(X_current,Cat_or_Num(X_current),True)
        Data['without_randomization_Reverse']['p_'+str(p_)] = sKnockOff_KernelTrick(X_current.iloc[:,::-1],Cat_or_Num(X_current.iloc[:,::-1]),True)
        Data['with_randomization']['p_'+str(p_)] = KnockOff_Reshuffled(X_current,Cat_or_Num(X_current),True,method=sKnockOff_KernelTrick)
    return Data

from joblib import Parallel, delayed
from multiprocessing import cpu_count
DATA += Parallel(n_jobs=cpu_count())(delayed(one_copy)(i) for i in range(n_itr))



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

n_itrPair = int((n_itr)*(n_itr-1)/2)
SCORES = {'p_'+str(p_): pd.DataFrame(columns=["without_randomization","without_randomization_Reverse","with_randomization"],index=range(n_itrPair)) for p_ in n_Cols}
p_Values = {'p_'+str(p_): pd.DataFrame(columns=["without_randomization","without_randomization_Reverse","with_randomization"],index=range(n_itrPair)) for p_ in n_Cols}
COMPARE = pd.DataFrame(columns=np.vectorize(lambda s:'p_'+str(s))(n_Cols),index=range(n_itr))
COMPARE_p = pd.DataFrame(columns=np.vectorize(lambda s:'p_'+str(s))(n_Cols),index=range(n_itr))


for p_ in n_Cols:
    itr = 0
    for i in range(n_itr):
        D1wo,D1wo_knockoff = DATA[i]['without_randomization']['p_'+str(p_)]
        D1w,D1w_knockoff = DATA[i]['with_randomization']['p_'+str(p_)]
        COMPARE.loc[i,'p_'+str(p_)] = energy_distance(D1w_knockoff,D1wo_knockoff)
        p_val,_ = energy_test(D1w_knockoff,D1wo_knockoff,num_resamples=100)
        COMPARE_p.loc[i,'p_'+str(p_)] = p_val
        D1wor,D1wor_knockoff = DATA[i]['without_randomization_Reverse']['p_'+str(p_)]
        for j in range(i+1,n_itr):
            D2wo,D2wo_knockoff = DATA[j]['without_randomization']['p_'+str(p_)]
            score,p_val = EnergyDistance_test(D1wo,D1wo_knockoff,D2wo,D2wo_knockoff,set_seed=itr,num_resamples=20)
            SCORES['p_'+str(p_)].loc[itr,"without_randomization"] = score
            p_Values['p_'+str(p_)].loc[itr,"without_randomization"] = p_val
            D2wor,D2wor_knockoff = DATA[j]['without_randomization_Reverse']['p_'+str(p_)]
            score,p_val = EnergyDistance_test(D1wor,D1wor_knockoff,D2wor,D2wor_knockoff,set_seed=itr,num_resamples=20)
            SCORES['p_'+str(p_)].loc[itr,"without_randomization_Reverse"] = score
            p_Values['p_'+str(p_)].loc[itr,"without_randomization_Reverse"] = p_val
            D2w,D2w_knockoff = DATA[j]['with_randomization']['p_'+str(p_)]
            score,p_val = EnergyDistance_test(D1w,D1w_knockoff,D2w,D2w_knockoff,set_seed=itr,num_resamples=20)
            SCORES['p_'+str(p_)].loc[itr,"with_randomization"] = score
            p_Values['p_'+str(p_)].loc[itr,"with_randomization"] = p_val
            itr += 1


sns.boxplot(data=COMPARE.iloc[:,::-1],palette=['grey']*len(n_Cols))
plt.axhline(color='black',linestyle='dashed')
plt.ylabel('energy distance')
plt.title("with vs without randomization")
plt.xticks(rotation=10)
plt.show()

sns.boxplot(data=COMPARE_p.iloc[:,::-1],palette=['grey']*len(n_Cols))
plt.ylabel('p values')
plt.title("with vs without randomization")
plt.xticks(rotation=10)
plt.ylim(0,1.01)
plt.show()


for p_ in n_Cols :
    D = SCORES['p_'+str(p_)]
    sns.boxplot(data=D,palette=['red','darkgrey','green'])
    plt.ylabel('energy distance score')
    plt.title('n_cols='+str(p_))
    plt.xticks(rotation=10)
    plt.show()
    D = p_Values['p_'+str(p_)]
    D.plot(kind="kde",subplots=[['without_randomization','without_randomization_Reverse']],color=['red','darkgrey','green'],sharex=True,title="p values | n_cols="+str(p_))
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



for p_ in n_Cols:
    D = pd.DataFrame({'without_randomization': absCORR['p_'+str(p_)]['without_randomization'].mean(axis=1),
    'without_randomization_Reverse': absCORR['p_'+str(p_)]['without_randomization_Reverse'].mean(axis=1),
    'with_randomization': absCORR['p_'+str(p_)]['with_randomization'].mean(axis=1)})
    Red = D['without_randomization']
    Grey = D['without_randomization_Reverse']
    Green = D['with_randomization']
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
chdir("E:\PROJECTWORK_M.Stat_FINAL\FINAL_PRESENTATION\SIMULATIONs\SIMULATION-3")

from sys import path
path.append("E:\PROJECTWORK_M.Stat_FINAL")
from KnockOff_101.Basics import *
from KnockOff_101 import Simulation_and_Visualization
from KnockOff_101.Compute_Multiple.KnockOff_Generating import sKnockOff_KernelTrick,KnockOff_Reshuffled
from KnockOff_101.Comparisons_and_checkQuality import EnergyDistance_test,energy_distance,energy_test,pairwise_absCorrelation


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
for k in range(1,11): Sigma += np.diag([(-0.5)**k]*(m-k),k) + np.diag([(-0.5)**k]*(m-k),-k)
plt.figure(figsize=(6,6))
sns.heatmap(Sigma,cmap="YlGnBu", annot=False,square=True)
plt.title("correlation matrix : Full")
plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def one_copy(i):
    X_Full = Simulation_and_Visualization.simulateJoint(n_obs,popln=lambda size,rng: rng.multivariate_normal(mu, Sigma,size),generator=RNG())
    Data = Data_Format()
    for p_ in n_Cols:
        X_current = X_Full.iloc[:,:p_]
        Data['without_randomization']['p_'+str(p_)] = sKnockOff_KernelTrick(X_current,Cat_or_Num(X_current),True)
        Data['without_randomization_Reverse']['p_'+str(p_)] = sKnockOff_KernelTrick(X_current.iloc[:,::-1],Cat_or_Num(X_current.iloc[:,::-1]),True)
        Data['with_randomization']['p_'+str(p_)] = KnockOff_Reshuffled(X_current,Cat_or_Num(X_current),True,method=sKnockOff_KernelTrick)
    return Data

from joblib import Parallel, delayed
from multiprocessing import cpu_count
DATA += Parallel(n_jobs=cpu_count())(delayed(one_copy)(i) for i in range(n_itr))



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

n_itrPair = int((n_itr)*(n_itr-1)/2)
SCORES = {'p_'+str(p_): pd.DataFrame(columns=["without_randomization","without_randomization_Reverse","with_randomization"],index=range(n_itrPair)) for p_ in n_Cols}
p_Values = {'p_'+str(p_): pd.DataFrame(columns=["without_randomization","without_randomization_Reverse","with_randomization"],index=range(n_itrPair)) for p_ in n_Cols}
COMPARE = pd.DataFrame(columns=np.vectorize(lambda s:'p_'+str(s))(n_Cols),index=range(n_itr))
COMPARE_p = pd.DataFrame(columns=np.vectorize(lambda s:'p_'+str(s))(n_Cols),index=range(n_itr))


for p_ in n_Cols:
    itr = 0
    for i in range(n_itr):
        D1wo,D1wo_knockoff = DATA[i]['without_randomization']['p_'+str(p_)]
        D1w,D1w_knockoff = DATA[i]['with_randomization']['p_'+str(p_)]
        COMPARE.loc[i,'p_'+str(p_)] = energy_distance(D1w_knockoff,D1wo_knockoff)
        p_val,_ = energy_test(D1w_knockoff,D1wo_knockoff,num_resamples=100)
        COMPARE_p.loc[i,'p_'+str(p_)] = p_val
        D1wor,D1wor_knockoff = DATA[i]['without_randomization_Reverse']['p_'+str(p_)]
        for j in range(i+1,n_itr):
            D2wo,D2wo_knockoff = DATA[j]['without_randomization']['p_'+str(p_)]
            score,p_val = EnergyDistance_test(D1wo,D1wo_knockoff,D2wo,D2wo_knockoff,set_seed=itr,num_resamples=20)
            SCORES['p_'+str(p_)].loc[itr,"without_randomization"] = score
            p_Values['p_'+str(p_)].loc[itr,"without_randomization"] = p_val
            D2wor,D2wor_knockoff = DATA[j]['without_randomization_Reverse']['p_'+str(p_)]
            score,p_val = EnergyDistance_test(D1wor,D1wor_knockoff,D2wor,D2wor_knockoff,set_seed=itr,num_resamples=20)
            SCORES['p_'+str(p_)].loc[itr,"without_randomization_Reverse"] = score
            p_Values['p_'+str(p_)].loc[itr,"without_randomization_Reverse"] = p_val
            D2w,D2w_knockoff = DATA[j]['with_randomization']['p_'+str(p_)]
            score,p_val = EnergyDistance_test(D1w,D1w_knockoff,D2w,D2w_knockoff,set_seed=itr,num_resamples=20)
            SCORES['p_'+str(p_)].loc[itr,"with_randomization"] = score
            p_Values['p_'+str(p_)].loc[itr,"with_randomization"] = p_val
            itr += 1


sns.boxplot(data=COMPARE.iloc[:,::-1],palette=['grey']*len(n_Cols))
plt.ylabel('energy distance')
plt.title("with vs without randomization")
plt.xticks(rotation=10)
plt.show()

sns.boxplot(data=COMPARE_p.iloc[:,::-1],palette=['grey']*len(n_Cols))
plt.ylabel('p values')
plt.title("with vs without randomization")
plt.xticks(rotation=10)
plt.ylim(0,1.01)
plt.show()


for p_ in n_Cols :
    D = SCORES['p_'+str(p_)]
    sns.boxplot(data=D,palette=['red','darkgrey','green'])
    plt.axhline(color='black',linestyle='dashed')
    plt.ylabel('energy distance score')
    plt.title('n_cols='+str(p_))
    plt.xticks(rotation=10)
    plt.show()
    D = p_Values['p_'+str(p_)]
    D.plot(kind="kde",subplots=[['without_randomization','without_randomization_Reverse']],color=['red','darkgrey','green'],title="p values | n_cols="+str(p_),sharex=True)
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



for p_ in n_Cols:
    D = pd.DataFrame({'without_randomization': absCORR['p_'+str(p_)]['without_randomization'].mean(axis=1),
    'without_randomization_Reverse': absCORR['p_'+str(p_)]['without_randomization_Reverse'].mean(axis=1),
    'with_randomization': absCORR['p_'+str(p_)]['with_randomization'].mean(axis=1)})
    Red = D['without_randomization']
    Grey = D['without_randomization_Reverse']
    Green = D['with_randomization']
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
chdir("E:\PROJECTWORK_M.Stat_FINAL\FINAL_PRESENTATION\SIMULATIONs\SIMULATION-4")

from sys import path
path.append("E:\PROJECTWORK_M.Stat_FINAL")
from KnockOff_101.Basics import *
from KnockOff_101 import Simulation_and_Visualization
from KnockOff_101.Compute_Multiple.KnockOff_Generating import sKnockOff_KernelTrick,KnockOff_Reshuffled
from KnockOff_101.Comparisons_and_checkQuality import energy_test,energy_distance,pairwise_absCorrelation


#### SIMULATION ===============================================================

X = pd.read_excel("my_data.xlsx",header = None)
_,m = X.shape

n_itr = 25
Data_Format = lambda : {"without_randomization": {},
        "without_randomization_Reverse":{},
        "with_randomization":{}}


mu = [0]*m
Sigma = X.corr()
plt.figure(figsize=(6,6))
sns.heatmap(Sigma,cmap="YlGnBu", annot=False,square=True)
plt.title("correlation matrix : Full")
plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def one_copy(i):
    Data = Data_Format()
    is_Cat = Cat_or_Num(X)
    Data['without_randomization'] = sKnockOff_KernelTrick(X,is_Cat,True)
    Data['without_randomization_Reverse'] = sKnockOff_KernelTrick(X.iloc[:,::-1],is_Cat[::-1],True)
    Data['with_randomization'] = KnockOff_Reshuffled(X,is_Cat,True,method=sKnockOff_KernelTrick)
    return Data

from joblib import Parallel, delayed
from multiprocessing import cpu_count
DATA = Parallel(n_jobs=cpu_count())(delayed(one_copy)(i) for i in range(n_itr))


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

SCORES = pd.Series(index=range(n_itr),dtype=float)
p_Vals = pd.Series(index=range(n_itr),dtype=float)


for i in range(n_itr):
    D1wo,D1wo_knockoff = DATA[i]['without_randomization']
    D1w,D1w_knockoff = DATA[i]['with_randomization']
    p_val,_ = energy_test(D1w_knockoff,D1wo_knockoff,num_resamples=100)
    SCORES[i] = energy_distance(D1w_knockoff,D1wo_knockoff)
    p_Vals[i] = p_val

SCORES.plot(kind='box',vert=False,ylabel="energy distance",title="with vs without randomization")
plt.axvline(color='black',linestyle='dashed')
plt.show()

(p_Vals).plot(kind='hist',title="p values",xlim=(0,1))
plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

absCORR = {"without_randomization": pd.DataFrame(index=np.array(range(m)).astype(str),columns=np.vectorize(lambda s:'itr'+str(s))(range(n_itr))),
           "without_randomization_Reverse": pd.DataFrame(index=np.array(range(m)).astype(str),columns=np.vectorize(lambda s:'itr'+str(s))(range(n_itr))),
           "with_randomization": pd.DataFrame(index=np.array(range(m)).astype(str),columns=np.vectorize(lambda s:'itr'+str(s))(range(n_itr))) }

for itr in range(n_itr):
    D1w,D1w_knockoff = DATA[itr]['with_randomization']
    absCORR['with_randomization']['itr'+str(itr)] = pairwise_absCorrelation(D1w,D1w_knockoff)
    D1wo,D1wo_knockoff = DATA[itr]['without_randomization']
    absCORR['without_randomization']['itr'+str(itr)] = pairwise_absCorrelation(D1wo,D1wo_knockoff)
    D1wor,D1wor_knockoff = DATA[itr]['without_randomization_Reverse']
    absCORR['without_randomization_Reverse']['itr'+str(itr)] = pairwise_absCorrelation(D1wor,D1wor_knockoff)



D = pd.DataFrame({'without_randomization': absCORR['without_randomization'].mean(axis=1),
                  'without_randomization_Reverse': absCORR['without_randomization_Reverse'].mean(axis=1),
                  'with_randomization': absCORR['with_randomization'].mean(axis=1)})
D0 = pd.melt(D,ignore_index=False)
D0.rename(columns={'variable':'sKnockOff'},inplace=True)
sns.lineplot(x=D0.index,y='value',hue='sKnockOff',data=D0,palette=['red','darkgrey','green']).set_xticks(range(0,m,m//10))
plt.ylabel('absolute pairwise corr')
plt.xlabel('column index ')
plt.title('n_cols='+str(m))
plt.xticks(rotation=40)
plt.show()
D = D.iloc[:,:-1]-D.iloc[:,-1].to_numpy().reshape(-1,1)
D.columns = ["forward-random","reverse-random"]
D0 = pd.melt(D,ignore_index=False)
D0.rename(columns={'variable':'sKnockOff'},inplace=True)
sns.lineplot(x=D0.index,y='value',hue='sKnockOff',data=D0,palette=['red','darkgrey']).set_xticks(range(0,m,m//10))
plt.ylabel('absolute pairwise corr difference')
plt.xlabel('column index ')
plt.title("without_randomization - with_randomization")
plt.xticks(rotation=40)
plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Data,Data_knockoff = DATA[np.random.randint(0,n_itr)]['with_randomization']
Simulation_and_Visualization.Visual(Data,Data_knockoff,appendTitle=' | with_randomization',scale_the_corrplot=0.25)

Data,Data_knockoff = DATA[np.random.randint(0,n_itr)]['without_randomization']
Simulation_and_Visualization.Visual(Data,Data_knockoff,appendTitle=' | without_randomization',scale_the_corrplot=0.25)

Data,Data_knockoff = DATA[np.random.randint(0,n_itr)]['without_randomization_Reverse']
Data = Data.iloc[:,::-1]
Data_knockoff = Data_knockoff.iloc[:,::-1]
Simulation_and_Visualization.Visual(Data,Data_knockoff,appendTitle=' | without_randomization_Reverse',scale_the_corrplot=0.25)


# *****************************************************************************
