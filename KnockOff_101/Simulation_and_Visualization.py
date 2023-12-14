"""
Created on Sat Sep 23 15:17:35 2023

Topic: Simulates random DataMatrix and Gives some visualization
@author: R.Nandi
"""

from .Basics import *



#### Simulate Samples =========================================================

def simulateIndependent(n_obs,col_type,
                        NUM=lambda size,rng: rng.normal(0,1,size),CAT=lambda size,rng: rng.choice(['A','B','C','D'],size,replace=True),
                        generator=RNG()):
    """
    A function to simulate a DataMatrix , where each column is sampled from some population , independent of each other.

    Parameters
    ----------
    n_obs : int
        Number of observations per column.
    col_type : tuple as (N,C) or array as [True,False,True,True,True,...]
        * If tuple, 1st element N denotes number of Numerical columns , 2nd element C denotes number of Categorical columns. When only one kind of columns are needed, input the other element as 0
        * If array as [True,False,...] , True denotes which position is for Numerical column, False denotes which position is for Categorical column.
    NUM : a function ;  default is lambda size: normal(0,1,size)
        Mentions from where to sample Numerical columns.
    CAT : a function ; default is lambda size: choice(['A','B','C','D'],size,replace=True))
        Mentions from where to sample Categorical columns.
    generator : random number generator ; default np.random.default_rng(seed=None)

    Returns
    -------
    DataFrame ; where each row is an observation , each column is a feature

    """
    if isinstance(col_type,tuple) :
        N,C = col_type
        col_type = [True]*N+[False]*C
    else :
        N,C = sum(col_type),len(col_type)-sum(col_type)

    NUM_Block = [pd.Series(NUM(n_obs,generator)) for _ in range(N)]
    CAT_Block = [pd.Series(CAT(n_obs,generator)) for _ in range(C)]
    BLOCKs = pd.concat(NUM_Block+CAT_Block,axis=1)
    BLOCKs.columns = [('Num'+str(i)) for i in range(N)] + [('Cat'+str(j)) for j in range(C)]
    col_order = []
    i,j,k = 0,0,0
    tot = N+C
    while k<tot:
        if col_type[k]:
            col_order += ['Num'+str(i)]
            i += 1
        else :
            col_order += ['Cat'+str(j)]
            j += 1
        k = i+j

    return BLOCKs[col_order]



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def simulateJoint(n_obs,popln=lambda size_,rng: rng.multivariate_normal(rng.normal(size=10), np.corrcoef(rng.random(size=(10,100))),size=size_),
                  generator=RNG()):
    """
    A function to simulate a DataMatrix from a joint-distribution specified.

    Parameters
    ----------
    n_obs : int
        Number of observations per column.
    popln : a function mentioning the joint distribution of the columns ; default is lambda size: multivariate_normal(normal(10), np.corrcoef(random(10,100)),size)
    generator : random number generator ; default np.random.default_rng(seed=None)

    Returns
    -------
    DataFrame ; where each row is an observation , each column is a feature

    """

    OUT = pd.DataFrame(popln(n_obs,generator))
    OUT.columns = ['col_'+str(i) for i in OUT.columns]

    return OUT




# *****************************************************************************
##
###
####
###
##
# Simulate Response|observed_Predictor ========================================

from tensorflow.nn import softmax as logOdds_to_probability
from os import environ as _environ
_environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from string import ascii_uppercase



def Simulate_Categorical(X=simulateIndependent(5,(2,0)),
                         Phi= lambda x: x, return_FeatureMap=False,
                         Beta_coeffs = np.random.randn(3,2),
                         categories = list(ascii_uppercase),
                         generator = RNG()):
    """
    A function to simulate Categorical Response based on Multinomial conditional probability & an observed DataMatrix.

    Parameters
    ----------
    X : DataFrame or 2D-array ; size=(n,p)
        The observed Predictors.
    Phi : a function that takes an 1D-array as input , returns an 1D-array
        The FeatureMap to be applied on each rows of X. Phi(X) has size=(n,m). Default is Linear FeatureMap.
    return_FeatureMap : bool ; default False
        Whether Phi(X) will be returned(True) or not(False) at the end.
    Beta_coeffs : 2D-array ; size=(k,m)
        The coefficients to be used to model the Multinomial probabilities. Each row corresponds to a response category.
        Setting all the coeficients to 0 results in independence of Y & X.
    categories : list or 1D-array ; length=k
        The list of response category names.
    generator : random number generator ; default np.random.default_rng(seed=None)

    Returns
    -------
    Series ; Contains the name of the simulated category at every index.

    """
    X = pd.DataFrame(X).copy()
    X = X.apply(Phi,axis=1,result_type='expand')
    XBeta = np.matmul(X,Beta_coeffs.T)
    Probs = logOdds_to_probability(XBeta).numpy()
    Y = generator.multinomial(1,Probs).argmax(axis=1)
    Y = pd.Series(Y,index=X.index)
    Y = Y.apply(lambda i: categories[i])

    return Y if not return_FeatureMap else (Y,X)




# *****************************************************************************
##
###
####
###
##
#### Visualization ============================================================
'''
 ** Gives some visualization of 'how the data looks like' or 'how well a generated knockoff copy is'
 ** For very large or very small data, maybe problematic

'''


def Visual(X,X_knockoff=None,appendTitle='',Means=False,KDE=False,scale_the_corrplot=0.25):
    """ Gives some visualization of the combined data [X,X_knockoff]
    """
    n,p = X.shape
    names = X.columns
    is_Cat = Cat_or_Num(X)
    if X_knockoff is None:
        Xcombined = X
        isCat_Combined = list(is_Cat)
    else:
        Xcombined = pd.concat([X,X_knockoff],axis=1)
        isCat_Combined = list(is_Cat)+list(is_Cat)
    k = p-sum(is_Cat)

   ## for Numerical variables ------------------------------
    if k:
        XcombinedN = Xcombined.iloc[:,np.invert(isCat_Combined)].copy()
        #> means .......................................
        if Means:
            means = XcombinedN.apply(lambda x: np.mean(x)).to_numpy()
            pd.DataFrame({'original': means[:k],
                          'knockoff': means[k:]} if (X_knockoff is not None) else {'data':means} ,index=names[np.invert(is_Cat)]).plot(kind='barh',title='Means'+appendTitle,rot=45,figsize=(8,0.25*k))
            plt.show()
        # KDE ..........................................
        if KDE:
            if X_knockoff is not None:
                fig, axes = plt.subplots(2,1,figsize=(7,12))
                plt.subplots_adjust(hspace=0.2)
                XcombinedN.iloc[:,:k].plot(kind='density',title='Original'+appendTitle,ax=axes[0],grid=True)
                XcombinedN.iloc[:,k:].plot(kind='density',title='Knockoff'+appendTitle,ax=axes[1],grid=True)
                plt.show()
            else: XcombinedN.plot(kind='density',title='data'+appendTitle,grid=True)

        #> variance ....................................
        if X_knockoff is not None : XcombinedN.insert(loc=int(k),column='-',value=np.zeros((n,)))
        plt.figure(figsize=(scale_the_corrplot*k,scale_the_corrplot*k))
        sns.heatmap(XcombinedN.cov(),cmap="YlGnBu", annot=False,square=True)
        plt.title(('combined ' if (X_knockoff is not None) else '')+'Covariance Heatmap'+appendTitle)
        plt.show()

   ## for Categorical Variables ----------------------------
    XcombinedC = Xcombined.iloc[:,isCat_Combined]
    namesC = XcombinedC.columns
    k = p-k
    if k:
        r,c = int(np.ceil(k/2)) , (2 if k!=1 else 1)
        fig, axes = plt.subplots(r,c, figsize= (3*c,3*r) if k!=1 else (3,3.5))
        fig.suptitle('Counts'+appendTitle)
        plt.subplots_adjust(hspace=0.5)
        for i in range(k):
            current_plot = axes[i//2 , i%2] if k>2 else (axes[i] if k==2 else axes)
            pd.DataFrame({'original': Counter(XcombinedC.iloc[:,i]),
                          'knockoff': Counter(XcombinedC.iloc[:,i+k])} if (X_knockoff is not None) else {'data':Counter(XcombinedC.iloc[:,i])}).plot(kind='bar',ax=current_plot, title=str(namesC[i]),rot=30,legend=np.invert(bool(i)))
        plt.show()




# *****************************************************************************
