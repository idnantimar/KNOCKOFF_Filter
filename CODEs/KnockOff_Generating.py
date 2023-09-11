"""
Created on Sun Sep  3 23:47:24 2023

Topic: Generating KnockOff copy of data matrix
@author: R.Nandi
"""

from Basics import *


#### Sequential KnockOff generation ===========================================
'''
 ** Generates KnockOff copy of data matrix X ,
    treating X as a random observtion from an underlying population
 ** For j=1,2,...,p this method sequentially generates
        Xj_knockoff ~ Xj|(X[-j],X_knockoff[1:j-1])
 ** For Continuous Xj this conditional distn. is fitted as Normal
      Categorical Xj this conditional distn. is fitte as Multinomial
    to estimate the underlying parameters
 ** The restriction 'Xj & Xj_knockoff should be as uncorrelated as possible' has not implemented explicitly in this method

'''

from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import GridSearchCV ,RepeatedStratifiedKFold,RepeatedKFold
import warnings
from sklearn.exceptions import ConvergenceWarning

def sKnockOff(X, is_Cat, scaling=False, seed_for_sample=None, seed_for_KernelTrick=None, seed_for_CV=None,  Kernel_nComp=100) :
        """
    Generates KnockOff copy of DataMatrix by 'sequential knockoff' method.

    Parameters
    ----------
    X : DataFrame or 2D-array ; size=(n,p)
        The DataMatrix whose KnockOff copy is required to be generated.

    is_Cat : list or array of True/False values ; length=p
        Each element determines whether the corresponding column of X is of Categorical(True) or Numerical(False) type.

    scaling : bool ; default False
        Whether the numerical columns of X will be standardized before further calculation.

    seed_for_sample, seed_for_KernelTrick, seed_for_CV : int ; default None
        Seeds of various pseudo-random number generation steps, to be specified for reproducable Output.
        
    Kernel_nComp : int ; default 100
        Dimensionality of the feature space(approximate RBF kernel feature map) used in regression.


    Returns
    -------
    tuple in the form (X,X_knockoff)
        1st element is the DataMatrix (after scaling, if any) ;
        2nd element is the corresponding KnockOff copy

    """

    X = pd.DataFrame(X).copy()
    n,p = X.shape
    names = X.columns
    idx = X.index
    X.rename(columns={name:str(name) for name in names},inplace=True)
    names = X.columns # making sure the col names are string , not int

   ## standardizing continuous columns ------------------------
    if scaling : Scale_Numeric(X,is_Cat)

   ## initialize KnockOff copy --------------------------------
    X_knockoff = pd.DataFrame(index=idx)

   ## kernel trick --------------------------------------------
    rbf_sampler = RBFSampler(gamma='scale',random_state=seed_for_KernelTrick,n_components=Kernel_nComp)

   ## sequencing over columns ---------------------------------
    np.random.seed(seed_for_sample)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for j in range(p) :
        name = names[j]
        Xj = X[name] # response , in the regression model of the conditional distribution   Xj|(X[-j],X_knockoff[1:j-1])
        Xcombined_j = pd.concat([X.drop(name,axis=1),X_knockoff],axis=1) # predictors
        current_isCat = Cat_or_Num(Xcombined_j)
        Xcombined_jKernel = rbf_sampler.fit_transform(Xcombined_j.iloc[:,np.invert(current_isCat)]) # kernel trick on numerical columns
        Xcombined_jCat = Xcombined_j.iloc[:,current_isCat] # categorical columns

        Xcombined_j = pd.get_dummies(pd.concat([pd.DataFrame(Xcombined_jKernel,index=idx),Xcombined_jCat],axis=1),drop_first=True)
        Xcombined_j.rename(columns={nam:str(nam) for nam in Xcombined_j.columns},inplace=True)


        if is_Cat[j] :
            #> fit ........................................
             Model = LogisticRegression()
             CV_type = RepeatedStratifiedKFold(n_repeats=5,n_splits=3,random_state=seed_for_CV) if min(Counter(Xj).values())>=3 else RepeatedKFold(n_repeats=5,n_splits=3,random_state=seed_for_CV) # stratified K-fold is preferred unless not enough observations available per class
             Model = (GridSearchCV( Model, param_grid={'C':[0.1,0.4,1,2.5,10]}, scoring='accuracy',cv=CV_type)).fit(Xcombined_j,Xj)
             categories = Model.classes_
             probabilities = pd.DataFrame(Model.predict_proba(Xcombined_j),index=idx)
            #> new sample .................................
             Xj_copy = probabilities.apply(lambda x : list(multinomial(1,x)).index(1),axis=1)
             Xj_copy = categories[Xj_copy]

        else :
            #> fit ........................................
             Model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],random_state=seed_for_CV)
             Model.fit(Xcombined_j,Xj)
             Xj_copy = Model.predict(Xcombined_j)
             s = np.std(Xj-Xj_copy)
            #> new sample ..................................
             Xj_copy = normal(Xj_copy,s)

        X_knockoff[name+'_kn.off'] = Xj_copy
        
    warnings.filterwarnings("default", category=ConvergenceWarning)

   ## KnockOff copy --------------------------------------------
    return tuple([X,X_knockoff])



# modified Sequential Knockoff ++++++++++++++++++++++++++++++++++++++++++++++++
'''
 ** It appears that as we move from first feature to last feature , we are modelling the conditional distribution Xj|(X[-j],X_knockoff[1:j-1]) based on larger data available. So there can be systematic bias in quality.
 ** To address this problem , split the data in a few blocks , shuffle the order of columns in each block , generate Sequential Knockoff as usual , then reshuffle them back to original order. Finally stack them together as the beginning

'''



def sKnockOff_Modified(X, is_Cat, scaling=False, seed_for_randomizing=None, seed_for_sample=None, seed_for_KernelTrick=None, seed_for_CV=None,  Kernel_nComp=100) :
    """
    This function splits the data in a 3 blocks , shuffle the order of columns in each block , generate Sequential KnockOff as usual in each block, then reshuffle them back to original order.
    
    WARNING: takes too much time than ogiginal sKnockOff method.

    """
    X = pd.DataFrame(X).copy()
    n,p = X.shape
    names = X.columns
    idx = X.index
    X.rename(columns={name:str(name) for name in names},inplace=True)
    names = X.columns # making sure the col names are string , not int
    names_knockoff = np.vectorize(lambda name: (name+'_kn.off'))(names)

   ## standardizing continuous columns --------------------------
    if scaling : Scale_Numeric(X,is_Cat)

   ## splitting 3 blocks --------------------------------------
    Block1,Block2 = train_test_split(X,test_size=0.66,random_state=seed_for_randomizing)
    Block2,Block3 = train_test_split(Block2,test_size=0.5,random_state=seed_for_randomizing)

   ## random shuffle ------------------------------------------
    np.random.seed(seed_for_randomizing)
    def Shuffle(Z):
        S = choice(range(p),size=p,replace=False)
        shuffled_Data = Z.iloc[:,S]
        is_Cat_similarly = list(pd.Series(is_Cat)[S])
        return (shuffled_Data, is_Cat_similarly)
    def ShuffleBack(Z,Z_knockoff):
        actualZ = Z[names]
        actualZ_knockoff = Z_knockoff[names_knockoff]
        return (actualZ,actualZ_knockoff)

   ## blockwise knockoff generation ---------------------------
    # Block1:-
    Block1,is_Cat1 = Shuffle(Block1)
    Block1,Block1_knockoff = sKnockOff(Block1, is_Cat1, False, seed_for_sample, seed_for_KernelTrick, seed_for_CV, Kernel_nComp)
    Block1,Block1_knockoff = ShuffleBack(Block1, Block1_knockoff)
    # Block2:-
    Block2,is_Cat2 = Shuffle(Block2)
    Block2,Block2_knockoff = sKnockOff(Block2, is_Cat2, False, seed_for_sample, seed_for_KernelTrick, seed_for_CV, Kernel_nComp)
    Block2,Block2_knockoff = ShuffleBack(Block2, Block2_knockoff)
    # Block3:-
    Block3,is_Cat3 = Shuffle(Block3)
    Block3,Block3_knockoff = sKnockOff(Block3, is_Cat3, False, seed_for_sample, seed_for_KernelTrick, seed_for_CV, Kernel_nComp)
    Block3,Block3_knockoff = ShuffleBack(Block3, Block3_knockoff)

   ## combining blocks -----------------------------------------
    X = pd.DataFrame(pd.concat([Block1,Block2,Block3],axis=0),index=idx)
    X_knockoff = pd.DataFrame(pd.concat([Block1_knockoff,Block2_knockoff,Block3_knockoff],axis=0),index=idx)
        # we want to recover both row order and column order

   ## KnockOff copy --------------------------------------------
    return tuple([X,X_knockoff])




# *****************************************************************************
##
###
####
###
##
# Visualization ===============================================================
'''
 ** Gives some visualization of 'how well a generated knockoff copy is'
 ** For very large or very small data, maybe problematic

'''


def Visual(X,X_knockoff,is_Cat,appendTitle='',Means=False,KDE=False,scale_the_corrplot=0.25):
    n,p = X.shape
    names = X.columns
    Xcombined = pd.concat([X,X_knockoff],axis=1)
    isCat_Combined = list(is_Cat)+list(is_Cat)
    k = p-sum(is_Cat)

   ## for Numerical variables ------------------------------
    XcombinedN = Xcombined.iloc[:,np.invert(isCat_Combined)].copy()
    #> means .......................................
    if Means:
        means = XcombinedN.apply(lambda x: np.mean(x)).to_numpy()
        pd.DataFrame({'original': means[:k],
                      'knockoff': means[k:]},index=names[np.invert(is_Cat)]).plot(kind='barh',title='Means'+appendTitle,rot=45,figsize=(8,0.2*k))
        plt.show()
    # KDE ..........................................
    if KDE:
        fig, axes = plt.subplots(2,1,figsize=(7,12))
        plt.subplots_adjust(hspace=0.2)
        XcombinedN.iloc[:,:k].plot(kind='density',title='Original'+appendTitle,ax=axes[0],grid=True)
        XcombinedN.iloc[:,k:].plot(kind='density',title='Knockoff'+appendTitle,ax=axes[1],grid=True)
        plt.show()
    #> variance ....................................
    XcombinedN.insert(loc=int(k),column='-',value=np.zeros((n,)))
    plt.figure(figsize=(scale_the_corrplot*k,scale_the_corrplot*k))
    sns.heatmap(XcombinedN.corr(),cmap="YlGnBu", annot=False,square=True)
    plt.title('combined Correlation Heatmap'+appendTitle)
    plt.show()

   ## for Categorical Variables ----------------------------
    XcombinedC = Xcombined.iloc[:,isCat_Combined]
    k = p-k
    if k>2:
        r,c = int(np.ceil(k/2)),2
        fig, axes = plt.subplots(r,c,figsize=(3*c,3*r))
        fig.suptitle('Counts'+appendTitle)
        plt.subplots_adjust(hspace=0.5)
        for i in range(k):
            r, c = i//2 , i%2
            pd.DataFrame({'original': (XcombinedC.iloc[:,i].groupby(XcombinedC.iloc[:,i])).count(),
                          'knockoff': (XcombinedC.iloc[:,i+k].groupby(XcombinedC.iloc[:,i+k])).count()}).plot(kind='bar',ax=axes[r,c],title=str(XcombinedC.iloc[:,i].name),rot=30,legend=np.invert(bool(i)))
        plt.show()
    elif k==2:
        fig, axes = plt.subplots(1,2,figsize=(3*2,3))
        fig.suptitle('Counts'+appendTitle)
        plt.subplots_adjust(hspace=0.5)
        for i in range(k):
            pd.DataFrame({'original': (XcombinedC.iloc[:,i].groupby(XcombinedC.iloc[:,i])).count(),
                          'knockoff': (XcombinedC.iloc[:,i+k].groupby(XcombinedC.iloc[:,i+k])).count()}).plot(kind='bar',ax=axes[i],title=str(XcombinedC.iloc[:,i].name),rot=30,legend=np.invert(bool(i)))
        plt.show()
    elif k==1:
        fig, axes = plt.subplots(1,1,figsize=(3,3.5))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle('Counts'+appendTitle)
        pd.DataFrame({'original': (XcombinedC.iloc[:,0].groupby(XcombinedC.iloc[:,0])).count(),
                      'knockoff': (XcombinedC.iloc[:,1].groupby(XcombinedC.iloc[:,1])).count()}).plot(kind='bar',ax=axes,title=str(XcombinedC.iloc[:,0].name),rot=30,legend=True)
        plt.show()




# *****************************************************************************
