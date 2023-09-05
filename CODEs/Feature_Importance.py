"""
Created on Sun Sep  3 23:50:41 2023

Topic: Computing relative importance of the features and threshold for selection
@author: R.Nandi
"""

from Basics import *


#### Feature Importance Statistics ============================================
'''
 ** Computes a statistic with the importance of feature Xj & its KnockOff counterpart on regressing y based on combined data .
Constructed in a manner that it can take both +ve & -ve value , and large +ve value indicates that the feature is important.
    example: regress y on [X,X_knockoff] , let Bj & Bj_ko be estimated coeff. corresponding to Xj & Xj_knockoff respectively.
                    Wj = |Bj|-|Bj_ko| ;j=1,...,p
            is an example of such statistic

 NOTE: To construct KnockOff copy we do not need response y. But to construct Importance Statistic we need y . Since, without looking at y, X & X_knockoff are indistinguishable by definition.

'''


def THRESHOLD(W,FDR=0.1) :
    ## based on available feature importance W , determine the threshold for selection/rejection of feature
    T = (np.abs(W)).sort_values()
    def isFDPlessFDR(t):
        return (((1+np.sum(W<=-t))/max(1,np.sum(W>=t))) <=FDR)
    Threshold = np.inf
    for t in T :
        if not t : continue
        else :
            if not isFDPlessFDR(t): continue
            else :
                Threshold = t
                break
    return Threshold


def appendTHRESHOLDs(W,FDR):
    """
        Based on available feature importance , determine the threshold for selection/rejection of features.

    Parameters
    ----------
    W : Series ; length=number_of_features
        Each element corresponds to importance of a feature.
    FDR : float between [0,1] ; can be array or list of such numbers
        The False Discovery Rate upperbound to be specified.

    Returns
    -------
    Series ; length=number_of_features+number_of_FDR_input
        Threshold value corresponding to each FDR input is appended together with W.

    """

    vTHRESHOLD = np.vectorize(lambda f : THRESHOLD(W,f))
    vId = np.vectorize(lambda f : 'THRESHOLD_'+str(f*100))
    Threshold = vTHRESHOLD(FDR)
    vId = vId(FDR).reshape((-1,))

    return pd.concat([W,pd.Series(Threshold,index=vId)])




##### =========================================================================

def absDiff(Z1,Z2):
    """
    array ; |Z1|-|Z2|

    """
    return np.abs(Z1)-np.abs(Z2)



def signedMax(Z1,Z2):
    """
    array ; max(Z1,Z2).sign(Z1-Z2)

    """
    return np.vstack((Z1,Z2)).max(axis=0)*np.sign(Z1-Z2)




#### ==========================================================================

def basicImp_ContinuousResponse(X,X_knockoff,y,FDR=0.1,Scoring=signedMax,seedCV=None):
    """
    Regression Coefficient based importance for continuous response case
    """
    X = pd.get_dummies(X,drop_first=True)
    n,p = X.shape
    names = X.columns
    X_knockoff = pd.get_dummies(X_knockoff, drop_first=True)
    y = pd.DataFrame(y)
    y = StandardScaler().fit_transform(y)
        # get_dummies() put all the numeric variables first and dummy variables at the end
         # So to maintain same ordering of original and knockoff variables , we have to first dummify them individually then concat, instead of first concat then dummify the combined data
    y = np.ravel(y)

   ## feature importance ------------------------------------
    Model = (ElasticNetCV(l1_ratio=[0.1,0.5,0.95],random_state=seedCV))
    Model.fit(pd.concat([X,X_knockoff],axis=1),y)
        # not valid for categorical y, use different statistic accordingly
        # combined data is n*2p , so need n>=2p for identifiability
    W = Model.coef_

    Z = np.abs(W[:p])
    Z_knockoff = np.abs(W[p:])
    W = pd.Series(Scoring(Z,Z_knockoff),index=names)

   ## selecting Threshold -----------------------------------
    return appendTHRESHOLDs(W,FDR)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def basicImp_BinaryResponse(X,X_knockoff,y,FDR=0.1,Scoring=signedMax):
    """
    Regression Coefficient based importance for binary response case
    """

    X = pd.get_dummies(X,drop_first=True)
    n,p = X.shape
    names = X.columns
    X_knockoff = pd.get_dummies(X_knockoff, drop_first=True)
   ## feature importance ------------------------------------
    Model = LogisticRegression()
    Model.fit(pd.concat([X,X_knockoff],axis=1),y)
        # Just linear regression is replaced by Logistic regression
    W = Model.coef_[0]

    Z = np.abs(W[:p])
    Z_knockoff = np.abs(W[p:])
    W = pd.Series(Scoring(Z,Z_knockoff),index=names)

   ## selecting Threshold -----------------------------------
    return appendTHRESHOLDs(W,FDR)




#### ==========================================================================

from sklearn.metrics import log_loss


def LOFO_ImpCategorical(X,X_knockoff,y,FDR=0.1,seed=None,take_diff=True,Scoring=signedMax):
    """
    For each column in combined data,
    fit the model y~[X,X_knockoff] once in the presence of that column & once in absence.

    Higher the increase in error in absence of a column , more the importance of that feature.

    Parameters
    ----------
    X : DataFrame
        The DataMatrix.

    X_knockoff : DataFrame
        KnockOff copy of X.

    y : Series ; index=index_of_data
        Must be a categorical variable with any number(>1) of categories.

    FDR : float between [0,1] or list of such float values ; default 0.1
        The False Discovery Rate upperbound to be specified.

    seed : int ; default None
        Seed for reproducable output.

    take_diff : bool ; default True
        Whether to shift(True) the origin of Z & Z_knockoff axis from (0,0) to full_Model error. Any feature that produces LOFO_error<fullModel_error will be rejected then irrespective of its importance score.

    Scoring : a function that determines how important an original feature is , compared to its knockoff copy ; default signedMax

    Returns
    -------
    Series ; length=number_of_features+number_of_FDR_input
        Importance of each feature.
        And Threshold for selection/rejection appended with it.

    """
    X = pd.get_dummies(X,drop_first=True)
    n,p = X.shape
    names = X.columns
    X_knockoff = pd.get_dummies(X_knockoff, drop_first=True)

   ## combining Xj & Xj_knockoff -----------------------------
    Xcombined = (pd.concat([X,X_knockoff],axis=1))
    namesCombined = Xcombined.columns

   ## LOFO feature importance --------------------------------
    k = y.nunique()
    for i in range(20) :
        Xcombined_train, Xcombined_test, y_train, y_test = train_test_split(Xcombined, y , test_size=0.4,random_state=seed if (seed is None) else (seed+i))
        if (y_train.nunique()==k) and (y_test.nunique()==k) : break

    lofoModel = LogisticRegression()
    lofoScore = []
        # when k(>2) categories are there in response, there will be k-1 coefficients corresponding to each feature if we use Multinomial Logistic Model. So we can not use coefficient based methods here for scoring
    for name in namesCombined :
        Xcombined_trainNew = Xcombined_train.drop(columns=name)
        Xcombined_testNew = Xcombined_test.drop(columns=name)
        lofoModel.fit(Xcombined_trainNew,y_train)
        prob_pred = lofoModel.predict_proba(Xcombined_testNew)
        lofoScore += [log_loss(y_test,prob_pred)]
    W = np.array(lofoScore) # if high error is there in the absence on a feature , that means it is important
    if take_diff :
        fullModel = LogisticRegression()
        fullModel.fit(Xcombined_train,y_train)
        fullScore = log_loss(y_test,fullModel.predict_proba(Xcombined_test))
        W -= fullScore # entry can be negative when there is over fitting

    W = W.reshape((2,-1))
    Z = W[0]
    Z_knockoff = W[1]
    not_3rd = 1-np.array((Z<0)*(Z_knockoff<0),dtype=int) if take_diff else 1 # indicator that (Zj,Zj_knockoff) is not at 3rd Quadrant wrt current origin

    W = pd.Series(Scoring(Z,Z_knockoff)*not_3rd,index=names)

   ## selecting Threshold -----------------------------------
    return appendTHRESHOLDs(W,FDR)
