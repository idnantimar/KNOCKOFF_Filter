"""
Created on Sun Dec 24 14:01:32 2023

Topic: My Template of Feature Importance
@author: R.Nandi
"""



#### My_Template ==============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from scipy.stats import rankdata



class My_Template_FeatureImportance(SelectorMixin,BaseEstimator):
    """
        A common template for all the feature-importance techniques under this projectwork.

        Attributes
        ----------
        threshold : float , default 0
            A cut-off, any feature with importance exceeding this value will be selected,
            otherwise will be rejected.

        max_features : int , default None
            The maximum possible number of selection. None implies no constrain,
            otherwise the 'threshold' will be updated automatically if it attempts to
            select more than 'max_features'.
    """
    threshold = 0
    max_features = None
    def __init__(self,random_state=None):
        self.random_state = random_state

    def fit(self,X,y):
        """
        This is a basic fit method , computing two atributes 'n_features_in_' and 'feature_names_in_'.

        (Override this with the actual implementation of required feature-importance technique ,
        that computes the attribute 'feature_importances_')

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values .

        Returns
        -------
        X_ : A copy of X

        y_ : A copy of y

        """
        X_,y_ = X.copy(),y.copy()
                ## to prevent any accidental change in original data
        if hasattr(self,'feature_importances_') :
            delattr(self,'feature_importances_')
                ## delete feature_importances_ from previous fit
        self.n_features_in_ = getattr(X_,'shape')[1]
        self.feature_names_in_ = getattr(X_,'columns',None)
        return X_,y_

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected

        Returns
        -------
        support : boolean array of shape (n_features,)
            An element is True iff its corresponding feature is selected for
            retention.
        """
        check_is_fitted(self,attributes="feature_importances_",
                        msg="The %(name)s instance must have a 'feature_importances_' attribute.")

        ranking_ = rankdata(-self.feature_importances_,method='ordinal')
                ## the most important feature has rank=1
                 ## the least important feature has rank=n_features_
        if (self.max_features is None) :
            self.threshold_ = self.threshold
        else:
            cut_off = np.where(ranking_==self.max_features)
            self.threshold_ = max(self.feature_importances_[cut_off],
                                  self.threshold)
                ## constrain the maximum possible number of selection
                 ## by a given constant
        self.support_ = (self.feature_importances_ >= self.threshold_)
        self.ranking_ = ranking_
        self.n_features_ = self.support_.sum(dtype=int)
        return self.support_

    def plot(self,sort=False,kind='bar',ax=None,
             xlabel='features',ylabel='importance',title=None,**kwargs):
        """
        Make plot of 'feature_importances_'.

        Parameters
        ----------
        sort : bool, default False
            Whether to sort the features according to 'feature_importances_' before plotting.

        kind, ax, xlabel, ylabel, title, **kwargs : keyword arguments to pass to matplotlib plotting method.

        """
        check_is_fitted(self,attributes="feature_importances_",all_or_any=all,
                        msg="The %(name)s instance is missing a 'feature_importances_' or 'thresholds_' attribute.")
        if self.feature_names_in_ is None :
            ix = (np.vectorize(lambda j:"X_"+str(j)))(range(self.n_features_in_))
        else : ix = self.feature_names_in_
        if title is None :
            title = self.__class__.__name__
        imp = pd.Series(self.feature_importances_,index=ix)
        if sort : imp.sort_values(ascending=False,inplace=True)
        imp.plot(kind=kind,ax=ax,xlabel=xlabel,ylabel=ylabel,title=title,**kwargs)
        plt.axhline(self.thresholds_,color='red',linestyle='dashed')
        plt.show()



# *****************************************************************************

