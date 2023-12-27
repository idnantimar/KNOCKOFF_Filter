"""
Created on Sat Dec 23 10:13:35 2023

Topic: Penalized Regression based feature selection (for categorical response)
@author: R.Nandi
"""

from ..Basics import *
from ..My_Template import My_Template_FeatureImportance


#### Weighted LASSO ============================================================
from sklearn.linear_model import LogisticRegressionCV
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone


class WeightedLASSO_importance(My_Template_FeatureImportance):
    Estimator_Type = LogisticRegressionCV(penalty='l1',multi_class='multinomial',
                                          solver='saga')
    def __init__(self,random_state=None,*,
                 max_features=None,threshold=1e-10):
        """


        Parameters
        ----------
        random_state : TYPE, optional
            DESCRIPTION. The default is None.
        * : TYPE
            DESCRIPTION.
        max_features : TYPE, optional
            DESCRIPTION. The default is None.
        threshold : TYPE, optional
            DESCRIPTION. The default is 1e-10.

        Returns
        -------
        None.

        """
        super().__init__(random_state)
        self.threshold = threshold
        self.max_features = max_features

    def fit(self,X,y,penalty_weights=None,*,n_jobs=None,max_iter=100,cv=None,scoring=None,
            Cs=10,reduce_norm=1):
        """


        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        penalty_weights : TYPE, optional
            DESCRIPTION. The default is None.
        * : TYPE
            DESCRIPTION.
        n_jobs : TYPE, optional
            DESCRIPTION. The default is None.
        max_iter : TYPE, optional
            DESCRIPTION. The default is 100.
        cv : TYPE, optional
            DESCRIPTION. The default is None.
        scoring : TYPE, optional
            DESCRIPTION. The default is None.
        Cs : TYPE, optional
            DESCRIPTION. The default is 10.
        reduce_norm : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.estimator = clone(self.Estimator_Type)
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        super().fit(X,y)
        self.estimator.set_params(**{'n_jobs':n_jobs,'max_iter':max_iter,'cv':cv,
                                     'scoring':scoring,'Cs':Cs,'random_state':self.random_state})
        if penalty_weights is not None:
            X /= np.ravel(penalty_weights)
            ## weighted lasso with features X_j and penalties w_j|b_j| is same as
             # ordinary lasso with rescaled features X_j/wj
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        self.estimator.fit(X,y)
        warnings.filterwarnings("default", category=ConvergenceWarning)
        self.coef_ = self.estimator.coef_
        self.C_ = self.estimator.C_
        self.Cs_ = self.estimator.Cs_
        self.reduce_norm = reduce_norm
        self.feature_importances_ = np.linalg.norm(self.coef_,
                                                   ord=self.reduce_norm,axis=0)
        return self

    def get_error_rates(self,true_coef):
        """


        Parameters
        ----------
        true_coef : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.get_support()
        self.true_coef = np.array(true_coef)
        if (self.true_coef.dtype==bool) :
            self.true_support = self.true_coef
        else :
            true_support = np.linalg.norm(self.true_coef.reshape(-1,self.n_features_in_),
                                          ord=self.norm_order,axis=0)
            self.true_support = (true_support > self.threshold_)
        super().get_error_rates()




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class vanillaLASSO_importance(WeightedLASSO_importance):
    Penalty_Weights = None
    Cs = 10
    reduce_norm = 1
    scoring = None
    cv=None
    def fit(self,X,y,*,
            n_jobs=None,max_iter=100):
        """


        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        * : TYPE
            DESCRIPTION.
        n_jobs : TYPE, optional
            DESCRIPTION. The default is None.
        max_iter : TYPE, optional
            DESCRIPTION. The default is 100.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return super().fit(X,y,n_jobs=n_jobs,max_iter=max_iter,
                           cv=self.cv,scoring=self.scoring,Cs=self.Cs,
                           penalty_weights=self.Penalty_Weights,
                           reduce_norm=self.reduce_norm)




# *****************************************************************************
##
###
####
###
##
#
#### Different Custom Penalties ===============================================

from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from scikeras.wrappers import KerasClassifier



class penalizedLOGISTIC_importance_tf(My_Template_FeatureImportance):
    def __init__(self,random_state=None,*,
                 max_features=None,threshold=1e-10):
        super().__init__(random_state)
        self.threshold = threshold
        self.max_features = max_features


    def Estimator_Type(self,n_features_in,n_classes,**kwargs):
        _random_state = self.random_state
        _penalty = self.penalty
        ##> implementing Logistic Regression using tensorflow ...........
        class _LogisticReg(BaseEstimator):
            def __init__(self,C=1e+2):
                self.C = C
                    ## 1/penalty_strength in LogisticRegression , smaller value of C implies stronger penalty

            def fit(self,_X,_y):
                nn = keras.Sequential(name='Logistic_Regression')
                nn.add(keras.Input(shape=(n_features_in,),name="Input"))
                nn.add(keras.layers.Dense(units=n_classes,activation='softmax',name='SoftMax',
                                             use_bias=True,
                                             kernel_initializer=keras.initializers.GlorotNormal(seed=_random_state),
                                             bias_initializer='zeros',
                                             kernel_regularizer=_penalty(self.C)))
                    ## constructing a single-layer NN with 'softmax' activation and
                    ## minimizing 'categorical_crossentrophy' is equivalent in concept,
                     ## to fitting Logistic Regression by solving MLE
                compile_configuration = {'optimizer':'adam',
                                         'loss':'categorical_crossentropy',
                                         'metrics':['accuracy']}
                    ## default cofiguration with 'adam' optimizer and 'accuracy' metric
                compile_configuration.update(**kwargs)
                self.nn = KerasClassifier(model=nn,**compile_configuration)
                self.nn.fit(_X,_y)
            def score(self,_X,_y):
                return self.nn.score(_X,_y)

            def _get_coef(self):
                return (self.nn.model_.weights[0]).numpy().T
        #> ..............................................................
        return _LogisticReg


    def penalty(self,C=1e+2):
        return keras.regularizers.L1(l1=1/C)


    def fit(self,X,y,*,epochs=100,verbose=0,callbacks=None,
            validation_split=0.0,validation_freq=10,
            cv=None,n_jobs=None,Cs=np.logspace(-4,4,10),
            reduce_norm=1,
            **kwargs_for_Estimator_Type):
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        super().fit(X,y)
        y = pd.get_dummies(y,drop_first=False,dtype=int)

        configuration = {'fit__epochs':epochs,'fit__verbose':verbose,'fit__callbacks':callbacks,
                         'fit__validation_split':validation_split,'fit__validation_freq':validation_freq,
                         'fit__shuffle':(self.random_state is None),
                         'predict__verbose':verbose,'predict__callbacks':callbacks
                         }
        configuration.update(**kwargs_for_Estimator_Type)
        estimator = self.Estimator_Type(self.n_features_in_,self.n_classes_,
                                        **configuration)
        estimator = estimator()
                ## this is an _LogisticReg() instance, defined above

        crossvalidation_configuration = {'cv':cv,'n_jobs':n_jobs,'verbose':verbose,'refit':True}
        self.gridsearch = GridSearchCV(estimator,{'C':np.array(Cs)},**crossvalidation_configuration)
        self.gridsearch.fit(X,y)
        self.estimator = self.gridsearch.best_estimator_.nn

        self.coef_ = (self.estimator.model_.weights[0]).numpy().T
        self.reduce_norm = reduce_norm
        self.feature_importances_ = np.linalg.norm(self.coef_,
                                                   ord=self.reduce_norm,axis=0)
        return self


    def get_error_rates(self,true_coef):
        self.get_support()
        self.true_coef = np.array(true_coef)
        if (self.true_coef.dtype==bool) :
            self.true_support = self.true_coef
        else :
            true_support = np.linalg.norm(self.true_coef.reshape(-1,self.n_features_in_),
                                          ord=self.norm_order,axis=0)
            self.true_support = (true_support > self.threshold_)
        super().get_error_rates()



