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
    def __init__(self,random_state=None,
                 max_features=None,threshold=1e-10,reduce_norm=1,
                 Cs=list(np.logspace(-4,+4,10))):
        super().__init__(random_state)
        self.Cs = Cs
        self.estimator = clone(self.Estimator_Type)
        self.estimator.set_params(**{'random_state':self.random_state,'Cs':self.Cs})
        self.threshold = threshold
        self.reduce_norm = reduce_norm
        self.max_features = max_features

    def fit(self,X,y,penalty_weights=None,n_jobs=None,max_iter=100,cv=None):
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        super().fit(X,y)
        self.estimator.set_params(**{'n_jobs':n_jobs,'max_iter':max_iter,'cv':cv})
        if penalty_weights is not None:
            X /= np.ravel(penalty_weights)
            ## weighted lasso with features X_j and penalties w_j|b_j| is same as
             # ordinary lasso with rescaled features X_j/wj
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        self.estimator.fit(X,y)
        warnings.filterwarnings("default", category=ConvergenceWarning)
        self.coef_ = self.estimator.coef_
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




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class vanillaLASSO_importance(WeightedLASSO_importance):
    Penalty_Weights = None
    def fit(self,X,y,
            n_jobs=None,max_iter=100,cv=None):
        return super().fit(X,y,n_jobs=n_jobs,max_iter=max_iter,cv=cv,
                           penalty_weights=self.Penalty_Weights)




# *****************************************************************************
##
###
####
###
##
#
#### Different Custom Penalties ===============================================

from tensorflow import keras


class penalizedLOGISTIC_importance_tf(My_Template_FeatureImportance):
    def __init__(self,random_state=None,
                 max_features=None,threshold=1e-10,reduce_norm=1,
                 Cs=list(np.logspace(-4,+4,10))):
        super().__init__(random_state)
        self.Cs = Cs
        self.threshold = threshold
        self.max_features = max_features
        self.reduce_norm = reduce_norm

    def Estimator_Type(self,n_features_in,n_classes,
                       optimizer='adam',metrics=['accuracy'],**kwargs):
        model = keras.Sequential(name='Logistic_Regression')
        model.add(keras.Input(shape=(n_features_in),name="Input"))
        model.add(keras.layers.Dense(units=n_classes,activation='softmax',name='SoftMax',
                                     use_bias=True,
                                     kernel_initializer=keras.initializers.GlorotNormal(seed=self.random_state),
                                     bias_initializer='zeros',
                                     kernel_regularizer=self.penalty()))
            ## constructing a single-layer NN with 'softmax' activation and
            ## minimizing 'categorical_crossentrophy' is equivalent in concept,
             ## to fitting Logistic Regression by solving MLE
        compile_configuration = {'optimizer':optimizer,
                                 'loss':'categorical_crossentropy',
                                 'metrics':metrics}
        compile_configuration.update(**kwargs)
        return (model,compile_configuration)

    def penalty(self,C=1e+2):
        return keras.regularizers.L1(l1=1/C)

    def fit(self,X,y,epochs=100,verbose=1,validation_split=0.0,callbacks=None,
            **kwargs):
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        super().fit(X,y)
        y = pd.get_dummies(y,drop_first=False,dtype=int)
        self.estimator,compile_configuration = self.Estimator_Type(self.n_features_in_,self.n_classes_,**kwargs)
        self.estimator.compile(**compile_configuration)
        self.estimator.fit(X,y,epochs=epochs,callbacks=callbacks,shuffle = (self.random_state is None),
                           verbose=verbose,validation_split=validation_split)
        self.coef_ = (self.estimator.weights[0]).numpy().T
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
