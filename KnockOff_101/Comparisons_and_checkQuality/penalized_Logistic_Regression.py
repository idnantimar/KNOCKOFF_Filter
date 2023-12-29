"""
Created on Sat Dec 23 10:13:35 2023

Topic: Penalized Regression based feature selection (for categorical response)
@author: R.Nandi
"""

from ..Basics import *
from ..My_Template import My_Template_FeatureImportance


#### Vanilla LASSO ===========================================================
from sklearn.linear_model import LogisticRegressionCV
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone



class vanillaLASSO_importance(My_Template_FeatureImportance):
    """
        Feature selection using Logistic LASSO regression with default configuration of ``LogisticRegressionCV``.

        Class Variables
        ---------------
        Estimator_Type : ``LogisticRegressionCV(penalty='l1',multi_class='multinomial',solver='saga',scoring=None,cv=None,Cs=10)``

        Parameters
        ----------
        random_state : int ; default None
            Seed for reproducible results across multiple function calls.

        max_features : int ; default None
            The maximum possible number of selection. None implies no constrain,
            otherwise the `threshold` will be updated automatically if it attempts to
            select more than `max_features`.

        threshold : float ; default 1e-10
            A cut-off, any feature with importance exceeding this value will be selected,
            otherwise will be rejected.

    """

    Estimator_Type = LogisticRegressionCV(penalty='l1',multi_class='multinomial',
                                          solver='saga',
                                          scoring=None,cv=None,Cs=10)
    def __init__(self,random_state=None,*,
                 max_features=None,threshold=1e-10):
        super().__init__(random_state)
        self.threshold = threshold
        self.max_features = max_features


    def fit(self,X,y,*,n_jobs=None,max_iter=100,reduce_norm=1):
        """
        ``fit`` method for LASSO.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            `n_features_in_` will be number of columns after ``pd.get_dummies(X,drop_first=True)`` is applied.

        y : Series of shape (n_samples,)
            The target values.

        n_jobs : int ; default None
            Number of CPU cores used during the cross-validation loop in ``LogisticRegressionCV``.

        max_iter : int ; default 100
            Maximum number of iterations of the optimization algorithm.

        reduce_norm : non-zero int, inf, -inf ; default 1
            Order of the norm used to compute `feature_importances_` in the case where the `coef_` of the
            ``LogisticRegressionCV`` is of dimension 2. By default 'l1'-norm is being used.

        Returns
        -------
        self
            The fitted ``vanillaLASSO_importance`` instance is returned.

        """
        self.estimator = clone(self.Estimator_Type)
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        super().fit(X,y)
        self.estimator.set_params(**{'n_jobs':n_jobs,'max_iter':max_iter,'random_state':self.random_state})
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


    def transform(self,X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        2-D array of shape (n_samples, n_selected_features)
            The input samples with only the selected features.

        """
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        return super().transform(X)


    def get_error_rates(self,true_coef,*,plot=False):
        """
        Computes various error-rates when true importance of the features are known.

        Parameters
        ----------
        true_coef : array of shape (`n_features_in_`,) or (`n_classes_`,`n_features_in_`)
            If a 1-D boolean array , True implies the feature is important in true model, null feature otherwise.
            If a 1-D array of floats , it represent the `feature_importances_` of the true model,
            2-D array of floats represnt `coef_` of the true model.

        plot : bool ; default False
            Whether to plot the `confusion_matrix_for_features_`.

        Returns
        -------
        dict
            Returns the empirical estimate of various error-rates
           {'PCER': per-comparison error rate is given by `pfer_`/'n_features_in_',
            'FDR': false discovery rate is given by `pfer_`/`n_features_selected_`,
            'PFER': per-family error rate is given by total number of `false_discoveries_`,
            'TPR': true positive rate is given by (`n_features_in_`-`pfer_`)/(`n_features_in_`-`pfer_`+`n_false_negatives_`)
            }

        """
        self.get_support()
        self.true_coef = np.array(true_coef)
        if (self.true_coef.dtype==bool) :
            self.true_support = self.true_coef
        else :
            true_support = np.linalg.norm(self.true_coef.reshape(-1,self.n_features_in_),
                                          ord=self.reduce_norm,axis=0)
            self.true_support = (true_support > self.threshold_)
        return super().get_error_rates(plot=plot)





#### ==========================================================================
##
###
####
###
##
#
#### Different Custom Penalties ===============================================

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from scikeras.wrappers import KerasClassifier
from sklearn.linear_model import LogisticRegression



##> implementing Logistic Regression using tensorflow & sklearn ..........

class _LogisticReg(BaseEstimator):
    """
        We know that, constructing a single-layer neural network with 'softmax' activation and
        minimizing 'categorical_crossentrophy' is conceptually equivalent to fitting Logistic
        Regression with maximum likelihood estimation (MLE).

        [ For internal use mainly ]

        This class determines the architecture of that network.
        'adam' ``optimizer`` and 'accuracy' ``metrics`` are used by default.

    """
    def __init__(self,n_features_in,n_classes,penalty_class,penalty_param,random_state):
        self.n_features_in = n_features_in
            ## input dimension
        self.n_classes = n_classes
            ## output dimension
        self.penalty_class = penalty_class
            ## the class of required penalty function, not an instant
        self.penalty_param = penalty_param
            ## this can be anything, like- scalar,array,list etc.
             ## design your custom penalty accordingly , in terms of a single tuning parameter.
            ## its best possible value will be decided by GridSearch later
        self.random_state = random_state

    def fit(self,X,y,weights0,bias0,**kwargs_KerasClassifier):
        _penalty_fun = self.penalty_class(self.penalty_param)
            ## a callable penalty function,
             ## derived from the penalty_class with given penalty_param
        _kernel_initializer = _custom_initializer(weights0)
        _bias_initializer = _custom_initializer(bias0)
            ## pass the coef_.T and intercept_ of a LogisticRegression(penalty=None)
             ## to leverage the training process
        nn = keras.Sequential(name='Logistic_Regression')
        nn.add(keras.Input(shape=(self.n_features_in,),name="Input"))
        nn.add(keras.layers.Dense(units=self.n_classes,activation='softmax',name='SoftMax',
                                  use_bias=True,
                                  kernel_initializer=_kernel_initializer,
                                  bias_initializer=_bias_initializer,
                                  kernel_regularizer=_penalty_fun))
        compile_configuration = {'optimizer':'adam',
                                 'loss':'categorical_crossentropy',
                                 'metrics':['accuracy']}
        compile_configuration.update(**kwargs_KerasClassifier)
        self.nn = KerasClassifier(model=nn,**compile_configuration)
        self.nn.fit(X,y)

    def score(self,X,y):
            ## same as 'metrics' used in compilation ,
            ## this will be used as 'scoring' in GridSearch
        return self.nn.score(X,y)

    def _get_coef(self):
            ## returns the estimated coefficients of underlying Logistic Regression
             ## shape (n_classes,n_features_in)
        return (self.nn.model_.weights[0]).numpy().T

#> .......................................................................


##> ..................................................................
class custom_penalty(keras.regularizers.Regularizer):
    """
        A format for custom penalty to be applied. Default is 0-penalty.

        [ Override the ``__call__`` method accordingly, that will return the total penalty upon ``coef`` ]

        The ``__init__`` method must be intact in terms of a single parameter   ``penalty_param``,
        which can be any thing like- scalar, array, list etc.
        e.g. - if we need two tuning parameters `alpha`, `beta`, the ``__call__`` method should look like

            >>>     def __call__(self,coef):
            ...:        alpha,beta = self.penalty_param
            ...:        ## rest of the code, in terms of `alpha`, `beta`

        And then the class should be instantiated as -

            >>> instance = custom_penalty((`alpha0`,`beta0`))

    """
    def __init__(self,penalty_param):
        self.penalty_param = penalty_param
    def __call__(self,coef):
        return 0
    def get_config(self):
        return {'penalty_param':self.penalty_param}


class _custom_initializer(keras.initializers.Initializer):
    """
        For leveraging the training by providing initial guess about weights

        Initial guess must be in required shape -
            for kernel weights shape=(input_dim,output_dim)
            for bias shape=(n_units,)
    """
    def __init__(self,initial_guess):
        self.initial_guess = initial_guess
    def __call__(self,shape,dtype=None):
        return tf.constant(self.initial_guess,dtype=dtype)
    def get_config(self):
        return {'initial_guess':self.initial_guess}

#> ...................................................................




class penalizedLOGISTIC_importance(My_Template_FeatureImportance):
    """
        Feature selection using Logistic Regression with any custom penalty , implemented using ``tensorflow`` and ``sklearn``.

        Parameters
        ----------
        random_state : int ; default None
            Seed for reproducible results across multiple function calls.

        max_features : int ; default None
            The maximum possible number of selection. None implies no constrain,
            otherwise the `threshold` will be updated automatically if it attempts to
            select more than `max_features`.

        threshold : float ; default 1e-10
            A cut-off, any feature with importance exceeding this value will be selected,
            otherwise will be rejected.

    """

    def __init__(self,random_state=None,*,
                 max_features=None,threshold=1e-10):
        super().__init__(random_state)
        self.threshold = threshold
        self.max_features = max_features


    def penalty(self):
        """
        The custom penalty function class implementation. Default is no penalty.

        [ Override it accordingly ]

        The ``__init__`` method must have a single parameter tuning the shape of the penalty function.
        There must be a ``__call__`` method.

        Returns
        -------
        The custom penalty function class (not an instance or a callable)
        """
        return custom_penalty


    def fit(self,X,y,penalty_params,*,epochs=100,verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=10,restore_best_weights=True,
                                     min_delta=1e-6,start_from_epoch=0)],
            validation_split=0,validation_data=None,validation_freq=1,
            cv=None,n_jobs=None,
            reduce_norm=1,
            initial_guess = LogisticRegression(penalty=None,multi_class='multinomial',solver='lbfgs'),
            **kwargs_for_Estimator_Type):
        """
        ``fit`` method for the Logistic Regression with custom penalty.

        Note : each time calling ``fit`` will compile the model based on the shape of the data, so previous record will be lost.
        To resume a previous training run proceed as follows

            >>> Model.fit(X,y,...) # fitting for the first time with required parameters
            >>> Model.estimator.fit(X,pd.get_dummies(y)) # resuming previous run

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            `n_features_in_` will be number of columns after ``pd.get_dummies(X,drop_first=True)`` is applied.

        y : Series of shape (n_samples,)
            The target values.

        penalty_params : list
            Possible values of ``penalty_param`` tuning the shape of custom penalty function.
            The best one will be chosen by ``GridSearchCV``. For a list of length 1, it will be used as it is.

        epochs : int ; default 100
            The number of iterations the model will be trained.

        verbose : int ; default 0
            The verbosity level of training.

        callbacks : list of keras.callbacks.Callback instances ; default ``[EarlyStopping(monitor='loss', patience=10,restore_best_weights=True,min_delta=1e-6,start_from_epoch=0)]``
            List of callbacks to apply during training.

        validation_split : float in [0,1) ; default 0
            Fraction of the training data to be used as validation data.

        validation_data : tuple like (X_val,y_val) ; default None
            Data on which to evaluate the loss and any model metrics at the end of each epoch.

        validation_freq : int ; default 1
            Specifies how many training epochs to run before a new validation run is performed.

        cv : int, cross-validation generator or an iterable ; default None
            Determines the cross-validation splitting strategy in ``GridSearchCV``

        n_jobs : int ; default None
            Number of CPU cores used during the cross-validation loop in ``GridSearchCV``.

        reduce_norm : non-zero int, inf, -inf ; default 1
            Order of the norm used to compute `feature_importances_` in the case where the `coef_` of the
            underlying Logistic Regression is of dimension 2. By default 'l1'-norm is being used.

        initial_guess : default ``LogisticRegression(penalty=None,multi_class='multinomial',solver='lbfgs')``
            Any classifier with a ``coef_`` and an ``intercept_`` attribute,
            those will be used as ``kernel_initializer`` and ``bias_initializer``
            of the underlying neural network to leverage the training.

        **kwargs_for_Estimator_Type : keyword arguments to be passed to ``scikeras.wrappers.KerasClassifier``
            (The input passed to ``metrics`` will be used to evaluate ``scoring`` in ``GridSearchCV``, default ``scoring`` is 'accuracy').

        Returns
        -------
        self
            The fitted ``penalizedLOGISTIC_importance`` instance is returned.

        """
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        super().fit(X,y)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        initial_guess.fit(X, y)
        warnings.filterwarnings("default", category=ConvergenceWarning)
        y = pd.get_dummies(y,drop_first=False,dtype=int)
        if validation_data is not None :
            X_val,y_val = validation_data
            X_val = pd.get_dummies(X_val,drop_first=True,dtype=int)
            y_val = pd.get_dummies(y_val,drop_first=False,dtype=int)
            validation_data = (X_val,y_val)

        configuration = {'fit__epochs':epochs,'fit__verbose':verbose,'fit__callbacks':callbacks,
                         'fit__validation_split':validation_split,'fit__validation_data':validation_data,
                         'fit__validation_freq':validation_freq,
                         'fit__shuffle':(self.random_state is None),
                         'predict__verbose':0,'predict__callbacks':None
                         }
        configuration.update(**kwargs_for_Estimator_Type)

        estimator = _LogisticReg(n_features_in=self.n_features_in_,
                                 n_classes=self.n_classes_,
                                 penalty_class=self.penalty(),
                                 penalty_param=penalty_params[0],
                                 random_state=self.random_state)
                ## this estimator is an _LogisticReg() instance, defined above
        configuration.update({'weights0':initial_guess.coef_.T,'bias0':initial_guess.intercept_})

        if len(penalty_params)>1 :
            crossvalidation_configuration = {'cv':cv,'n_jobs':n_jobs,'verbose':2*bool(verbose),'refit':True}
            self.gridsearch = GridSearchCV(estimator,{'penalty_param':penalty_params},**crossvalidation_configuration)
            self.gridsearch.fit(X,y,**configuration)
            self.estimator = self.gridsearch.best_estimator_.nn
                    ## this self.estimator is the best fitted KerasClassifier
            self.best_penalty_ = self.gridsearch.best_params_['penalty_param']
        else :
            estimator.fit(X,y,**configuration)
            self.estimator = estimator.nn
                    ## skip the GridSearch when only one possible penalty_param is given
            self.best_penalty_ = penalty_params[0]

        self.coef_ = (self.estimator.model_.weights[0]).numpy().T
        self.reduce_norm = reduce_norm
        self.feature_importances_ = np.linalg.norm(self.coef_,
                                                   ord=self.reduce_norm,axis=0)
        return self


    def transform(self,X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        2-D array of shape (n_samples, n_selected_features)
            The input samples with only the selected features.

        """
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        return super().transform(X)


    def get_error_rates(self,true_coef,*,plot=False):
        """
        Computes various error-rates when true importance of the features are known.

        Parameters
        ----------
        true_coef : array of shape (`n_features_in_`,) or (`n_classes_`,`n_features_in_`)
            If a 1-D boolean array , True implies the feature is important in true model, null feature otherwise.
            If a 1-D array of floats , it represent the `feature_importances_` of the true model,
            2-D array of floats represnt `coef_` of the true model.

        plot : bool ; default False
            Whether to plot the `confusion_matrix_for_features_`.

        Returns
        -------
        dict
            Returns the empirical estimate of various error-rates
           {'PCER': per-comparison error rate ,
            'FDR': false discovery rate ,
            'PFER': per-family error rate ,
            'TPR': true positive rate
            }

        """
        self.get_support()
        self.true_coef = np.array(true_coef)
        if (self.true_coef.dtype==bool) :
            self.true_support = self.true_coef
        else :
            true_support = np.linalg.norm(self.true_coef.reshape(-1,self.n_features_in_),
                                          ord=self.reduce_norm,axis=0)
            self.true_support = (true_support > self.threshold_)
        return super().get_error_rates(plot=plot)





### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class WeightedLASSO_importance



### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class MCP_importance(penalizedLOGISTIC_importance_tf):
    def penalty(self,C=1e+2):
        pass





### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class groupLASSO_importance(penalizedLOGISTIC_importance_tf):
    def penalty(self,C=1e+2):
        pass





#### ==========================================================================
