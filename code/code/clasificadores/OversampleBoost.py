import matplotlib.pyplot as plt
import numpy as np
import statistics as stats
import math
import random
import array

from time import time

#metrics
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import classification_report

#model_selection
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

#resampling
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

#clasificadores
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier

#DEAP library for evolutionary algorithms
from deap import base
from deap import creator
from deap import tools

#datasets
from collections import Counter

import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble.forest import BaseForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.tree.tree import BaseDecisionTree
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y

class OversampleBoost(AdaBoostClassifier):
    """Implementation of a boosting method integrating an oversampling technique.
    OversampleBoost introduces data sampling into the AdaBoost algorithm by
    using a resampling method (SMOTE, SMOTETomek or SMOTEENN) on each boosting iteration [1].
    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.
    Parameters
    ----------
    oversampling_algorithm: string
        Name of the oversampling technique to implement on each boosting iteration
        (SMOTE, SMOTETomek or SMOTEENN)
    k_neighbors : int, optional (default=5)
        Number of nearest neighbors for the oversampling algorithm
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer.
           "SMOTEBoost: Improving Prediction of the Minority Class in
           Boosting." European Conference on Principles of Data Mining and
           Knowledge Discovery (PKDD), 2003.
    """

    def __init__(self,
                 oversampling_algorithm,
                 k_neighbors=5,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.smote_kneighbors = k_neighbors
        self.algorithm = algorithm
        
        if oversampling_algorithm=='SMOTE':
            self.smote = SMOTE(k_neighbors = self.smote_kneighbors)
        elif oversampling_algorithm=='SMOTE-TOMEK':
            self.smote = SMOTETomek()
        elif oversampling_algorithm=='SMOTE-ENN':
            self.smote = SMOTEENN()
        
        super(OversampleBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing the instantiated oversampling method during each boosting step.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        minority_target : int
            Minority class label.
        Returns
        -------
        self : object
            Returns self.
        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            X_min = X[np.where(y == self.minority_target)]

            # oversampling step.
            if len(X_min) >= self.smote_kneighbors:
                #oversample
                X_res, y_res = self.smote.fit_resample(X, y)
                #number of generated synthetic samples
                n_syn = X_res.shape[0]-X.shape[0]
                #weights for the synthetics
                sample_weight_syn = np.empty(n_syn, dtype=np.float64)
                sample_weight_syn[:] = 1. / X.shape[0]

                # Combine the weights.
                sample_weight = \
                    np.append(sample_weight, sample_weight_syn).reshape(-1, 1)
                sample_weight = \
                    np.squeeze(normalize(sample_weight, axis=0, norm='l1'))
                
            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X_res, y_res,
                sample_weight,
                random_state)

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break
            sample_weight = sample_weight[:X.shape[0]]
            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self
