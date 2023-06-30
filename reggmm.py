import typing as ty
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.estimator_checks import check_estimator
from sklearn.cluster import kmeans_plusplus

import c_reggmm

CovType_t = ty.Literal['diag', 'spherical', 'tieddiag']


@dataclass(init=False, repr=False, eq=False)
class GMMParams:
    pi: np.ndarray
    mu: np.ndarray
    Sigma: np.ndarray

    def __init__(self, pi: np.ndarray, mu: np.ndarray, Sigma: np.ndarray):
        self.pi = pi.astype(np.float64)
        self.mu = mu.astype(np.float64)
        self.Sigma = Sigma.astype(np.float64)


def _init_pi_kmeanspp(
    assignment: np.ndarray,
    K: int,
) -> np.ndarray:
    m = assignment.shape[0]
    return np.array([np.sum(assignment == k) / m for k in range(K)])


def _init_Sigma_kmeanspp_diagcov(
    X: np.ndarray,
    assignment: np.ndarray,
    K: int,
) -> np.ndarray:
    return np.stack([np.std(X[assignment == k], axis=0) for k in range(K)],
                    axis=0)


def _init_Sigma_kmeanspp_sphericalcov(
    X: np.ndarray,
    assignment: np.ndarray,
    K: int,
) -> np.ndarray:
    return np.array([np.std(X[assignment == k]) for k in range(K)])


def _init_Sigma_kmeanspp_tieddiagcov(
    X: np.ndarray,
    assignment: np.ndarray,
    K: int,
) -> np.ndarray:
    return np.mean(
        np.stack([np.std(X[assignment == k], axis=0) for k in range(K)],
                 axis=0),
        axis=0)


def init_params_kmenaspp(
    X: np.ndarray,
    n_clusters: int,
    covariance_type: CovType_t,
) -> GMMParams:
    """
    Initialize diag-covariance GMM parameters from kmeans++.
    :param X: the data, of shape (m, p)
    :param n_clusters: expected number of clusters
    :param covariance_type: the covariance type
    :return: the initialized GMM parameters
    """
    centers, indices = kmeans_plusplus(X, n_clusters)
    assignment = np.argmin(cdist(X, X[indices]), axis=1)
    pi = _init_pi_kmeanspp(assignment, n_clusters)
    if covariance_type == 'diag':
        Sigma = _init_Sigma_kmeanspp_diagcov(X, assignment, n_clusters)
    elif covariance_type == 'spherical':
        Sigma = _init_Sigma_kmeanspp_sphericalcov(X, assignment, n_clusters)
    elif covariance_type == 'tieddiag':
        Sigma = _init_Sigma_kmeanspp_tieddiagcov(X, assignment, n_clusters)
    else:
        raise ValueError('covariance_type')
    return GMMParams(pi, centers.copy(), Sigma)


def stack_params(params: ty.List[GMMParams]) -> GMMParams:
    return GMMParams(
        np.stack([p.pi for p in params]),
        np.stack([p.mu for p in params]),
        np.stack([p.Sigma for p in params]),
    )


class RegularizedGaussianMixture(BaseEstimator):
    def __init__(
        self,
        n_components: int = 1,
        covariance_type: CovType_t = 'diag',
        eta: float = 0.0,
        tol: float = 1e-3,
        eps: float = 1e-60,
        max_iter: int = 100,
        n_init: int = 1,
        n_jobs: int = 1,
    ) -> None:
        """
        :param n_components:
        :param covariance_type:
        :param eta:
        :param tol:
        :param eps:
        :param max_iter:
        :param n_init:
        :param n_jobs: if -1, n_jobs is decided by OpenMP
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.eta = eta
        self.tol = tol
        self.eps = eps
        self.max_iter = max_iter
        self.n_init = n_init
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = np.asarray(check_array(X, dtype=np.float64))
        self.n_features_in_ = X.shape[1]
        m, p = X.shape
        params = stack_params([
            init_params_kmenaspp(X, self.n_components, self.covariance_type)
            for _ in range(self.n_init)
        ])
        eta = self.eta * np.ones(self.n_components)
        fit_func = {
            'diag': c_reggmm.reggmm_diagcov_mt,
            'spherical': c_reggmm.reggmm_sphericalcov_mt,
            'tieddiag': c_reggmm.reggmm_tieddiagcov_mt,
        }[self.covariance_type]
        self.n_iter_ = fit_func(X, params.pi, params.mu, params.Sigma, eta, m,
                                self.n_components, p, self.max_iter, self.tol,
                                self.eps, self.n_init, self.n_jobs)
        self.converged_ = True if self.n_iter_ > 0 else False
        self.weights_ = params.pi[0]
        self.means_ = params.mu[0]
        self.covariances_ = params.Sigma[0]
        self.precisions_ = np.reciprocal(self.covariances_)
        return self

    def score(self, X, y=None):
        check_is_fitted(self, [
            'n_features_in_', 'n_iter_', 'converged_', 'weights_', 'means_',
            'covariances_', 'precisions_'
        ])
        X = np.asarray(check_array(X, dtype=np.float64))
        m, p = X.shape
        score_func = {
            'diag': c_reggmm.ll_diagcov,
            'spherical': c_reggmm.ll_sphericalcov,
            'tieddiag': c_reggmm.ll_tieddiagcov,
        }[self.covariance_type]
        return score_func(X, self.weights_, self.means_, self.covariances_, m,
                          self.n_components, p, self.eps)

    def bic_score(self, X, y=None):
        """The larger, the better"""
        m, p = X.shape
        ll = self.score(X)
        ddof = {
            'diag': self.n_components + 2 * self.n_components * p,
            'spherical': self.n_components * (2 + p),
            'tieddiag': self.n_components + self.n_components * p + p,
        }[self.covariance_type]
        return 2 * ll - ddof * np.log(m)

    def predict_proba(self, X):
        check_is_fitted(self, [
            'n_features_in_', 'n_iter_', 'converged_', 'weights_', 'means_',
            'covariances_', 'precisions_'
        ])
        X = np.asarray(check_array(X, dtype=np.float64))
        m, p = X.shape
        predict_func = {
            'diag': c_reggmm.responsibility_diagcov,
            'spherical': c_reggmm.responsibility_sphericalcov,
            'tieddiag': c_reggmm.responsibility_tieddiagcov,
        }[self.covariance_type]
        gamma = np.empty((m, self.n_components), dtype=np.float64)
        predict_func(X, self.weights_, self.means_, self.covariances_, m,
                     self.n_components, p, self.eps, gamma)
        return gamma

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def test_regularized_gaussian_mixture():
    check_estimator(RegularizedGaussianMixture())
