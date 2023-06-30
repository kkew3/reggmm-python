import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport exp, sqrt, log, fabs
from libc.stdlib cimport malloc, free


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double c_det_diagcov(
    double [:] Sigma_k,
    int p,
) nogil:
    """determinant of diagonal covariance"""
    cdef int i
    cdef double x = 1.0
    for i in range(p):
        x *= Sigma_k[i]
    return x


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef double c_jointprob_diagcov(
#     double [:] x_i,
#     double pi_k,
#     double [:] mu_k,
#     double [:] Sigma_k,
#     int p,
#     double eps,
# ) nogil:
#     """P(z=k, x_i)"""
#     cdef double pw = 0.0
#     cdef int j
#     for j in range(p):
#         pw += 1.0 / (eps + Sigma_k[j]) * (x_i[j] - mu_k[j])**2
#     pw = exp(-0.5 * pw)
#     return pi_k * 1.0 / (eps + sqrt(c_det_diagcov(Sigma_k, p))) * pw


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double c_logsumexp_diagcov(
    double [:] x_i,
    double [:] pi,
    double [:, :] mu,
    double [:, :] Sigma,
    int K,
    int p,
    double eps,
) nogil:
    cdef:
        double *u = <double *> malloc(K * sizeof(double))
        double pw
        double c
        int j
        int k
    for j in range(K):
        u[j] = 0.0
        u[j] += log(pi[j])
        u[j] -= 0.5 * log(c_det_diagcov(Sigma[j], p))
        pw = 0.0
        for k in range(p):
            pw += 1.0 / (eps + Sigma[j, k]) * (x_i[k] - mu[j, k])**2
        u[j] -= pw / 2
    c = u[0]
    for j in range(1, K):
        if u[j] > c:
            c = u[j]
    for j in range(K):
        u[j] -= c
    pw = 0.0
    for j in range(K):
        pw += exp(u[j])
    free(u)
    return c + log(pw)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double c_ll_diagcov(
    double [:, :] X,
    double [:] pi,
    double [:, :] mu,
    double [:, :] Sigma,
    int m,
    int K,
    int p,
    double eps,
) nogil:
    """The log likelihood"""
    cdef:
        int i
        double z
    z = 0.0
    for i in range(m):
        z += c_logsumexp_diagcov(X[i], pi, mu, Sigma, K, p, eps)
    return z / m
#     cdef:
#         int i
#         int k
#         double x
#         double z
#     z = 0.0
#     for i in range(m):
#         x = 0.0
#         for k in range(K):
#             x += c_jointprob_diagcov(X[i], pi[k], mu[k], Sigma[k], p, eps)
#         z += log(x)
#     return z / m


def ll_diagcov(
    double [:, ::1] X,
    double [::1] pi,
    double [:, ::1] mu,
    double [:, ::1] Sigma,
    int m,
    int K,
    int p,
    double eps,
):
    return c_ll_diagcov(X, pi, mu, Sigma, m, K, p, eps)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void c_responsibility_diagcov(
    double [:, :] X,
    double [:] pi,
    double [:, :] mu,
    double [:, :] Sigma,
    int m,
    int K,
    int p,
    double eps,
    double [:, :] gamma,
) nogil:
    cdef:
        int i
        int j
        int k
        double pw
        double z
    for i in range(m):
        for k in range(K):
            z = 0.0
            z += log(pi[k])
            z -= 0.5 * log(c_det_diagcov(Sigma[k], p))
            pw = 0.0
            for j in range(p):
                pw += 1.0 / (eps + Sigma[k, j]) * (X[i, j] - mu[k, j])**2
            z -= pw / 2
            z -= c_logsumexp_diagcov(X[i], pi, mu, Sigma, K, p, eps)
            gamma[i, k] = exp(z)
    # cdef:
    #     int i
    #     int j
    #     int k
    # for i in range(m):
    #     for k in range(K):
    #         x = c_jointprob_diagcov(X[i], pi[k], mu[k], Sigma[k], p, eps)
    #         y = 0.0
    #         for j in range(K):
    #             y += c_jointprob_diagcov(X[i], pi[j], mu[j], Sigma[j], p, eps)
    #         gamma[i, k] = x / (y + eps)


def responsibility_diagcov(
    double [:, ::1] X,
    double [::1] pi,
    double [:, ::1] mu,
    double [:, ::1] Sigma,
    int m,
    int K,
    int p,
    double eps,
    double [:, ::1] gamma,
):
    c_responsibility_diagcov(X, pi, mu, Sigma, m, K, p, eps, gamma)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int c_reggm_diagcov(
    double [:, :] X,
    double [:] pi,
    double [:, :] mu,
    double [:, :] Sigma,
    double [:] eta,
    double [:, :] gamma,
    int m,
    int K,
    int p,
    int max_iter,
    double tol,
    double eps,
) nogil:
    """
    :param X: the data, of shape (m, p), readonly
    :param pi: initial priors, of shape (K,)
    :param mu: initial means, of shape (K, p)
    :param Sigma: initial covariances, of shape (K, p)
    :param eta: of shape (K,)
    :param gamma: of shape (m, K)
    :param m: the number of samples
    :param K: the number of clusters
    :param p: the dimensionality
    :param max_iter: max number of iterations
    :param tol: when the maximum update is no larger than `tol`, the algorithm
           is regarded as converged
    :param eps: small positive real to avoid division by zero error
    :return: -1 for not converged, otherwise the steps to converge
    """
    cdef:
        int t
        int i
        int j
        int k
        double x
        double y
        double z
        double u
    for t in range(max_iter):
        # E-step
        c_responsibility_diagcov(X, pi, mu, Sigma, m, K, p, eps, gamma)
        # M-step
        u = 0.0
        for k in range(K):
            x = 0.0
            for i in range(m):
                x += gamma[i, k]
            x /= m
            u += fabs(pi[k] - x)
            pi[k] = x
        for k in range(K):
            y = 0.0
            for i in range(m):
                y += gamma[i, k]
            for j in range(p):
                x = 0.0
                for i in range(m):
                    x += gamma[i, k] * X[i, j]
                z = x / y
                u += fabs(mu[k, j] - z)
                mu[k, j] = z
        for k in range(K):
            y = eta[k]
            for i in range(m):
                y += gamma[i, k]
            for j in range(p):
                x = eta[k]
                for i in range(m):
                    x += gamma[i, k] * (X[i, j] - mu[k, j])**2
                z = x / y
                u += fabs(Sigma[k, j] - z)
                Sigma[k, j] = z

        if u / (K + 2 * K * p) <= tol:
            return t + 1
    return -1


def reggmm_diagcov_mt(
    double [:, ::1] X,
    double [:, ::1] pi,
    double [:, :, ::1] mu,
    double [:, :, ::1] Sigma,
    double [::1] eta,
    int m,
    int K,
    int p,
    int max_iter,
    double tol,
    double eps,
    int n_init,
    int n_jobs,
):
    """
    `c_reggmm_diagcov` with multiple inits. The best parameters will be put at
    index 0 along the first axis of `pi`, `mu` and `Sigma`.
    """
    gamma_buff = np.empty((n_init, m, K), dtype=np.double)
    cdef double [:, :, ::1] gamma_buff_view = gamma_buff
    conv = np.zeros(n_init, dtype=np.intc)
    cdef int [::1] conv_view = conv
    scores = np.empty(n_init, dtype=np.double)
    cdef double [::1] scores_view = scores
    cdef int t
    if n_jobs > 1:
        for t in prange(n_init, nogil=True, num_threads=n_jobs):
            with cython.boundscheck(False):
                conv_view[t] = c_reggm_diagcov(
                    X, pi[t], mu[t], Sigma[t], eta, gamma_buff_view[t], m, K,
                    p, max_iter, tol, eps)
                scores_view[t] = c_ll_diagcov(
                    X, pi[t], mu[t], Sigma[t], m, K, p, eps)
    elif n_jobs == 1:
        for t in range(n_init):
            conv_view[t] = c_reggm_diagcov(
                X, pi[t], mu[t], Sigma[t], eta, gamma_buff_view[t], m, K,
                p, max_iter, tol, eps)
            scores_view[t] = c_ll_diagcov(
                X, pi[t], mu[t], Sigma[t], m, K, p, eps)
    else:
        for t in prange(n_init, nogil=True):
            with cython.boundscheck(False):
                conv_view[t] = c_reggm_diagcov(
                    X, pi[t], mu[t], Sigma[t], eta, gamma_buff_view[t], m, K,
                    p, max_iter, tol, eps)
                scores_view[t] = c_ll_diagcov(
                    X, pi[t], mu[t], Sigma[t], m, K, p, eps)
    cdef double max_score = scores_view[0]
    cdef int max_index = 0
    for t in range(1, n_init):
        if scores_view[t] > max_score:
            max_score = scores_view[t]
            max_index = t
    if t != 0:
        pi[0] = pi[t]
        mu[0] = mu[t]
        Sigma[0] = Sigma[t]
    return conv_view[t]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double c_det_sphericalcov(
    double Sigma_k,
    int p,
) nogil:
    """determinant of spherical covariance"""
    return Sigma_k * p


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double c_logsumexp_sphericalcov(
    double [:] x_i,
    double [:] pi,
    double [:, :] mu,
    double [:] Sigma,
    int K,
    int p,
    double eps,
) nogil:
    cdef:
        double *u = <double *> malloc(K * sizeof(double))
        double pw
        double c
        int j
        int k
    for j in range(K):
        u[j] = 0.0
        u[j] += log(pi[j])
        u[j] -= 0.5 * log(c_det_sphericalcov(Sigma[j], p))
        pw = 0.0
        for k in range(p):
            pw += (x_i[k] - mu[j, k])**2
        u[j] -= pw / (eps + Sigma[j]) / 2
    c = u[0]
    for j in range(1, K):
        if u[j] > c:
            c = u[j]
    for j in range(K):
        u[j] -= c
    pw = 0.0
    for j in range(K):
        pw += exp(u[j])
    free(u)
    return c + log(pw)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef double c_jointprob_sphericalcov(
#     double [:] x_i,
#     double pi_k,
#     double [:] mu_k,
#     double Sigma_k,
#     int p,
#     double eps,
# ) nogil:
#     """P(z=k, x_i)"""
#     cdef double pw = 0.0
#     cdef int j
#     for j in range(p):
#         pw += (x_i[j] - mu_k[j])**2
#     pw = exp(-0.5 / (eps + Sigma_k) * pw)
#     return pi_k * 1.0 / (eps + sqrt(c_det_sphericalcov(Sigma_k, p))) * pw


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double c_ll_sphericalcov(
    double [:, :] X,
    double [:] pi,
    double [:, :] mu,
    double [:] Sigma,
    int m,
    int K,
    int p,
    double eps,
) nogil:
    """The log likelihood"""
    cdef:
        int i
        double z
    z = 0.0
    for i in range(m):
        z += c_logsumexp_sphericalcov(X[i], pi, mu, Sigma, K, p, eps)
    return z / m
    # z = 0.0
    # for i in range(m):
    #     x = 0.0
    #     for k in range(K):
    #         x += c_jointprob_sphericalcov(X[i], pi[k], mu[k], Sigma[k], p, eps)
    #     z += log(x)
    # return z / m


def ll_sphericalcov(
    double [:, ::1] X,
    double [::1] pi,
    double [:, ::1] mu,
    double [::1] Sigma,
    int m,
    int K,
    int p,
    double eps,
):
    return c_ll_sphericalcov(X, pi, mu, Sigma, m, K, p, eps)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void c_responsibility_sphericalcov(
    double [:, :] X,
    double [:] pi,
    double [:, :] mu,
    double [:] Sigma,
    int m,
    int K,
    int p,
    double eps,
    double [:, :] gamma,
) nogil:
    cdef:
        int i
        int j
        int k
        double pw
        double z
    for i in range(m):
        for k in range(K):
            z = 0.0
            z += log(pi[k])
            z -= 0.5 * log(c_det_sphericalcov(Sigma[k], p))
            pw = 0.0
            for j in range(p):
                pw += (X[i, j] - mu[k, j])**2
            z -= pw / (eps + Sigma[k]) / 2
            z -= c_logsumexp_sphericalcov(X[i], pi, mu, Sigma, K, p, eps)
            gamma[i, k] = exp(z)
    # for i in range(m):
    #     for k in range(K):
    #         x = c_jointprob_sphericalcov(X[i], pi[k], mu[k], Sigma[k], p, eps)
    #         y = 0.0
    #         for j in range(K):
    #             y += c_jointprob_sphericalcov(X[i], pi[j], mu[j], Sigma[j], p, eps)
    #         gamma[i, k] = x / (y + eps)


def responsibility_sphericalcov(
    double [:, ::1] X,
    double [::1] pi,
    double [:, ::1] mu,
    double [::1] Sigma,
    int m,
    int K,
    int p,
    double eps,
    double [:, ::1] gamma,
):
    c_responsibility_sphericalcov(X, pi, mu, Sigma, m, K, p, eps, gamma)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int c_reggmm_sphericalcov(
    double [:, :] X,
    double [:] pi,
    double [:, :] mu,
    double [:] Sigma,
    double [:] eta,
    double [:, :] gamma,
    int m,
    int K,
    int p,
    int max_iter,
    double tol,
    double eps,
) nogil:
    """
    :param X: the data, of shape (m, p), readonly
    :param pi: initial priors, of shape (K,)
    :param mu: initial means, of shape (K, p)
    :param Sigma: initial covariances, of shape (K,)
    :param eta: of shape (K,)
    :param gamma: of shape (m, K)
    :param m: the number of samples
    :param K: the number of clusters
    :param p: the dimensionality
    :param max_iter: max number of iterations
    :param tol: when the maximum update is no larger than `tol`, the algorithm
           is regarded as converged
    :param eps: small positive real to avoid division by zero error
    :return: -1 for not converged, otherwise the steps to converge
    """
    cdef:
        int t
        int i
        int j
        int k
        double x
        double y
        double z
        double u
    for t in range(max_iter):
        # E-step
        c_responsibility_sphericalcov(X, pi, mu, Sigma, m, K, p, eps, gamma)
        # M-step
        u = 0.0
        for k in range(K):
            x = 0.0
            for i in range(m):
                x += gamma[i, k]
            x /= m
            u += fabs(pi[k] - x)
            pi[k] = x
        for k in range(K):
            y = 0.0
            for i in range(m):
                y += gamma[i, k]
            for j in range(p):
                x = 0.0
                for i in range(m):
                    x += gamma[i, k] * X[i, j]
                z = x / y
                u += fabs(mu[k, j] - z)
                mu[k, j] = z
        for k in range(K):
            y = eta[k]
            for i in range(m):
                y += gamma[i, k]
            x = eta[k] * p
            for j in range(p):
                for i in range(m):
                    x += gamma[i, k] * (X[i, j] - mu[k, j])**2
            z = x / y
            u += fabs(Sigma[k] - z)
            Sigma[k] = z

        if u / (2 * K + K * p) <= tol:
            return t + 1
    return -1


def reggmm_sphericalcov_mt(
    double [:, ::1] X,
    double [:, ::1] pi,
    double [:, :, ::1] mu,
    double [:, ::1] Sigma,
    double [::1] eta,
    int m,
    int K,
    int p,
    int max_iter,
    double tol,
    double eps,
    int n_init,
    int n_jobs,
):
    """
    `c_reggmm_sphericalcov` with multiple inits. The best parameters will be
    put at index 0 along the first axis of `pi`, `mu` and `Sigma`.
    """
    gamma_buff = np.empty((n_init, m, K), dtype=np.double)
    cdef double [:, :, ::1] gamma_buff_view = gamma_buff
    conv = np.zeros(n_init, dtype=np.intc)
    cdef int [::1] conv_view = conv
    scores = np.empty(n_init, dtype=np.double)
    cdef double [::1] scores_view = scores
    cdef int t
    if n_jobs > 0:
        for t in prange(n_init, nogil=True, num_threads=n_jobs):
            with cython.boundscheck(False):
                conv_view[t] = c_reggmm_sphericalcov(
                    X, pi[t], mu[t], Sigma[t], eta, gamma_buff_view[t], m, K,
                    p, max_iter, tol, eps)
                scores_view[t] = c_ll_sphericalcov(
                    X, pi[t], mu[t], Sigma[t], m, K, p, eps)
    elif n_jobs == 1:
        for t in range(n_init, nogil=True):
            conv_view[t] = c_reggmm_sphericalcov(
                X, pi[t], mu[t], Sigma[t], eta, gamma_buff_view[t], m, K,
                p, max_iter, tol, eps)
            scores_view[t] = c_ll_sphericalcov(
                X, pi[t], mu[t], Sigma[t], m, K, p, eps)
    else:
        for t in prange(n_init, nogil=True):
            with cython.boundscheck(False):
                conv_view[t] = c_reggmm_sphericalcov(
                    X, pi[t], mu[t], Sigma[t], eta, gamma_buff_view[t], m, K,
                    p, max_iter, tol, eps)
                scores_view[t] = c_ll_sphericalcov(
                    X, pi[t], mu[t], Sigma[t], m, K, p, eps)
    cdef double max_score = scores_view[0]
    cdef int max_index = 0
    for t in range(1, n_init):
        if scores_view[t] > max_score:
            max_score = scores_view[t]
            max_index = t
    if t != 0:
        pi[0] = pi[t]
        mu[0] = mu[t]
        Sigma[0] = Sigma[t]
    return conv_view[t]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double c_logsumexp_tieddiagcov(
    double [:] x_i,
    double [:] pi,
    double [:, :] mu,
    double [:] Sigma,
    int K,
    int p,
    double eps,
) nogil:
    cdef:
        double *u = <double *> malloc(K * sizeof(double))
        double pw
        double c
        int j
        int k
    for j in range(K):
        u[j] = 0.0
        u[j] += log(pi[j])
        u[j] -= 0.5 * log(c_det_diagcov(Sigma, p))
        pw = 0.0
        for k in range(p):
            pw += 1.0 / (eps + Sigma[k]) * (x_i[k] - mu[j, k])**2
        u[j] -= pw / 2
    c = u[0]
    for j in range(1, K):
        if u[j] > c:
            c = u[j]
    for j in range(K):
        u[j] -= c
    pw = 0.0
    for j in range(K):
        pw += exp(u[j])
    free(u)
    return c + log(pw)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double c_ll_tieddiagcov(
    double [:, :] X,
    double [:] pi,
    double [:, :] mu,
    double [:] Sigma,
    int m,
    int K,
    int p,
    double eps,
) nogil:
    """The log likelihood"""
    cdef:
        int i
        double z
    z = 0.0
    for i in range(m):
        z += c_logsumexp_tieddiagcov(X[i], pi, mu, Sigma, K, p, eps)
    return z / m
    # z = 0.0
    # for i in range(m):
    #     x = 0.0
    #     for k in range(K):
    #         x += c_jointprob_diagcov(X[i], pi[k], mu[k], Sigma, p, eps)
    #     z += log(x)
    # return z / m


def ll_tieddiagcov(
    double [:, ::1] X,
    double [::1] pi,
    double [:, ::1] mu,
    double [::1] Sigma,
    int m,
    int K,
    int p,
    double eps,
):
    return c_ll_tieddiagcov(X, pi, mu, Sigma, m, K, p, eps)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void c_responsibility_tieddiagcov(
    double [:, :] X,
    double [:] pi,
    double [:, :] mu,
    double [:] Sigma,
    int m,
    int K,
    int p,
    double eps,
    double [:, :] gamma,
) nogil:
    cdef:
        int i
        int j
        int k
        double pw
        double z
    for i in range(m):
        for k in range(K):
            z = 0.0
            z += log(pi[k])
            z -= 0.5 * log(c_det_diagcov(Sigma, p))
            pw = 0.0
            for j in range(p):
                pw += 1.0 / (eps + Sigma[j]) * (X[i, j] - mu[k, j])**2
            z -= pw / 2
            z -= c_logsumexp_tieddiagcov(X[i], pi, mu, Sigma, K, p, eps)
            gamma[i, k] = exp(z)
    # for i in range(m):
    #     for k in range(K):
    #         x = c_jointprob_diagcov(X[i], pi[k], mu[k], Sigma, p, eps)
    #         y = 0.0
    #         for j in range(K):
    #             y += c_jointprob_diagcov(X[i], pi[j], mu[j], Sigma, p, eps)
    #         gamma[i, k] = x / (y + eps)


def responsibility_tieddiagcov(
    double [:, ::1] X,
    double [::1] pi,
    double [:, ::1] mu,
    double [::1] Sigma,
    int m,
    int K,
    int p,
    double eps,
    double [:, ::1] gamma,
):
    c_responsibility_tieddiagcov(X, pi, mu, Sigma, m, K, p, eps, gamma)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int c_reggmm_tieddiagcov(
    double [:, :] X,
    double [:] pi,
    double [:, :] mu,
    double [:] Sigma,
    double [:] eta,
    double [:, :] gamma,
    int m,
    int K,
    int p,
    int max_iter,
    double tol,
    double eps,
) nogil:
    """
    :param X: the data, of shape (m, p), readonly
    :param pi: initial priors, of shape (K,)
    :param mu: initial means, of shape (K, p)
    :param Sigma: initial covariances, of shape (p,)
    :param eta: of shape (K,)
    :param gamma: of shape (m, K)
    :param m: the number of samples
    :param K: the number of clusters
    :param p: the dimensionality
    :param max_iter: max number of iterations
    :param tol: when the maximum update is no larger than `tol`, the algorithm
           is regarded as converged
    :param eps: small positive real to avoid division by zero error
    :return: -1 for not converged, otherwise the steps to converge
    """
    cdef:
        int t
        int i
        int j
        int k
        double x
        double y
        double z
        double u
    for t in range(max_iter):
        # E-step
        c_responsibility_tieddiagcov(X, pi, mu, Sigma, m, K, p, eps, gamma)
        # M-step
        u = 0.0
        for k in range(K):
            x = 0.0
            for i in range(m):
                x += gamma[i, k]
            x /= m
            u += fabs(pi[k] - x)
            pi[k] = x
        for k in range(K):
            y = 0.0
            for i in range(m):
                y += gamma[i, k]
            for j in range(p):
                x = 0.0
                for i in range(m):
                    x += gamma[i, k] * X[i, j]
                z = x / y
                u += fabs(mu[k, j] - z)
                mu[k, j] = z
        for j in range(p):
            y = m
            for k in range(K):
                y += eta[k]
            x = 0.0
            for k in range(K):
                x += eta[k]
            for k in range(K):
                for i in range(m):
                    x += gamma[i, k] * (X[i, j] - mu[k, j])**2
            z = x / y
            u += fabs(Sigma[j] - z)
            Sigma[j] = z

        if u / (K + K * p + p) <= tol:
            return t + 1
    return -1


def reggmm_tieddiagcov_mt(
    double [:, ::1] X,
    double [:, ::1] pi,
    double [:, :, ::1] mu,
    double [:, ::1] Sigma,
    double [::1] eta,
    int m,
    int K,
    int p,
    int max_iter,
    double tol,
    double eps,
    int n_init,
    int n_jobs,
):
    """
    `c_reggmm_tieddiagcov` with multiple inits. The best parameters will be put at
    index 0 along the first axis of `pi`, `mu` and `Sigma`.
    """
    gamma_buff = np.empty((n_init, m, K), dtype=np.double)
    cdef double [:, :, ::1] gamma_buff_view = gamma_buff
    conv = np.zeros(n_init, dtype=np.intc)
    cdef int [::1] conv_view = conv
    scores = np.empty(n_init, dtype=np.double)
    cdef double [::1] scores_view = scores
    cdef int t
    if n_jobs > 0:
        for t in prange(n_init, nogil=True, num_threads=n_jobs):
            with cython.boundscheck(False):
                conv_view[t] = c_reggmm_tieddiagcov(
                    X, pi[t], mu[t], Sigma[t], eta, gamma_buff_view[t], m, K,
                    p, max_iter, tol, eps)
                scores_view[t] = c_ll_tieddiagcov(
                    X, pi[t], mu[t], Sigma[t], m, K, p, eps)
    elif n_jobs == 1:
        for t in range(n_init, nogil=True):
            conv_view[t] = c_reggmm_tieddiagcov(
                X, pi[t], mu[t], Sigma[t], eta, gamma_buff_view[t], m, K,
                p, max_iter, tol, eps)
            scores_view[t] = c_ll_tieddiagcov(
                X, pi[t], mu[t], Sigma[t], m, K, p, eps)
    else:
        for t in prange(n_init, nogil=True):
            with cython.boundscheck(False):
                conv_view[t] = c_reggmm_tieddiagcov(
                    X, pi[t], mu[t], Sigma[t], eta, gamma_buff_view[t], m, K,
                    p, max_iter, tol, eps)
                scores_view[t] = c_ll_tieddiagcov(
                    X, pi[t], mu[t], Sigma[t], m, K, p, eps)
    cdef double max_score = scores_view[0]
    cdef int max_index = 0
    for t in range(1, n_init):
        if scores_view[t] > max_score:
            max_score = scores_view[t]
            max_index = t
    if t != 0:
        pi[0] = pi[t]
        mu[0] = mu[t]
        Sigma[0] = Sigma[t]
    return conv_view[t]
