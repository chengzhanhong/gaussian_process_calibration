import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
import random
from numpy.linalg import det, inv, solve, cholesky
from numpy import log, ndarray, pi, eye, diag, exp
import pickle
import seaborn as sns


def reset_random_seeds(n=0):
    os.environ['PYTHONHASHSEED'] = str(n)
    np.random.seed(n)
    random.seed(n)

#%% different k-v mean functions
def zero_mean(x):
    return 0


def linear(x, beta=1, intercept=1):
    return beta * x + intercept


def greenshields(k, vf, kj):
    return vf * (1 - k / kj)


def greenberg(k, v0, kj):
    return v0 * log(kj / k)


def greenberg_inverse(v, v0, kj):
    return kj/exp(v/v0)


def underwood(k, vf, k0):
    return vf * exp(-k / k0)


def northwestern(k, vf, k0):
    return vf * exp(-1 / 2 * (k / k0) ** 2)


def newell(k, vf, kj, lam):
    return vf * (1 - exp(-lam / vf * (1 / k - 1 / kj)))


def three_params(k, vf, kc, theta):
    return vf / (1 + exp((k - kc) / theta))


#%% Kernel functions and mean functions for marginal likelihood GP
def SE_kernel(x1, x2, length_scale=1.0, variance=1.0):
    return variance * np.exp(-(x1 - np.transpose(x2)) ** 2 / (2 * length_scale ** 2))


def RQ_kernel(x1, x2, length_scale=1.0, variance=1.0, alpha=1.0):
    return variance * (1 + (x1 - np.transpose(x2)) ** 2 / (2 * alpha * length_scale ** 2))**(-alpha)


def Matern32(x1, x2, length_scale=1.0, variance=1.0):
    param = 3**0.5
    d = x1 - np.transpose(x2)
    return variance * (1 + param*np.abs(d)/length_scale) * np.exp(-param*np.abs(d)/length_scale)


#%% likelihood functions and
def neg_log_likelihood(x, y, mean_fun=zero_mean, cov_fun=SE_kernel,
                       mean_params=(), cov_params=(), sigma=1):
    """The log likelihood of the Gaussian process model.
    """
    x = x.reshape([-1, 1])
    y = y.reshape([-1, 1])
    n = x.shape[0]
    e = y - mean_fun(x, *mean_params)
    cov_mat = cov_fun(x, x, *cov_params) + sigma ** 2 * np.eye(n) + 1e-8 * np.eye(n)
    L = np.linalg.cholesky(cov_mat)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, e))
    neg_ll = 1 / 2 * e.T @ alpha + np.log(np.diag(L)).sum() + n / 2 * np.log(2 * np.pi)
    return neg_ll


def neg_log_likelihood_sparse(x, y, u, mean_fun=zero_mean, cov_fun=SE_kernel,
                              mean_params=(), cov_params=(), sigma=1):
    """The log likelihood of the sparse Gaussian process model."""
    x = x.reshape([-1, 1])
    y = y.reshape([-1, 1])
    u = u.reshape([-1, 1])
    n = x.shape[0]
    m = u.shape[0]
    e = y - mean_fun(x, *mean_params)

    K_uu = cov_fun(u, u, *cov_params)
    K_fu = cov_fun(x, u, *cov_params)

    L = cholesky(K_uu + 1e-5 * np.eye(m))
    A = solve(L, K_fu.T) * sigma ** (-1)
    B = A @ A.T + eye(m)
    Lb = cholesky(B)

    c = solve(Lb, A @ e) * sigma ** (-1)
    LL1 = 1 / 2 * (2 * log(diag(Lb)).sum() + 2 * n * log(sigma) +
                   sigma ** (-2) * e.T @ e -
                   c.T @ c +
                   n * log(2 * pi)) + \
          1 / 2 * sigma ** (-2) * n * cov_fun(1, 1, *cov_params) - \
          1 / 2 * np.sum(A * A)

    return LL1


def predict(x, y, u, x_new, mean_fun=zero_mean, cov_fun=SE_kernel,
            mean_params=(), cov_params=(), sigma=1):
    x = x.reshape([-1, 1])
    y = y.reshape([-1, 1])
    u = u.reshape([-1, 1])
    x_new = x_new.reshape([-1, 1])
    n = x.shape[0]
    m = u.shape[0]
    e = y - mean_fun(x, *mean_params)

    K_uu = cov_fun(u, u, *cov_params)
    K_fu = cov_fun(x, u, *cov_params)
    K_xu = cov_fun(x_new, u, *cov_params)

    L = cholesky(K_uu + 1e-5 * np.eye(m))
    A = solve(L, K_fu.T) * sigma ** (-1)
    B = A @ A.T + eye(m)
    Lb = cholesky(B)

    c = solve(Lb, A @ e) * sigma ** (-1)
    y_new_mean = K_xu @ inv(L).T @ inv(Lb).T @ c
    return y_new_mean + mean_fun(x_new, *mean_params)


def estimate_three_models(k, v, mean_fun, cov_fun, mean_param_init, cov_param_init,
                          sigma_init, num_u=20, display=True, plot=True):
    """
    k: density observations
    v: velocity observations
    mean_fun: the k-v fundamental diagram function
    cov_fun: the covariance function of the Gaussian process
    num_u: number of inducing points
    """
    w = get_weight(k)
    u = np.linspace(k.min(), k.max(), num_u)
    num1 = len(mean_param_init)
    num2 = len(cov_param_init)

    ls_fun = lambda x: square_loss(k, v, mean_fun=mean_fun, mean_params=x[0:num1])
    wls_fun = lambda x: square_loss(k, v, w, mean_fun=mean_fun, mean_params=x[0:num1])
    gp_sparse_fun = lambda x: neg_log_likelihood_sparse(k, v, u, mean_fun=mean_fun, cov_fun=cov_fun,
                                                        mean_params=x[0:num1],
                                                        cov_params=x[num1:(num1+num2)],
                                                        sigma=x[num2+num1])

    ls_res = minimize(ls_fun, x0=np.array(mean_param_init), method='L-BFGS-B')
    if display:
        print(f'LS:{ls_res.x}')

    wls_res = minimize(wls_fun, x0=np.array(ls_res.x), method='L-BFGS-B')
    if display:
        print(f'WLS:{wls_res.x}')

    bounds = [(None, None) for i in range(num1)] + [(0.001, np.inf) for i in range(num2+1)]
    gp_sparse_result = minimize(gp_sparse_fun, x0=np.concatenate((ls_res.x, cov_param_init, [sigma_init])),
                                method='L-BFGS-B', bounds=bounds)
    if display:
        print(f'GP{gp_sparse_result.x}')

    if plot:
        _, ax = plt.subplots()
        ax.plot(k, v, '.', alpha=0.1, ms=3, lw=1.5)
        ax.plot(k, mean_fun(k, *ls_res.x), label='least square')
        ax.plot(k, mean_fun(k, *wls_res.x), label='weighted least square')
        ax.plot(k, mean_fun(k, *gp_sparse_result.x[0:num1]), label='GP sparse estimation')
        # ax.plot(k, mean_fun(k, *gp_result.x[0:num1]), label='GP estimation')
        plt.legend()
    return ls_res, wls_res, gp_sparse_result, ax


def get_weight(k):
    """Get weight for weighted least square problem.
    Reference: Qu, X., Wang, S., & Zhang, J. (2015). On the fundamental diagram
    for freeway traffic: A novel calibration approach for single-regime models.
    Transportation Research Part B: Methodological, 73, 91-102.
    """
    mat = pd.DataFrame(data={'k':k})
    group = mat.groupby('k').size().reset_index(name='counts')

    # calculate weight
    w = []
    n = group.shape[0]
    for iter in range(n):
        c = group['counts'].iloc[iter]
        if iter == 0:
            k1 = group['k'].iloc[iter]
            k2 = group['k'].iloc[iter+1]
            w.append((k2-k1)/c)
        if iter > 0 and iter < n-1:
            k1 = group['k'].iloc[iter-1]
            k2 = group['k'].iloc[iter+1]
            w.append((k2-k1)/(2*c))
        if iter == n-1:
            k1 = group['k'].iloc[iter-1]
            k2 = group['k'].iloc[iter]
            w.append((k2-k1)/c)

    group['w'] = w
    result = pd.merge(mat, group, on='k')
    return result.w.values


def square_loss(k, v, w=1, mean_fun=greenshields, mean_params=()):
    return np.mean(w*(v - mean_fun(k, *mean_params)) ** 2)


custom_params = {"xtick.direction": 'in', "ytick.direction": 'in',
                 'xtick.major.size':4, 'ytick.major.size':4,
                 "xtick.bottom":True, "ytick.left":True,
                 "axes.linewidth":0.8,"xtick.major.width":0.8, "ytick.major.width":0.8,
                 "lines.solid_joinstyle": "butt","lines.solid_joinstyle":'bevel',
                "font.family":"sans-serif", "font.sans-serif":["Helvetica", "Arial"],
                 "font.size":10, "mathtext.fontset":'cm'}
sns.set_theme(style='white', rc=custom_params)



