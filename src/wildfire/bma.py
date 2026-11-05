"""Bayesian logistic subset averaging with Laplace marginal likelihoods.

Independent Bernoulli block-inclusion model prior over 16 subsets; independent N(0, prior_scale**2)
coefficients, including intercept. Predictive integration uses Gauss-Hermite
quadrature over each linear predictor's Laplace Gaussian posterior.
"""
from itertools import product
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from .data import BLOCKS


class BayesianModelAverage(ClassifierMixin, BaseEstimator):
    def __init__(self, prior_scale=1.0, quadrature_points=20, inclusion_prior=0.5):
        self.inclusion_prior = inclusion_prior
        self.prior_scale = prior_scale
        self.quadrature_points = quadrature_points

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        if self.prior_scale <= 0 or set(np.unique(y)) != {0, 1}:
            raise ValueError('Positive prior scale and both binary classes are required.')
        if not 0 < self.inclusion_prior < 1:
            raise ValueError("inclusion_prior must lie strictly between 0 and 1.")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.models_ = []
        precision = 1 / self.prior_scale**2
        for mask in product([False, True], repeat=len(BLOCKS)):
            names = [name for name, active in zip(BLOCKS, mask) if active]
            columns = [j for name in names for j in BLOCKS[name]]
            design = np.column_stack([np.ones(len(X)), X[:, columns]])
            k = design.shape[1]

            def objective(beta):
                z = design @ beta
                return np.sum(np.logaddexp(0, z) - y * z) + .5 * precision * (beta @ beta)

            def gradient(beta):
                return design.T @ (expit(design @ beta) - y) + precision * beta

            def hessian(beta):
                p = expit(design @ beta)
                return design.T @ ((p * (1 - p))[:, None] * design) + precision * np.eye(k)

            fit = minimize(objective, np.zeros(k), jac=gradient, hess=hessian,
                           method='trust-exact', options={'gtol': 1e-7, 'maxiter': 300})
            grad_norm = float(np.max(np.abs(gradient(fit.x))))
            if not np.isfinite(fit.fun) or grad_norm > 1e-4:
                raise RuntimeError(f'BMA optimization failed for {names}: {fit.message}')
            H = hessian(fit.x)
            sign, logdet = np.linalg.slogdet(H)
            if sign <= 0:
                raise RuntimeError('Non-positive posterior precision.')
            # log p(y|M) ≈ log likelihood + normalized log prior
            #                  + k/2 log(2π) - 1/2 log|H|.
            evidence = -fit.fun - k * np.log(self.prior_scale) - .5 * logdet
            self.models_.append(dict(name=' + '.join(names) or 'intercept_only',
                                     columns=columns, mean=fit.x, covariance=np.linalg.inv(H),
                                     log_evidence=float(evidence), gradient_max=grad_norm,
                                     log_model_prior=float(len(names) * np.log(self.inclusion_prior)
                                         + (len(BLOCKS) - len(names)) * np.log1p(-self.inclusion_prior))))
        log_evidence = np.array([m['log_evidence'] + m['log_model_prior'] for m in self.models_])
        self.weights_ = np.exp(log_evidence - logsumexp(log_evidence))
        return self

    def candidate_probabilities(self, X):
        check_is_fitted(self, 'weights_')
        X = np.asarray(X, dtype=float)
        nodes, weights = hermgauss(self.quadrature_points)
        predictions = []
        for model in self.models_:
            design = np.column_stack([np.ones(len(X)), X[:, model['columns']]])
            mean = design @ model['mean']
            variance = np.einsum('ij,jk,ik->i', design, model['covariance'], design)
            z = mean[:, None] + np.sqrt(2 * np.maximum(variance, 0))[:, None] * nodes
            predictions.append(expit(z) @ weights / np.sqrt(np.pi))
        return np.column_stack(predictions)

    def probability_interval(self, X, n_draws=2000, random_state=42):
        """Pointwise 90% latent-probability credible interval, conditional on this family.

        Samples both the model index and its Laplace coefficient posterior. This is
        not a frequentist confidence interval, an outcome interval, or protection
        against dataset bias. Intended for small interactive scenario batches.
        """
        check_is_fitted(self, 'weights_')
        if n_draws < 100:
            raise ValueError('Use at least 100 posterior draws.')
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(random_state)
        choices = rng.choice(len(self.models_), size=n_draws, p=self.weights_)
        draws = np.empty((len(X), n_draws))
        for i, model in enumerate(self.models_):
            mask = choices == i
            if mask.any():
                beta = rng.multivariate_normal(model['mean'], model['covariance'], size=mask.sum())
                design = np.column_stack([np.ones(len(X)), X[:, model['columns']]])
                draws[:, mask] = expit(design @ beta.T)
        return np.quantile(draws, [.05, .95], axis=1).T

    def predict_proba(self, X):
        p = self.candidate_probabilities(X) @ self.weights_
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)

    def weight_table(self):
        check_is_fitted(self, 'weights_')
        return pd.DataFrame([{'candidate': m['name'], 'weight': w,
                              'log_evidence': m['log_evidence'],
                              'log_model_prior': m['log_model_prior'],
                              'parameters': len(m['mean']), 'gradient_max': m['gradient_max']}
                             for m, w in zip(self.models_, self.weights_)]).sort_values('weight', ascending=False)
