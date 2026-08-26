"""CMA-ES: Covariance Matrix Adaptation Evolution Strategy.

Reference:
    Hansen, N. (2006). The CMA evolution strategy: A tutorial.
    arXiv preprint arXiv:1604.00772.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from atlas.core.base_algorithm import BaseAlgorithm
from atlas.core.base_problem import BaseProblem
from atlas.utils.registry import register_algorithm


@register_algorithm("cmaes", aliases=["cma_es", "cmaes_nfe"])
class CMAES(BaseAlgorithm):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

    Args:
        problem: Optimisation problem instance.
        max_iter: Maximum number of iterations.
        pop_size: Offspring sample size lambda (defaults to
            ``4 + floor(3 * ln(dim))``).
        seed: Random seed for reproducibility.
        sigma0: Initial step size.
        verbose: Print progress.
    """

    def __init__(
        self,
        problem: BaseProblem,
        max_iter: int = 500,
        pop_size: Optional[int] = None,
        seed: int = 42,
        sigma0: float = 0.3,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        dim = problem.get_dim()
        # Default lambda: 4 + floor(3 * ln(dim))
        lam = pop_size if pop_size is not None else (4 + int(3 * math.log(dim)))
        super().__init__(
            problem,
            max_iter=max_iter,
            pop_size=lam,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )
        self.sigma0 = sigma0
        self._lam = lam  # offspring size (lambda)
        self._mu = lam // 2  # number of parents (mu)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        dim = self.dim
        lb, ub = self.lb, self.ub

        # Initial mean: centre of the search space
        self._mean: np.ndarray = (np.asarray(lb) + np.asarray(ub)) / 2.0
        self._sigma: float = self.sigma0

        # Covariance matrix and its Cholesky factor
        self._C: np.ndarray = np.eye(dim)
        self._A: np.ndarray = np.eye(dim)  # Cholesky of C (lower triangular)

        # Evolution paths
        self._p_sigma: np.ndarray = np.zeros(dim)
        self._p_c: np.ndarray = np.zeros(dim)

        # Iteration counter for the generation clock
        self._gen: int = 0

        # ----- Weights and learning rates (Hansen 2006 Eq. 53) -----
        mu = self._mu
        lam = self._lam

        # Raw positive weights for log-ranking
        raw_w = np.array([math.log((lam + 1) / 2) - math.log(i + 1) for i in range(lam)])
        # Keep only positive weights (top mu)
        pos_w = raw_w[:mu]
        self._weights: np.ndarray = pos_w / pos_w.sum()  # normalized, shape (mu,)

        mu_eff = 1.0 / (self._weights ** 2).sum()  # effective sample size

        # Step-size control (CSA)
        self._c_sigma: float = (mu_eff + 2) / (dim + mu_eff + 5)
        self._d_sigma: float = (
            1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + self._c_sigma
        )
        self._chi_n: float = math.sqrt(dim) * (
            1 - 1 / (4 * dim) + 1 / (21 * dim ** 2)
        )

        # Covariance matrix adaptation
        self._c_c: float = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        self._c_1: float = 2 / ((dim + 1.3) ** 2 + mu_eff)
        self._c_mu: float = min(
            1 - self._c_1,
            2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff),
        )

        # Evaluate initial mean to set g_best_f
        f0 = self.evaluate(self._mean.copy())
        # Populate self.population and self.fitness with a single point
        # so that diversity tracker (if enabled) doesn't crash
        self.population = self._mean.reshape(1, dim)
        self.fitness = np.array([f0])

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------
    def iterate(self, iter_idx: int) -> float:
        dim = self.dim
        lam = self._lam
        mu = self._mu

        # ----- 1. Sample lambda offspring -----
        z_arr = self.rng.standard_normal((lam, dim))  # (lam, dim)
        y_arr = z_arr @ self._A.T  # (lam, dim)  — A is lower-triangular Cholesky
        x_arr = self._mean + self._sigma * y_arr  # (lam, dim)
        x_arr = self._clip(x_arr)

        # ----- 2. Evaluate -----
        f_arr = np.empty(lam)
        for k in range(lam):
            f_arr[k] = self.evaluate(x_arr[k])

        # ----- 3. Sort by fitness (ascending) -----
        order = np.argsort(f_arr)
        x_sorted = x_arr[order]  # (lam, dim)
        y_sorted = y_arr[order]  # (lam, dim)

        # Update public population/fitness for any trackers
        self.population = x_sorted
        self.fitness = f_arr[order]

        # ----- 4. Weighted recombination: update mean -----
        x_mu = x_sorted[:mu]  # (mu, dim)
        y_mu = y_sorted[:mu]  # (mu, dim)

        old_mean = self._mean.copy()
        self._mean = (self._weights @ x_mu)  # (dim,)

        # Weighted step in y-space
        y_w = self._weights @ y_mu  # (dim,)

        # ----- 5. Step-size path (CSA) -----
        invsqrt_C = self._invsqrt_C()
        self._p_sigma = (
            (1 - self._c_sigma) * self._p_sigma
            + math.sqrt(self._c_sigma * (2 - self._c_sigma) * self._mu_eff()) * (invsqrt_C @ y_w)
        )

        # Hedges' h_sigma flag
        p_sigma_norm = float(np.linalg.norm(self._p_sigma))
        gen = self._gen + 1
        h_sigma = 1.0 if (
            p_sigma_norm / math.sqrt(1 - (1 - self._c_sigma) ** (2 * gen))
            < (1.4 + 2 / (dim + 1)) * self._chi_n
        ) else 0.0

        # Update sigma
        self._sigma *= math.exp(
            (self._c_sigma / self._d_sigma) * (p_sigma_norm / self._chi_n - 1)
        )
        self._sigma = max(1e-20, self._sigma)

        # ----- 6. Covariance evolution path (CMA) -----
        self._p_c = (
            (1 - self._c_c) * self._p_c
            + h_sigma * math.sqrt(self._c_c * (2 - self._c_c) * self._mu_eff()) * y_w
        )

        # Rank-mu update
        delta_h_sigma = (1 - h_sigma) * self._c_c * (2 - self._c_c)
        rank_mu = sum(
            self._weights[k] * np.outer(y_mu[k], y_mu[k]) for k in range(mu)
        )

        self._C = (
            (1 - self._c_1 - self._c_mu) * self._C
            + self._c_1 * (np.outer(self._p_c, self._p_c) + delta_h_sigma * self._C)
            + self._c_mu * rank_mu
        )

        # ----- 7. Numerical stabilization -----
        self._C = (self._C + self._C.T) / 2.0
        # Clamp eigenvalues to be positive
        eigvals = np.linalg.eigvalsh(self._C)
        if eigvals.min() < 1e-20:
            self._C += (1e-20 - eigvals.min()) * np.eye(dim)

        # Recompute Cholesky factor
        try:
            self._A = np.linalg.cholesky(self._C)
        except np.linalg.LinAlgError:
            # Fall back to re-conditioning
            self._C = np.eye(dim)
            self._A = np.eye(dim)

        self._gen += 1
        return self.g_best_f

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _mu_eff(self) -> float:
        """Effective sample size (cached from weights)."""
        return float(1.0 / (self._weights ** 2).sum())

    def _invsqrt_C(self) -> np.ndarray:
        """Compute C^{-1/2} via eigendecomposition."""
        eigvals, eigvecs = np.linalg.eigh(self._C)
        eigvals = np.maximum(eigvals, 1e-20)
        return eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
