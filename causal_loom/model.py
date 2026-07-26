"""Constrained weight model used by the synthetic-control estimators.

The implementation builds on the synthetic-control formulation demonstrated in
Matheus Facure's *Causal Inference for the Brave and True*. See
``THIRD_PARTY_NOTICES.md`` for attribution and licensing details.
"""

import warnings
from typing import Literal, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class SynthControlModel:
    """Synthetic control weights estimator with simplex constraints.

    Solves constrained optimization:
    - w_i >= 0
    - sum_i w_i = 1
    using SciPy SLSQP.

    Args:
        seed (int): Random seed. Default to 42.
        loss_func (Literal["mse", "mae"], optional): Loss function of optimizer. Default to "mse".
        max_iter (int): Maximum number of SLSQP iterations. Defaults to 2000.
        tolerance (float): SLSQP convergence tolerance. Defaults to 1e-8.
        epochs (int, optional): Deprecated alias for ``max_iter``.
        lr (float, optional): Deprecated alias for ``tolerance``.

    Returns:
        None
    """

    def __init__(
        self,
        seed: int = 42,
        loss_func: Literal["mse", "mae"] = "mse",
        epochs: int | None = None,
        lr: float | None = None,
        *,
        max_iter: int | None = None,
        tolerance: float | None = None,
    ) -> None:
        """
        Initialize the SynthControlModel.

        Args:
            seed (int): Random seed for reproducibility. Defaults to 42.
            loss_func (Literal["mse", "mae"]): Loss function for the optimizer. Defaults to "mse".
            epochs (int, optional): Deprecated alias for ``max_iter``.
            lr (float, optional): Deprecated alias for ``tolerance``.
            max_iter (int, optional): Maximum number of SLSQP iterations.
                Defaults to 2000.
            tolerance (float, optional): SLSQP convergence tolerance.
                Defaults to 1e-8.

        Returns:
            None
        """

        if epochs is not None and max_iter is not None:
            raise ValueError("Pass only one of 'max_iter' and deprecated 'epochs'.")
        if lr is not None and tolerance is not None:
            raise ValueError("Pass only one of 'tolerance' and deprecated 'lr'.")

        if epochs is not None:
            warnings.warn(
                "'epochs' is deprecated; use 'max_iter' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if lr is not None:
            warnings.warn(
                "'lr' is deprecated; use 'tolerance' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        resolved_max_iter = max_iter if max_iter is not None else epochs
        resolved_tolerance = tolerance if tolerance is not None else lr
        self.max_iter = 2000 if resolved_max_iter is None else int(resolved_max_iter)
        self.tolerance = (
            1e-8 if resolved_tolerance is None else float(resolved_tolerance)
        )
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be positive; got {self.max_iter}.")
        if not 0.0 < self.tolerance < 1.0:
            raise ValueError(
                f"tolerance should be strictly between 0 and 1; got {self.tolerance}."
            )

        self.seed = seed
        # Compatibility attributes for code that reads the old public fields.
        self.epochs = self.max_iter
        self.lr = self.tolerance
        self.__weights = None
        self.__loss_value = None
        possible_loss_functions = ("mse", "mae")
        self.loss_func = loss_func
        if loss_func not in possible_loss_functions:
            raise ValueError(
                f"loss_func should be one of {possible_loss_functions}; "
                f"got {loss_func!r}."
            )

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]
    ) -> None:
        """
        Fit the model to the given data.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Input features of shape (N, K) where N is the number of samples and K is the number of features.
            y (Union[np.ndarray, pd.DataFrame]): Target values of shape (N,).

        Returns:
            None
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError(f"X should be 2D array, got shape={X_arr.shape}.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y should have same row count, got {X_arr.shape[0]} and {y_arr.shape[0]}."
            )
        if X_arr.shape[1] == 0:
            raise ValueError("X should have at least one donor column.")

        rng = np.random.default_rng(self.seed)
        n_donors = X_arr.shape[1]
        init_w = rng.dirichlet(np.ones(n_donors))

        def objective(weights: np.ndarray) -> float:
            residuals = y_arr - X_arr.dot(weights)
            if self.loss_func == "mse":
                return float(np.mean(np.square(residuals)))
            return float(np.mean(np.abs(residuals)))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(n_donors)]
        result = minimize(
            objective,
            init_w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": self.tolerance},
        )
        if not result.success:
            raise ValueError(f"SLSQP optimization failed: {result.message}")

        weights = np.asarray(result.x, dtype=float)
        # Numerical safety projection to simplex.
        weights = np.clip(weights, 0.0, 1.0)
        total = float(weights.sum())
        if np.isclose(total, 0.0):
            raise ValueError("SLSQP produced near-zero weights sum.")
        weights = weights / total

        self.__weights = weights
        self.__loss_value = float(objective(weights))

    def get_weights(self) -> np.ndarray:
        """
        Get the weights of the model.

        Args:
            None

        Returns:
            np.ndarray: Array of weights of shape (K,) where K is the number of features.
        """
        weights = getattr(self, "_SynthControlModel__weights", None)
        if weights is None:
            raise ValueError("Fit model first with .fit() to obtain weights.")
        return weights

    def get_loss_value(self) -> float:
        """
        Get the loss value after fitting the model.

        Args:
            None

        Returns:
            float: Loss value of the model.
        """
        loss_value = getattr(self, "_SynthControlModel__loss_value", None)
        if loss_value is None:
            raise ValueError("Fit model first with .fit() to obtain loss value.")
        return loss_value
