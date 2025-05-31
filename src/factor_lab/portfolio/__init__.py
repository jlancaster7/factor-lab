"""
Portfolio optimization module for the Factor Investing Laboratory.

Implements various portfolio optimization techniques including:
- Mean-variance optimization
- Factor-based portfolio construction
- Risk parity approaches
- Black-Litterman model
- Robust optimization methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Callable
import logging
import warnings
from scipy.optimize import minimize
import cvxpy as cp
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.objective_functions import L2_reg
from pypfopt.exceptions import OptimizationError

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Main portfolio optimization class."""

    def __init__(
        self, returns_data: pd.DataFrame, factor_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize PortfolioOptimizer.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            Historical returns data with dates as index and assets as columns
        factor_data : Optional[pd.DataFrame]
            Factor exposure data for factor-based optimization
        """
        # Clean and validate returns data
        self.returns_data = returns_data.dropna()
        self.factor_data = factor_data
        self.universe = list(self.returns_data.columns)

        # Validate returns data
        self._validate_returns_data()

        # Calculate expected returns and covariance matrix with proper handling
        try:
            self.expected_returns = expected_returns.mean_historical_return(
                self.returns_data
            )

            # Ensure expected_returns is a pandas Series
            if not isinstance(self.expected_returns, pd.Series):
                if hasattr(self.expected_returns, "values"):
                    self.expected_returns = pd.Series(
                        self.expected_returns.values, index=self.universe
                    )
                else:
                    # Fallback to simple mean calculation
                    self.expected_returns = pd.Series(
                        self.returns_data.mean().values, index=self.universe
                    )

        except Exception as e:
            logger.warning(
                f"Error calculating expected returns with pypfopt: {e}. Using simple mean."
            )
            # Fallback to simple mean calculation
            self.expected_returns = pd.Series(
                self.returns_data.mean().values, index=self.universe
            )

        # Handle NaN values in expected returns
        if self.expected_returns.isna().any():
            logger.warning("Found NaN values in expected returns. Replacing with 0.")
            self.expected_returns = self.expected_returns.fillna(0)

        # Ensure expected returns are reasonable (not too extreme)
        if abs(self.expected_returns.mean()) > 1:
            logger.warning("Expected returns seem too large. Scaling down.")
            self.expected_returns = self.expected_returns / 100

        logger.info(
            f"Expected returns range: [{self.expected_returns.min():.6f}, {self.expected_returns.max():.6f}]"
        )

        # Use shrinkage covariance to improve conditioning
        try:
            self.cov_matrix = risk_models.CovarianceShrinkage(
                self.returns_data
            ).ledoit_wolf()
        except Exception as e:
            logger.warning(
                f"Shrinkage covariance failed: {e}. Using sample covariance."
            )
            self.cov_matrix = risk_models.sample_cov(self.returns_data)

        # Check and fix covariance matrix conditioning
        self._fix_covariance_matrix()

    def _validate_returns_data(self):
        """
        Validate and potentially fix returns data scaling.
        """
        # Check if returns are in percentage terms (e.g., 5.0 instead of 0.05)
        abs_mean = abs(self.returns_data.mean().mean())
        if abs_mean > 1:
            logger.warning(
                f"Returns appear to be in percentage terms (mean={abs_mean:.2f}). Converting to decimal."
            )
            self.returns_data = self.returns_data / 100

        # Check for extreme values
        abs_max = abs(self.returns_data).max().max()
        if abs_max > 1:
            logger.warning(
                f"Extreme return values detected (max={abs_max:.2f}). This may indicate data quality issues."
            )

        # Log data statistics
        logger.info(
            f"Returns data stats - Mean: {self.returns_data.mean().mean():.6f}, "
            f"Std: {self.returns_data.std().mean():.6f}, "
            f"Min: {self.returns_data.min().min():.6f}, "
            f"Max: {self.returns_data.max().max():.6f}"
        )

    def _fix_covariance_matrix(self):
        """
        Fix covariance matrix conditioning issues.
        """
        import numpy as np
        from scipy.linalg import LinAlgError

        try:
            # Check if matrix is positive definite
            np.linalg.cholesky(self.cov_matrix)
            logger.info("Covariance matrix is positive definite.")
        except LinAlgError:
            logger.warning(
                "Covariance matrix is not positive definite. Applying regularization."
            )

            # Get eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(self.cov_matrix)
            min_eigenval = np.min(eigenvals)

            if min_eigenval <= 0:
                # Add regularization to make positive definite
                regularization = abs(min_eigenval) + 1e-8
                self.cov_matrix += regularization * np.eye(len(self.cov_matrix))
                logger.info(f"Added regularization: {regularization}")

        # Check condition number
        eigenvals = np.linalg.eigvals(self.cov_matrix)
        condition_number = np.max(eigenvals) / np.min(eigenvals)

        logger.info(f"Covariance matrix condition number: {condition_number:.2e}")
        logger.info(
            f"Eigenvalue range: [{np.min(eigenvals):.2e}, {np.max(eigenvals):.2e}]"
        )

        # If condition number is too high, apply additional regularization
        if condition_number > 1e12:
            logger.warning(
                "Very high condition number. Applying additional regularization."
            )
            additional_reg = np.mean(np.diag(self.cov_matrix)) * 1e-6
            self.cov_matrix += additional_reg * np.eye(len(self.cov_matrix))

    def mean_variance_optimization(
        self,
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
        weight_bounds: Tuple[float, float] = (0, 1),
        sector_constraints: Optional[Dict] = None,
    ) -> Dict:
        """
        Perform mean-variance optimization with robust fallback mechanisms.

        Parameters:
        -----------
        target_return : Optional[float]
            Target portfolio return (annualized)
        target_volatility : Optional[float]
            Target portfolio volatility (annualized)
        weight_bounds : Tuple[float, float]
            Min and max weights for each asset
        sector_constraints : Optional[Dict]
            Sector exposure constraints

        Returns:
        --------
        Dict
            Optimization results including weights, expected return, volatility
        """
        try:
            # Validate inputs
            if len(self.returns_data) < 30:
                logger.warning(
                    f"Only {len(self.returns_data)} observations available. Results may be unreliable."
                )

            # Try multiple solvers in order of preference
            solvers_to_try = ["OSQP", "CLARABEL", "SCS", "ECOS"]

            for solver in solvers_to_try:
                try:
                    logger.info(
                        f"Attempting mean-variance optimization with {solver} solver"
                    )

                    ef = EfficientFrontier(
                        self.expected_returns,
                        self.cov_matrix,
                        weight_bounds=weight_bounds,
                        solver=solver,
                    )

                    # Add L2 regularization to prevent extreme weights
                    ef.add_objective(L2_reg, gamma=0.1)

                    # Perform optimization based on target
                    if target_return is not None:
                        weights = ef.efficient_return(target_return)
                    elif target_volatility is not None:
                        weights = ef.efficient_risk(target_volatility)
                    else:
                        weights = ef.max_sharpe()

                    cleaned_weights = ef.clean_weights()

                    # Calculate portfolio performance
                    portfolio_performance = ef.portfolio_performance(verbose=False)

                    logger.info(
                        f"Mean-variance optimization successful with {solver} solver"
                    )
                    return {
                        "weights": cleaned_weights,
                        "expected_return": portfolio_performance[0],
                        "volatility": portfolio_performance[1],
                        "sharpe_ratio": portfolio_performance[2],
                        "method": "mean_variance",
                        "solver": solver,
                    }

                except (OptimizationError, Exception) as e:
                    logger.warning(
                        f"Mean-variance optimization failed with {solver}: {e}"
                    )
                    continue

            # If all pypfopt solvers fail, try direct CVXPY implementation
            logger.info("Trying direct CVXPY implementation")
            return self._cvxpy_mean_variance_optimization(
                target_return, target_volatility, weight_bounds
            )

        except Exception as e:
            logger.error(f"All mean-variance optimization methods failed: {e}")
            return self._fallback_equal_weight_portfolio(
                "mean_variance_optimization", str(e)
            )

    def _cvxpy_mean_variance_optimization(
        self,
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
        weight_bounds: Tuple[float, float] = (0, 1),
    ) -> Dict:
        """
        Direct CVXPY implementation of mean-variance optimization.
        """
        try:
            n_assets = len(self.universe)
            w = cp.Variable(n_assets)

            # Convert to numpy arrays for CVXPY
            mu = self.expected_returns.values
            Sigma = self.cov_matrix.values

            # Portfolio return and risk
            portfolio_return = w.T @ mu
            portfolio_variance = cp.quad_form(w, Sigma)

            # Basic constraints
            constraints = [
                cp.sum(w) == 1,  # Weights sum to 1
                w >= weight_bounds[0],  # Lower bound
                w <= weight_bounds[1],  # Upper bound
            ]

            # Define objective based on targets
            if target_return is not None:
                # Minimize risk for target return
                constraints.append(portfolio_return >= target_return)
                objective = cp.Minimize(portfolio_variance)
            elif target_volatility is not None:
                # Maximize return for target risk
                constraints.append(cp.sqrt(portfolio_variance) <= target_volatility)
                objective = cp.Maximize(portfolio_return)
            else:
                # Maximize Sharpe ratio (approximated by maximizing return - risk_aversion * variance)
                risk_aversion = 1.0
                objective = cp.Maximize(
                    portfolio_return - 0.5 * risk_aversion * portfolio_variance
                )

            # Add regularization to prevent extreme weights
            reg_penalty = 0.01 * cp.sum_squares(w)
            if isinstance(objective.args[0], cp.expressions.expression.Expression):
                if "Minimize" in str(type(objective)):
                    objective = cp.Minimize(objective.args[0] + reg_penalty)
                else:
                    objective = cp.Maximize(objective.args[0] - reg_penalty)

            # Solve the problem
            problem = cp.Problem(objective, constraints)

            # Try different solvers
            cvxpy_solvers = ["CLARABEL", "SCS", "ECOS"]

            for solver in cvxpy_solvers:
                try:
                    if solver == "CLARABEL":
                        problem.solve(solver=cp.CLARABEL, verbose=False)
                    elif solver == "SCS":
                        problem.solve(solver=cp.SCS, verbose=False, max_iters=5000)
                    elif solver == "ECOS":
                        problem.solve(solver=cp.ECOS, verbose=False)

                    if problem.status == cp.OPTIMAL and w.value is not None:
                        weights_array = w.value
                        weights_dict = {
                            asset: max(0, float(weight))  # Ensure non-negative
                            for asset, weight in zip(self.universe, weights_array)
                        }

                        # Normalize weights to sum to 1
                        total_weight = sum(weights_dict.values())
                        if total_weight > 0:
                            weights_dict = {
                                k: v / total_weight for k, v in weights_dict.items()
                            }

                        # Calculate portfolio performance
                        portfolio_ret = np.dot(
                            list(weights_dict.values()), self.expected_returns
                        )
                        portfolio_vol = np.sqrt(
                            np.dot(
                                list(weights_dict.values()),
                                np.dot(self.cov_matrix, list(weights_dict.values())),
                            )
                        )
                        sharpe_ratio = (
                            portfolio_ret / portfolio_vol if portfolio_vol > 0 else 0
                        )

                        logger.info(
                            f"CVXPY mean-variance optimization successful with {solver}"
                        )
                        return {
                            "weights": weights_dict,
                            "expected_return": float(portfolio_ret),
                            "volatility": float(portfolio_vol),
                            "sharpe_ratio": float(sharpe_ratio),
                            "method": "mean_variance_cvxpy",
                            "solver": solver,
                        }

                except Exception as e:
                    logger.warning(f"CVXPY optimization failed with {solver}: {e}")
                    continue

            raise OptimizationError("All CVXPY solvers failed")

        except Exception as e:
            logger.error(f"CVXPY mean-variance optimization failed: {e}")
            return self._fallback_equal_weight_portfolio("cvxpy_mean_variance", str(e))

    def _fallback_equal_weight_portfolio(
        self, method_name: str, error_msg: str
    ) -> Dict:
        """
        Create an equal weight portfolio as fallback.
        """
        n_assets = len(self.universe)
        equal_weights = {asset: 1.0 / n_assets for asset in self.universe}

        # Calculate fallback performance
        portfolio_return = np.dot([1.0 / n_assets] * n_assets, self.expected_returns)
        portfolio_vol = np.sqrt(
            np.dot(
                [1.0 / n_assets] * n_assets,
                np.dot(self.cov_matrix, [1.0 / n_assets] * n_assets),
            )
        )
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

        return {
            "weights": equal_weights,
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_vol),
            "sharpe_ratio": float(sharpe_ratio),
            "method": "equal_weight_fallback",
            "original_method": method_name,
            "error": error_msg,
        }

    def risk_parity_optimization(self, method: str = "riskbudgeting") -> Dict:
        """
        Perform risk parity optimization.

        Parameters:
        -----------
        method : str
            Risk parity method ('riskbudgeting', 'equal_risk_contribution')

        Returns:
        --------
        Dict
            Optimization results
        """
        try:
            n_assets = len(self.universe)

            # Objective function: minimize sum of squared risk contributions
            def risk_parity_objective(weights, cov_matrix):
                weights = np.array(weights)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

                # Risk contributions
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                risk_contrib = weights * marginal_contrib

                # Target: equal risk contribution (1/n for each asset)
                target_risk_contrib = np.ones(n_assets) / n_assets

                # Minimize sum of squared deviations from target
                return np.sum(
                    (risk_contrib - target_risk_contrib * risk_contrib.sum()) ** 2
                )

            # Constraints
            constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]

            # Bounds
            bounds = [(0.0, 1.0) for _ in range(n_assets)]

            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets

            # Optimize
            result = minimize(
                fun=lambda w: risk_parity_objective(w, self.cov_matrix.values),
                x0=initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if result.success:
                weights_dict = {
                    asset: float(weight)
                    for asset, weight in zip(self.universe, result.x)
                }

                # Calculate portfolio performance
                portfolio_return = np.dot(result.x, self.expected_returns)
                portfolio_vol = np.sqrt(
                    np.dot(result.x.T, np.dot(self.cov_matrix, result.x))
                )
                sharpe_ratio = (
                    portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
                )

                return {
                    "weights": weights_dict,
                    "expected_return": float(portfolio_return),
                    "volatility": float(portfolio_vol),
                    "sharpe_ratio": float(sharpe_ratio),
                    "method": "risk_parity",
                }
            else:
                raise OptimizationError("Risk parity optimization failed")

        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            # Return equal weight portfolio as fallback
            n_assets = len(self.universe)
            equal_weights = {asset: 1.0 / n_assets for asset in self.universe}
            return {
                "weights": equal_weights,
                "method": "equal_weight_fallback",
                "error": str(e),
            }

    def factor_based_optimization(
        self,
        target_factor_exposures: Dict[str, float],
        factor_loadings: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Perform factor-based portfolio optimization.

        Parameters:
        -----------
        target_factor_exposures : Dict[str, float]
            Target exposures to factors
        factor_loadings : Optional[pd.DataFrame]
            Factor loadings matrix (assets x factors)

        Returns:
        --------
        Dict
            Optimization results
        """
        if factor_loadings is None and self.factor_data is None:
            raise ValueError("Factor loadings or factor data must be provided")

        try:
            n_assets = len(self.universe)

            # Create optimization variables
            w = cp.Variable(n_assets)

            # Objective: minimize portfolio variance
            portfolio_variance = cp.quad_form(w, self.cov_matrix.values)

            # Constraints
            constraints = [
                cp.sum(w) == 1,  # Weights sum to 1
                w >= 0,  # Long-only
                w <= 1,  # Maximum weight constraint
            ]

            # Factor exposure constraints
            if factor_loadings is not None:
                for factor_name, target_exposure in target_factor_exposures.items():
                    if factor_name in factor_loadings.columns:
                        factor_exposure = w @ factor_loadings[factor_name].values
                        constraints.append(factor_exposure == target_exposure)

            # Solve optimization problem
            prob = cp.Problem(cp.Minimize(portfolio_variance), constraints)
            prob.solve()

            if prob.status == cp.OPTIMAL:
                weights_dict = {
                    asset: float(weight)
                    for asset, weight in zip(self.universe, w.value)
                }

                # Calculate portfolio performance
                portfolio_return = np.dot(w.value, self.expected_returns)
                portfolio_vol = np.sqrt(portfolio_variance.value)
                sharpe_ratio = (
                    portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
                )

                return {
                    "weights": weights_dict,
                    "expected_return": float(portfolio_return),
                    "volatility": float(portfolio_vol),
                    "sharpe_ratio": float(sharpe_ratio),
                    "method": "factor_based",
                    "target_exposures": target_factor_exposures,
                }
            else:
                raise OptimizationError(
                    f"Factor-based optimization failed: {prob.status}"
                )

        except Exception as e:
            logger.error(f"Factor-based optimization failed: {e}")
            # Return equal weight portfolio as fallback
            n_assets = len(self.universe)
            equal_weights = {asset: 1.0 / n_assets for asset in self.universe}
            return {
                "weights": equal_weights,
                "method": "equal_weight_fallback",
                "error": str(e),
            }

    def minimum_variance_optimization(self) -> Dict:
        """
        Perform minimum variance optimization with robust fallback mechanisms.

        Returns:
        --------
        Dict
            Optimization results
        """
        try:
            # Try pypfopt first with multiple solvers
            solvers_to_try = ["OSQP", "CLARABEL", "SCS", "ECOS"]

            for solver in solvers_to_try:
                try:
                    logger.info(
                        f"Attempting minimum variance optimization with {solver} solver"
                    )

                    ef = EfficientFrontier(
                        self.expected_returns, self.cov_matrix, solver=solver
                    )
                    weights = ef.min_volatility()
                    cleaned_weights = ef.clean_weights()

                    portfolio_performance = ef.portfolio_performance(verbose=False)

                    logger.info(
                        f"Minimum variance optimization successful with {solver}"
                    )
                    return {
                        "weights": cleaned_weights,
                        "expected_return": portfolio_performance[0],
                        "volatility": portfolio_performance[1],
                        "sharpe_ratio": portfolio_performance[2],
                        "method": "minimum_variance",
                        "solver": solver,
                    }

                except (OptimizationError, Exception) as e:
                    logger.warning(
                        f"Minimum variance optimization failed with {solver}: {e}"
                    )
                    continue

            # If pypfopt fails, try direct CVXPY implementation
            logger.info("Trying direct CVXPY minimum variance implementation")
            return self._cvxpy_minimum_variance_optimization()

        except Exception as e:
            logger.error(f"All minimum variance optimization methods failed: {e}")
            return self._fallback_equal_weight_portfolio(
                "minimum_variance_optimization", str(e)
            )

    def _cvxpy_minimum_variance_optimization(self) -> Dict:
        """
        Direct CVXPY implementation of minimum variance optimization.
        """
        try:
            n_assets = len(self.universe)
            w = cp.Variable(n_assets)

            # Convert to numpy arrays for CVXPY
            Sigma = self.cov_matrix.values

            # Portfolio variance
            portfolio_variance = cp.quad_form(w, Sigma)

            # Constraints
            constraints = [
                cp.sum(w) == 1,  # Weights sum to 1
                w >= 0,  # Long-only
                w <= 1,  # Maximum weight constraint
            ]

            # Objective: minimize portfolio variance
            objective = cp.Minimize(portfolio_variance)

            # Add small regularization to prevent extreme weights
            reg_penalty = 0.001 * cp.sum_squares(w)
            objective = cp.Minimize(portfolio_variance + reg_penalty)

            # Solve the problem
            problem = cp.Problem(objective, constraints)

            # Try different CVXPY solvers
            cvxpy_solvers = ["CLARABEL", "SCS", "ECOS"]

            for solver in cvxpy_solvers:
                try:
                    if solver == "CLARABEL":
                        problem.solve(solver=cp.CLARABEL, verbose=False)
                    elif solver == "SCS":
                        problem.solve(solver=cp.SCS, verbose=False, max_iters=5000)
                    elif solver == "ECOS":
                        problem.solve(solver=cp.ECOS, verbose=False)

                    if problem.status == cp.OPTIMAL and w.value is not None:
                        weights_array = w.value
                        weights_dict = {
                            asset: max(0, float(weight))  # Ensure non-negative
                            for asset, weight in zip(self.universe, weights_array)
                        }

                        # Normalize weights to sum to 1
                        total_weight = sum(weights_dict.values())
                        if total_weight > 0:
                            weights_dict = {
                                k: v / total_weight for k, v in weights_dict.items()
                            }

                        # Calculate portfolio performance
                        portfolio_ret = np.dot(
                            list(weights_dict.values()), self.expected_returns
                        )
                        portfolio_vol = np.sqrt(
                            np.dot(
                                list(weights_dict.values()),
                                np.dot(self.cov_matrix, list(weights_dict.values())),
                            )
                        )
                        sharpe_ratio = (
                            portfolio_ret / portfolio_vol if portfolio_vol > 0 else 0
                        )

                        logger.info(
                            f"CVXPY minimum variance optimization successful with {solver}"
                        )
                        return {
                            "weights": weights_dict,
                            "expected_return": float(portfolio_ret),
                            "volatility": float(portfolio_vol),
                            "sharpe_ratio": float(sharpe_ratio),
                            "method": "minimum_variance_cvxpy",
                            "solver": solver,
                        }

                except Exception as e:
                    logger.warning(
                        f"CVXPY minimum variance optimization failed with {solver}: {e}"
                    )
                    continue

            raise OptimizationError("All CVXPY solvers failed for minimum variance")

        except Exception as e:
            logger.error(f"CVXPY minimum variance optimization failed: {e}")
            return self._fallback_equal_weight_portfolio(
                "cvxpy_minimum_variance", str(e)
            )

    def efficient_frontier(self, num_portfolios: int = 100) -> pd.DataFrame:
        """
        Generate efficient frontier.

        Parameters:
        -----------
        num_portfolios : int
            Number of portfolios on the frontier

        Returns:
        --------
        pd.DataFrame
            Efficient frontier data with returns, volatilities, and Sharpe ratios
        """
        try:
            # Get return range
            min_ret = self.expected_returns.min()
            max_ret = self.expected_returns.max()
            target_returns = np.linspace(min_ret, max_ret, num_portfolios)

            frontier_data = []

            for target_return in target_returns:
                try:
                    ef = EfficientFrontier(self.expected_returns, self.cov_matrix)
                    ef.efficient_return(target_return)
                    perf = ef.portfolio_performance(verbose=False)

                    frontier_data.append(
                        {
                            "Return": perf[0],
                            "Volatility": perf[1],
                            "Sharpe_Ratio": perf[2],
                        }
                    )
                except:
                    continue

            return pd.DataFrame(frontier_data)

        except Exception as e:
            logger.error(f"Error generating efficient frontier: {e}")
            return pd.DataFrame()


class PortfolioAnalyzer:
    """Portfolio analysis and performance evaluation."""

    def __init__(self):
        """Initialize PortfolioAnalyzer."""
        pass

    def calculate_portfolio_returns(
        self, weights: Dict[str, float], returns_data: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate portfolio returns given weights and asset returns.

        Parameters:
        -----------
        weights : Dict[str, float]
            Portfolio weights
        returns_data : pd.DataFrame
            Asset returns data

        Returns:
        --------
        pd.Series
            Portfolio returns time series
        """
        # Align weights with returns data
        aligned_weights = pd.Series(weights).reindex(returns_data.columns).fillna(0)

        # Calculate portfolio returns
        portfolio_returns = (returns_data * aligned_weights).sum(axis=1)

        return portfolio_returns

    def performance_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
    ) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        benchmark_returns : Optional[pd.Series]
            Benchmark returns for comparison
        risk_free_rate : float
            Risk-free rate (annualized)

        Returns:
        --------
        Dict
            Performance metrics
        """
        returns_clean = returns.dropna()

        # Basic metrics
        total_return = (1 + returns_clean).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns_clean)) - 1
        annualized_vol = returns_clean.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol

        # Risk metrics
        max_drawdown = (
            returns_clean.cumsum() - returns_clean.cumsum().expanding().max()
        ).min()
        var_95 = returns_clean.quantile(0.05)

        # Skewness and kurtosis
        skewness = returns_clean.skew()
        kurtosis = returns_clean.kurtosis()

        metrics = {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_vol,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "VaR (95%)": var_95,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
        }

        # Benchmark comparison metrics
        if benchmark_returns is not None:
            benchmark_clean = benchmark_returns.dropna()
            aligned_returns = returns_clean.align(benchmark_clean, join="inner")[0]
            aligned_benchmark = returns_clean.align(benchmark_clean, join="inner")[1]

            if len(aligned_returns) > 0:
                # Information ratio
                active_returns = aligned_returns - aligned_benchmark
                tracking_error = active_returns.std() * np.sqrt(252)
                information_ratio = (
                    active_returns.mean() * 252 / tracking_error
                    if tracking_error > 0
                    else 0
                )

                # Beta
                beta = aligned_returns.cov(aligned_benchmark) / aligned_benchmark.var()

                metrics.update(
                    {
                        "Information Ratio": information_ratio,
                        "Tracking Error": tracking_error,
                        "Beta": beta,
                    }
                )

        return metrics

    def factor_attribution(
        self, portfolio_returns: pd.Series, factor_returns: pd.DataFrame
    ) -> Dict:
        """
        Perform factor attribution analysis.

        Parameters:
        -----------
        portfolio_returns : pd.Series
            Portfolio returns
        factor_returns : pd.DataFrame
            Factor returns data

        Returns:
        --------
        Dict
            Factor attribution results
        """
        try:
            # Align data
            aligned_data = pd.concat(
                [portfolio_returns, factor_returns], axis=1, join="inner"
            )
            y = aligned_data.iloc[:, 0]  # Portfolio returns
            X = aligned_data.iloc[:, 1:]  # Factor returns

            # Add intercept
            X_with_intercept = np.column_stack([np.ones(len(X)), X])

            # OLS regression
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

            # Calculate R-squared
            y_pred = X_with_intercept @ coefficients
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Attribution results
            attribution = {
                "Alpha": coefficients[0],
                "R-squared": r_squared,
                "Factor Loadings": dict(zip(factor_returns.columns, coefficients[1:])),
            }

            return attribution

        except Exception as e:
            logger.error(f"Error in factor attribution: {e}")
            return {}

    def rolling_performance(
        self, returns: pd.Series, window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        window : int
            Rolling window size

        Returns:
        --------
        pd.DataFrame
            Rolling performance metrics
        """
        rolling_metrics = pd.DataFrame(index=returns.index)

        # Rolling return and volatility
        rolling_metrics["Rolling Return"] = returns.rolling(window).mean() * 252
        rolling_metrics["Rolling Volatility"] = returns.rolling(window).std() * np.sqrt(
            252
        )
        rolling_metrics["Rolling Sharpe"] = (
            rolling_metrics["Rolling Return"] / rolling_metrics["Rolling Volatility"]
        )

        # Rolling max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        rolling_metrics["Rolling Max Drawdown"] = (
            (cumulative / rolling_max - 1).rolling(window).min()
        )

        return rolling_metrics.dropna()
