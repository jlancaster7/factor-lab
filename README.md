# Factor Lab 🧪

A comprehensive Python framework for quantitative factor investing, portfolio optimization, and risk management. Built for researchers, portfolio managers, and quantitative analysts who need professional-grade tools for factor-based investment strategies.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20management-poetry-blue.svg)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

### 📊 Data Management

-   **Yahoo Finance integration**: Primary data provider with robust error handling and fallback mechanisms
-   **Unified data interface**: Consistent API for price and returns data with automatic data cleaning
-   **Automatic caching**: Efficient data storage and retrieval with data quality validation
-   **Data validation**: Built-in quality checks, outlier detection, and missing data handling

### 🔢 Factor Calculation

-   **Technical factors**: Momentum (1M, 3M, 6M, 12M), volatility, mean reversion, RSI, Bollinger Bands, MACD
-   **Fundamental factors**: P/E, P/B, ROE, Debt/Equity ratios (currently uses simulated data for demonstration)
-   **Statistical normalization**: Z-score, rank-based, and percentile normalization with cross-sectional analysis
-   **Factor risk model**: Correlation analysis and factor attribution with proper statistical interpretation
-   **Custom factors**: Easy framework for implementing new factors with validation

### 💼 Portfolio Optimization

-   **Robust optimization methods**: Mean-variance (with CVXPY), risk parity, minimum variance, equal weight
-   **Multiple solver support**: OSQP, CLARABEL, SCS, ECOS with automatic fallback mechanisms
-   **Advanced risk management**: Regularization, covariance shrinkage, constraint handling
-   **Risk budgeting**: Factor-based risk decomposition and attribution analysis

### 📈 Backtesting Framework

-   **Strategy simulation**: Momentum strategies and multi-factor strategy backtesting with monthly rebalancing
-   **Performance analytics**: Comprehensive metrics including Sharpe ratio, max drawdown, win rates
-   **Benchmark comparison**: Detailed analytics vs market benchmarks (SPY) with statistical significance
-   **Risk metrics**: Rolling performance analysis, drawdown analysis, and factor attribution

### 🛡️ Risk Management

-   **Dynamic hedging**: Correlation-based hedge ratios with tradeable ETFs (VIXY, VXX, TLT, GLD, UUP, SHY)
-   **Portfolio optimization**: Mean-variance, risk parity, minimum variance methods with robust solvers
-   **Factor attribution**: Risk decomposition with exposure analysis and performance attribution
-   **Concentration analysis**: Position sizing, diversification metrics, and risk budgeting

### 📊 Visualization & Analytics

-   **Interactive charts**: Plotly-based cumulative returns, drawdown analysis, rolling performance metrics
-   **Factor analysis**: Correlation heatmaps, factor exposure charts, and attribution analysis
-   **Performance reporting**: Equity curves, benchmark comparison, and risk-return scatter plots
-   **Export capabilities**: HTML output formats with professional styling for reports

## 🛠️ Installation

### Prerequisites

-   Python 3.11 or higher
-   Poetry (recommended) or pip

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/factor_lab.git
cd factor_lab

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/your-username/factor_lab.git
cd factor_lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration Setup

Before using Factor Lab, copy the example configuration file and customize it with your API keys:

```bash
# Copy the example configuration
cp config/environments.yaml.example config/environments.yaml

# Edit the configuration file with your API keys
# Most features work with Yahoo Finance (no API key required)
# For enhanced functionality, add your OpenBB Platform token
```

**Note**: The `config/environments.yaml` file is excluded from version control to protect your API keys.

## 🚀 Quick Start

### Basic Example

```python
from factor_lab.data import DataManager
from factor_lab.factors import FactorCalculator
from factor_lab.portfolio import PortfolioOptimizer
from factor_lab.backtesting import Backtester

# Initialize components
data_manager = DataManager(primary_provider="yahoo")
factor_calc = FactorCalculator(data_manager)
optimizer = PortfolioOptimizer()
backtester = Backtester(
    start_date="2022-01-01",
    end_date="2024-01-01",
    initial_capital=100000
)

# Define universe
universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Get price data
prices = data_manager.get_prices(
    symbols=universe,
    start_date='2022-01-01',
    end_date='2024-01-01'
)

# Calculate momentum factor
momentum_12m = factor_calc.momentum(prices, lookback=252)

# Run momentum strategy backtest
def momentum_strategy(factor_data, price_data, top_n=3, **kwargs):
    latest_factors = factor_data.iloc[-1]
    top_stocks = latest_factors.nlargest(top_n)
    weights = pd.Series(index=latest_factors.index, data=0.0)
    weights[top_stocks.index] = 1.0 / len(top_stocks)
    return weights

results = backtester.run_factor_strategy_backtest(
    price_data=prices,
    factor_data=momentum_12m,
    strategy_func=momentum_strategy,
    top_n=3
)

print(f"Annual Return: {results['performance_metrics']['Annualized Return']:.2%}")
print(f"Sharpe Ratio: {results['performance_metrics']['Sharpe Ratio']:.2f}")
```

### Jupyter Notebooks

Explore comprehensive examples in the `notebooks/` directory:

-   **[Getting Started](notebooks/getting_started.ipynb)**: Complete walkthrough of Factor Lab
-   **[Fundamental Factors](notebooks/fundamental_factors.ipynb)**: Combining technical and fundamental analysis (with simulated fundamental data)
-   **[Risk Management](notebooks/risk_management.ipynb)**: Advanced portfolio optimization and risk analysis

## ✅ Current Implementation Status

### Fully Implemented Features

-   ✅ **Data Management**: Yahoo Finance integration with caching and validation
-   ✅ **Technical Factors**: Momentum, volatility, mean reversion, RSI, Bollinger Bands, MACD
-   ✅ **Portfolio Optimization**: Mean-variance, risk parity, minimum variance with CVXPY solvers
-   ✅ **Backtesting**: Complete framework with momentum strategies and performance analytics
-   ✅ **Risk Management**: Dynamic hedging with tradeable ETFs, optimization methods
-   ✅ **Visualization**: Interactive Plotly charts for performance and factor analysis
-   ✅ **Results Export**: CSV export and HTML visualizations

### Demonstration/Simulated Features

-   🔄 **Fundamental Factors**: P/E, P/B, ROE, Debt/Equity (uses simulated data for demonstration)
-   🔄 **Multi-Factor Strategies**: Technical + fundamental integration (with simulated fundamentals)

### Planned Features

-   📋 **Real Fundamental Data**: OpenBB integration for live fundamental data
-   📋 **Style Factors**: Size, sector, country exposures
-   📋 **Alternative Data**: ESG, sentiment, alternative datasets
-   📋 **Advanced Analytics**: Machine learning models, stress testing

### CLI Examples

Run example strategies from the command line:

```bash
# Momentum strategy
python examples/momentum_strategy.py

# Multi-factor strategy
python examples/multi_factor_strategy.py
```

## 📋 Generated Results & Outputs

Factor Lab generates comprehensive results from its analysis:

### Results Files (`results/` directory)

-   `momentum_12m_factor.csv` - 12-month momentum factor scores across time
-   `factor_cross_section.csv` - Cross-sectional factor analysis
-   `momentum_strategy_portfolio_values.csv` - Portfolio value time series
-   `momentum_strategy_positions.csv` - Portfolio holdings over time
-   `performance_metrics.csv` - Comprehensive performance statistics
-   `strategy_comparison.csv` - Strategy vs benchmark comparison

### Interactive Visualizations (`data/figures/` directory)

-   `fundamental_strategy_equity_curve.html` - Strategy performance visualization
-   `factor_exposures.html` - Factor exposure analysis charts
-   `portfolio_optimization_comparison.html` - Optimization method comparison
-   `risk_contribution_breakdown.html` - Risk attribution analysis

## 📁 Project Structure

```
factor_lab/
├── src/factor_lab/           # Main package
│   ├── data/                 # Data acquisition and management
│   ├── factors/              # Factor calculation and utilities
│   ├── portfolio/            # Portfolio optimization and analysis
│   ├── backtesting/          # Backtesting framework and metrics
│   ├── visualization/        # Charts and dashboards
│   └── utils/                # Utilities and configuration
├── notebooks/                # Jupyter notebook examples
├── examples/                 # CLI example scripts
├── config/                   # Configuration files
├── data/                     # Data storage
│   ├── cache/                # Cached data files
│   ├── results/              # Backtest results
│   └── figures/              # Generated charts
├── tests/                    # Unit tests
└── docs/                     # Documentation
```

## 🔧 Configuration

Factor Lab uses YAML configuration files for settings:

### Configuration (`config/settings.yaml`)

```yaml
data:
    default_provider: yahoo
    storage:
        cache_dir: "./data/cache"
        results_dir: "./results"
    market:
        default_universe: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        benchmark: "SPY"

portfolio:
    optimization:
        default_method: "mean_variance"
        max_weight: 0.1
        target_volatility: 0.15

backtesting:
    costs:
        transaction_cost: 0.001
        bid_ask_spread: 0.0005
    rebalance_frequency: "monthly"
```

### Environment Configuration (`config/environments.yaml`)

```yaml
development:
    debug: true
    cache_enabled: true

production:
    debug: false
    cache_enabled: true

# API keys (set via environment variables - for future use)
api_keys:
    openbb:
        fmp: ${FMP_API_KEY}
        alpha_vantage: ${ALPHA_VANTAGE_API_KEY}
```

## 🧪 Testing & Verification

Factor Lab includes comprehensive testing to ensure reliability and help verify your setup.

### Quick Setup Verification

To verify your installation is working correctly:

```bash
# Verify setup and API connectivity
poetry run python verify_setup.py
```

This will test:

-   ✅ All package imports
-   ✅ Yahoo Finance connectivity
-   ✅ OpenBB Platform integration
-   ✅ Configuration loading
-   ✅ Basic factor calculations
-   ✅ Data manager functionality

### Unit Tests

Run the formal test suite:

```bash
# Run unit tests with pytest
poetry run python -m pytest tests/test_core.py -v
```

### Complete Test Suite

Run both verification and unit tests:

```bash
# Run everything
poetry run python run_tests.py
```

**Testing Structure:**

-   `tests/test_core.py` - Pytest-based unit tests for core functionality
-   `verify_setup.py` - Standalone setup verification and connectivity testing
-   `run_tests.py` - Combined test runner for both test types

## 📊 Supported Factors

### Technical Factors

-   **Momentum**: Price momentum over various periods (1M, 3M, 6M, 12M)
-   **Mean Reversion**: Short-term price reversal signals
-   **Volatility**: Historical volatility measures
-   **RSI**: Relative Strength Index
-   **Bollinger Bands**: Price position relative to Bollinger Bands
-   **MACD**: Moving Average Convergence Divergence

### Fundamental Factors

-   **Value**: P/E ratio, P/B ratio (currently simulated for demonstration purposes)
-   **Quality**: ROE, debt ratios (currently simulated for demonstration purposes)
-   **Growth**: Earnings growth, revenue growth (integration planned)
-   **Profitability**: Gross margins, operating margins (integration planned)

_Note: Real fundamental data integration with OpenBB or other providers is planned for future releases. Current implementation uses simulated data for testing and demonstration._

### Style Factors

-   **Size**: Market capitalization effects (planned for future implementation)
-   **Sector**: Industry and sector exposures (planned for future implementation)
-   **Country**: Geographic exposures (planned for future implementation)

_Note: Style factor integration is planned for future releases._

## 🧪 Example Strategies

### 1. Momentum Strategy

```python
def momentum_strategy(factor_data, price_data, top_n=5, **kwargs):
    """Simple momentum strategy using 12-month momentum factor."""
    try:
        # Get latest factor scores
        latest_factors = factor_data.iloc[-1]

        # Select top momentum stocks
        top_stocks = latest_factors.nlargest(top_n)

        # Equal weight the selected stocks
        weights = pd.Series(index=latest_factors.index, data=0.0)
        weights[top_stocks.index] = 1.0 / len(top_stocks)

        return weights
    except Exception as e:
        print(f"Error in momentum strategy: {e}")
        return None

# Run backtest with momentum strategy
momentum_12m = factor_calc.momentum(prices, lookback=252)
results = backtester.run_factor_strategy_backtest(
    price_data=prices,
    factor_data=momentum_12m,
    strategy_func=momentum_strategy,
    top_n=10
)
```

### 2. Multi-Factor Strategy

```python
def fundamental_momentum_strategy(date, composite_scores, n_positions=15):
    """Combined fundamental-technical strategy from notebooks."""
    if date not in composite_scores.index:
        return {}

    date_scores = composite_scores.loc[date].dropna()

    if len(date_scores) < n_positions:
        return {}

    # Select top scoring stocks
    top_stocks = date_scores.nlargest(n_positions)

    # Score-based weighting
    if len(top_stocks) > 0:
        normalized_scores = top_stocks - top_stocks.min() + 0.1
        weights_sum = normalized_scores.sum()

        weights = {}
        for symbol, score in normalized_scores.items():
            weights[symbol] = score / weights_sum

        return weights

    return {}

# Note: Requires simulated fundamental data as shown in fundamental_factors.ipynb
```

### 3. Risk Parity Strategy

```python
def risk_parity_strategy(returns_data):
    """Risk parity portfolio construction using implemented methods."""
    optimizer = PortfolioOptimizer(returns_data)

    result = optimizer.risk_parity_optimization()

    if 'weights' in result:
        return result['weights']
    else:
        return None

# Example usage with recent returns data
recent_returns = returns.tail(504)  # Last 2 years
optimizer = PortfolioOptimizer(recent_returns)
rp_result = optimizer.risk_parity_optimization()

print(f"Expected return: {rp_result['expected_return']:.4f}")
print(f"Volatility: {rp_result['volatility']:.4f}")
print(f"Sharpe ratio: {rp_result['sharpe_ratio']:.4f}")
```

## 📈 Performance Analytics

Factor Lab provides comprehensive performance analysis:

### Return Metrics

-   Total return, annualized return
-   Sharpe ratio, Sortino ratio, Calmar ratio
-   Maximum drawdown, average drawdown
-   Win rate, average win/loss

### Risk Metrics

-   Volatility, downside deviation
-   Value at Risk (VaR), Conditional VaR
-   Beta, tracking error, information ratio
-   Factor exposures and risk attribution

### Benchmark Comparison

-   Excess returns and alpha generation
-   Correlation and beta analysis
-   Performance attribution
-   Factor loading analysis

## 🧪 Testing

Run the test suite:

```bash
# Using Poetry
poetry run pytest

# Using pip
python -m pytest

# With coverage
poetry run pytest --cov=factor_lab
```

## 📚 Documentation

### API Documentation

Generate API documentation:

```bash
poetry run sphinx-build -b html docs docs/_build
```

### Examples and Tutorials

-   [Jupyter Notebooks](notebooks/) - Interactive tutorials
-   [Example Scripts](examples/) - Command-line examples
-   [Configuration Guide](docs/configuration.md) - Setup and configuration
-   [API Reference](docs/api/) - Detailed API documentation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/factor_lab.git
cd factor_lab

# Install development dependencies
poetry install --with dev

# Install pre-commit hooks
pre-commit install

# Run tests
poetry run pytest
```

### Code Style

-   Black for code formatting
-   flake8 for linting
-   isort for import sorting
-   Type hints using mypy

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

-   **Yahoo Finance** for reliable financial data access
-   **CVXPY** for convex optimization algorithms
-   **Plotly** for interactive visualizations
-   **NumPy/Pandas** for numerical computing and data analysis
-   **OpenBB Platform** for future fundamental data integration

## 📞 Support

-   **Documentation**: [https://factor-lab.readthedocs.io](https://factor-lab.readthedocs.io)
-   **Issues**: [GitHub Issues](https://github.com/your-username/factor_lab/issues)
-   **Discussions**: [GitHub Discussions](https://github.com/your-username/factor_lab/discussions)
-   **Email**: support@factor-lab.com

## 🗺️ Roadmap

### Upcoming Features

-   [ ] Real fundamental data integration with OpenBB
-   [ ] Machine learning factor models
-   [ ] Alternative data integration
-   [ ] Real-time portfolio monitoring
-   [ ] Advanced options strategies
-   [ ] ESG factor integration
-   [ ] Cryptocurrency factors
-   [ ] Style factors (size, sector, country)

### Version History

-   **v0.1.0** - Initial release with core functionality
-   **v0.2.0** - Added technical factors, portfolio optimization, and backtesting framework
-   **v0.3.0** - Enhanced risk management with tradeable ETF hedging and fundamental factor simulation

---

**Factor Lab** - Empowering quantitative investment research with professional-grade tools and methodologies.

_Built with ❤️ for the quantitative finance community_
