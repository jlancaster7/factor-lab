"""
Factor Investing Laboratory

A comprehensive Python package for factor investing research, including:
- Data acquisition from multiple sources
- Factor calculation and analysis
- Portfolio optimization
- Backtesting framework
- Visualization tools
"""

__version__ = "0.1.0"
__author__ = "Josh Lancaster"

# Core imports
from .data import DataManager, DataProvider
from .factors import FactorCalculator, FactorLibrary
from .portfolio import PortfolioOptimizer, PortfolioAnalyzer
from .backtesting import Backtester, PerformanceAnalyzer
from .visualization import ChartManager, DashboardBuilder

__all__ = [
    "DataManager",
    "DataProvider",
    "FactorCalculator",
    "FactorLibrary",
    "PortfolioOptimizer",
    "PortfolioAnalyzer",
    "Backtester",
    "PerformanceAnalyzer",
    "ChartManager",
    "DashboardBuilder",
]
