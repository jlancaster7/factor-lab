"""
Visualization module for the Factor Investing Laboratory.

Provides comprehensive charting and dashboard capabilities including:
- Performance charts
- Factor analysis plots
- Risk analysis visualizations
- Interactive dashboards
- Portfolio analytics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime
import warnings

# Set plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class ChartManager:
    """Main charting class for creating visualizations."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8), theme: str = "plotly"):
        """
        Initialize ChartManager.

        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size for matplotlib plots
        theme : str
            Plotting theme ('plotly', 'seaborn', 'classic')
        """
        self.figsize = figsize
        self.theme = theme
        self.colors = px.colors.qualitative.Set1

    def plot_cumulative_returns(
        self,
        returns_data: Union[pd.Series, pd.DataFrame],
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Cumulative Returns",
    ) -> go.Figure:
        """
        Plot cumulative returns chart.

        Parameters:
        -----------
        returns_data : Union[pd.Series, pd.DataFrame]
            Returns data to plot
        benchmark_returns : Optional[pd.Series]
            Benchmark returns for comparison
        title : str
            Chart title

        Returns:
        --------
        go.Figure
            Plotly figure object
        """
        fig = go.Figure()

        if isinstance(returns_data, pd.Series):
            cumulative = (1 + returns_data).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=cumulative.index,
                    y=cumulative.values,
                    mode="lines",
                    name="Portfolio",
                    line=dict(color=self.colors[0], width=2),
                )
            )
        else:
            for i, col in enumerate(returns_data.columns):
                cumulative = (1 + returns_data[col]).cumprod()
                fig.add_trace(
                    go.Scatter(
                        x=cumulative.index,
                        y=cumulative.values,
                        mode="lines",
                        name=col,
                        line=dict(color=self.colors[i % len(self.colors)], width=2),
                    )
                )

        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    mode="lines",
                    name="Benchmark",
                    line=dict(color="black", width=2, dash="dash"),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode="x unified",
            template="plotly_white",
        )

        return fig

    def plot_drawdown(
        self, returns_data: pd.Series, title: str = "Drawdown Analysis"
    ) -> go.Figure:
        """
        Plot drawdown chart.

        Parameters:
        -----------
        returns_data : pd.Series
            Returns data
        title : str
            Chart title

        Returns:
        --------
        go.Figure
            Plotly figure object
        """
        cumulative = (1 + returns_data).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max) - 1

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode="lines",
                fill="tonexty",
                name="Drawdown",
                line=dict(color="red", width=1),
                fillcolor="rgba(255, 0, 0, 0.3)",
            )
        )

        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            template="plotly_white",
        )

        return fig

    def plot_rolling_metrics(
        self,
        returns_data: pd.Series,
        window: int = 252,
        title: str = "Rolling Performance Metrics",
    ) -> go.Figure:
        """
        Plot rolling performance metrics.

        Parameters:
        -----------
        returns_data : pd.Series
            Returns data
        window : int
            Rolling window size
        title : str
            Chart title

        Returns:
        --------
        go.Figure
            Plotly figure object
        """
        # Calculate rolling metrics
        rolling_return = returns_data.rolling(window).mean() * 252
        rolling_vol = returns_data.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Rolling Return",
                "Rolling Volatility",
                "Rolling Sharpe Ratio",
            ),
            vertical_spacing=0.08,
        )

        # Rolling return
        fig.add_trace(
            go.Scatter(
                x=rolling_return.index,
                y=rolling_return.values * 100,
                mode="lines",
                name="Return",
                line=dict(color=self.colors[0]),
            ),
            row=1,
            col=1,
        )

        # Rolling volatility
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values * 100,
                mode="lines",
                name="Volatility",
                line=dict(color=self.colors[1]),
            ),
            row=2,
            col=1,
        )

        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode="lines",
                name="Sharpe Ratio",
                line=dict(color=self.colors[2]),
            ),
            row=3,
            col=1,
        )

        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)

        fig.update_layout(
            title=title, height=800, showlegend=False, template="plotly_white"
        )

        return fig

    def plot_factor_heatmap(
        self, factor_data: pd.DataFrame, title: str = "Factor Correlation Heatmap"
    ) -> go.Figure:
        """
        Plot factor correlation heatmap.

        Parameters:
        -----------
        factor_data : pd.DataFrame
            Factor data
        title : str
            Chart title

        Returns:
        --------
        go.Figure
            Plotly figure object
        """
        correlation_matrix = factor_data.corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale="RdBu",
                zmid=0,
                text=correlation_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Factors",
            yaxis_title="Factors",
            template="plotly_white",
        )

        return fig

    def plot_factor_returns(
        self, factor_data: pd.DataFrame, returns_data: pd.DataFrame, factor_name: str
    ) -> go.Figure:
        """
        Plot factor returns scatter plot.

        Parameters:
        -----------
        factor_data : pd.DataFrame
            Factor scores
        returns_data : pd.DataFrame
            Asset returns
        factor_name : str
            Name of factor to plot

        Returns:
        --------
        go.Figure
            Plotly figure object
        """
        if factor_name not in factor_data.columns:
            raise ValueError(f"Factor {factor_name} not found in factor data")

        # Get common assets and dates
        common_assets = factor_data.index.intersection(returns_data.columns)
        common_dates = factor_data.columns.intersection(returns_data.index)

        if len(common_assets) == 0 or len(common_dates) == 0:
            raise ValueError(
                "No common assets or dates between factor and returns data"
            )

        # Prepare data for scatter plot
        factor_scores = []
        asset_returns = []

        for date in common_dates[-252:]:  # Last year of data
            for asset in common_assets:
                if pd.notna(factor_data.loc[asset, date]) and pd.notna(
                    returns_data.loc[date, asset]
                ):
                    factor_scores.append(factor_data.loc[asset, date])
                    asset_returns.append(returns_data.loc[date, asset] * 100)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=factor_scores,
                y=asset_returns,
                mode="markers",
                marker=dict(size=4, opacity=0.6, color=self.colors[0]),
                name="Asset Returns",
            )
        )

        # Add trend line
        if len(factor_scores) > 0:
            z = np.polyfit(factor_scores, asset_returns, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(factor_scores), max(factor_scores), 100)
            y_trend = p(x_trend)

            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode="lines",
                    line=dict(color="red", width=2),
                    name="Trend Line",
                )
            )

        fig.update_layout(
            title=f"Factor Returns: {factor_name}",
            xaxis_title=f"{factor_name} Score",
            yaxis_title="Forward Returns (%)",
            template="plotly_white",
        )

        return fig

    def plot_efficient_frontier(
        self,
        frontier_data: pd.DataFrame,
        optimal_portfolio: Optional[Dict] = None,
        title: str = "Efficient Frontier",
    ) -> go.Figure:
        """
        Plot efficient frontier.

        Parameters:
        -----------
        frontier_data : pd.DataFrame
            Frontier data with Return, Volatility, Sharpe_Ratio columns
        optimal_portfolio : Optional[Dict]
            Optimal portfolio data for highlighting
        title : str
            Chart title

        Returns:
        --------
        go.Figure
            Plotly figure object
        """
        fig = go.Figure()

        # Efficient frontier
        fig.add_trace(
            go.Scatter(
                x=frontier_data["Volatility"] * 100,
                y=frontier_data["Return"] * 100,
                mode="lines",
                name="Efficient Frontier",
                line=dict(color=self.colors[0], width=3),
            )
        )

        # Color points by Sharpe ratio
        fig.add_trace(
            go.Scatter(
                x=frontier_data["Volatility"] * 100,
                y=frontier_data["Return"] * 100,
                mode="markers",
                marker=dict(
                    size=8,
                    color=frontier_data["Sharpe_Ratio"],
                    colorscale="Viridis",
                    colorbar=dict(title="Sharpe Ratio"),
                    opacity=0.8,
                ),
                name="Portfolios",
                hovertemplate="Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>",
            )
        )

        # Highlight optimal portfolio
        if optimal_portfolio:
            fig.add_trace(
                go.Scatter(
                    x=[optimal_portfolio["volatility"] * 100],
                    y=[optimal_portfolio["expected_return"] * 100],
                    mode="markers",
                    marker=dict(size=15, color="red", symbol="star"),
                    name="Optimal Portfolio",
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Volatility (%)",
            yaxis_title="Expected Return (%)",
            template="plotly_white",
        )

        return fig

    def plot_portfolio_composition(
        self, weights: Dict[str, float], title: str = "Portfolio Composition"
    ) -> go.Figure:
        """
        Plot portfolio composition pie chart.

        Parameters:
        -----------
        weights : Dict[str, float]
            Portfolio weights
        title : str
            Chart title

        Returns:
        --------
        go.Figure
            Plotly figure object
        """
        # Filter out zero weights
        filtered_weights = {k: v for k, v in weights.items() if abs(v) > 0.001}

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(filtered_weights.keys()),
                    values=list(filtered_weights.values()),
                    hole=0.3,
                    textinfo="label+percent",
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(title=title, template="plotly_white")

        return fig

    def plot_performance_comparison(
        self,
        performance_data: pd.DataFrame,
        title: str = "Strategy Performance Comparison",
    ) -> go.Figure:
        """
        Plot performance comparison bar chart.

        Parameters:
        -----------
        performance_data : pd.DataFrame
            Performance metrics by strategy
        title : str
            Chart title

        Returns:
        --------
        go.Figure
            Plotly figure object
        """
        metrics = [
            "Annualized Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Max Drawdown",
        ]
        available_metrics = [m for m in metrics if m in performance_data.columns]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=available_metrics[:4],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for i, metric in enumerate(available_metrics[:4]):
            row, col = positions[i]

            fig.add_trace(
                go.Bar(
                    x=performance_data.index,
                    y=performance_data[metric]
                    * (
                        100
                        if "Return" in metric
                        or "Volatility" in metric
                        or "Drawdown" in metric
                        else 1
                    ),
                    name=metric,
                    marker_color=self.colors[i % len(self.colors)],
                ),
                row=row,
                col=col,
            )

        fig.update_layout(
            title=title, height=600, showlegend=False, template="plotly_white"
        )

        return fig


class DashboardBuilder:
    """Build interactive dashboards for factor analysis."""

    def __init__(self):
        """Initialize DashboardBuilder."""
        self.chart_manager = ChartManager()

    def create_performance_dashboard(
        self, backtest_results: Dict
    ) -> Dict[str, go.Figure]:
        """
        Create comprehensive performance dashboard.

        Parameters:
        -----------
        backtest_results : Dict
            Backtest results dictionary

        Returns:
        --------
        Dict[str, go.Figure]
            Dictionary of dashboard charts
        """
        charts = {}

        if "portfolio_values" in backtest_results:
            portfolio_values = backtest_results["portfolio_values"]
            returns = portfolio_values.pct_change().dropna()

            # Cumulative returns
            charts["cumulative_returns"] = self.chart_manager.plot_cumulative_returns(
                returns, title="Portfolio Cumulative Returns"
            )

            # Drawdown
            charts["drawdown"] = self.chart_manager.plot_drawdown(
                returns, title="Portfolio Drawdown"
            )

            # Rolling metrics
            charts["rolling_metrics"] = self.chart_manager.plot_rolling_metrics(
                returns, title="Rolling Performance Metrics"
            )

        if "positions" in backtest_results:
            positions = backtest_results["positions"]

            # Portfolio composition over time (latest)
            if len(positions) > 0:
                latest_weights = positions.iloc[-1].drop(
                    "Portfolio_Value", errors="ignore"
                )
                latest_weights = latest_weights[latest_weights.notna()]
                charts["composition"] = self.chart_manager.plot_portfolio_composition(
                    latest_weights.to_dict(), title="Latest Portfolio Composition"
                )

        return charts

    def create_factor_analysis_dashboard(
        self, factor_data: pd.DataFrame, returns_data: pd.DataFrame
    ) -> Dict[str, go.Figure]:
        """
        Create factor analysis dashboard.

        Parameters:
        -----------
        factor_data : pd.DataFrame
            Factor scores data
        returns_data : pd.DataFrame
            Returns data

        Returns:
        --------
        Dict[str, go.Figure]
            Dictionary of dashboard charts
        """
        charts = {}

        # Factor correlation heatmap
        charts["correlation_heatmap"] = self.chart_manager.plot_factor_heatmap(
            factor_data.T, title="Factor Correlation Matrix"
        )

        # Factor returns scatter plots for key factors
        key_factors = factor_data.columns[:3]  # First 3 factors

        for i, factor in enumerate(key_factors):
            try:
                charts[f"factor_returns_{factor}"] = (
                    self.chart_manager.plot_factor_returns(
                        factor_data.T, returns_data, factor
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Could not create factor returns plot for {factor}: {e}"
                )

        return charts

    def export_dashboard_html(
        self, charts: Dict[str, go.Figure], filename: str = "factor_dashboard.html"
    ) -> str:
        """
        Export dashboard charts to HTML file.

        Parameters:
        -----------
        charts : Dict[str, go.Figure]
            Dictionary of charts
        filename : str
            Output filename

        Returns:
        --------
        str
            Path to created HTML file
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Factor Investing Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .chart-container { margin: 20px 0; }
                h1, h2 { color: #333; }
            </style>
        </head>
        <body>
            <h1>Factor Investing Laboratory Dashboard</h1>
        """

        for chart_name, fig in charts.items():
            chart_html = fig.to_html(include_plotlyjs=False, div_id=chart_name)
            html_content += f"""
            <div class="chart-container">
                <h2>{chart_name.replace('_', ' ').title()}</h2>
                {chart_html}
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        with open(filename, "w") as f:
            f.write(html_content)

        return filename
