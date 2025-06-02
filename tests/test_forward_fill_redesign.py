import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from src.factor_lab.data import FMPProvider


class TestForwardFillRedesign:
    @pytest.fixture
    def provider(self):
        return FMPProvider(api_key="test_key")
    
    @pytest.fixture
    def mock_quarterly_data(self):
        """Mock quarterly fundamental data with gaps"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='QE')
        data = []
        for date in dates:
            data.append({
                'symbol': 'AAPL',
                'acceptedDate': date.strftime('%Y-%m-%d'),
                'date': date.strftime('%Y-%m-%d'),
                'totalAssets': 100000 + np.random.randint(-5000, 5000),
                'totalStockholdersEquity': 50000 + np.random.randint(-2000, 2000),
                'netIncome': 5000 + np.random.randint(-500, 500),
                'totalDebt': 20000 + np.random.randint(-1000, 1000),
                'weightedAverageShsOut': 1000000,
            })
        return data
    
    @pytest.fixture
    def mock_price_data(self):
        """Mock daily price data"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')  # Business days
        prices = []
        base_price = 150.0
        for i, date in enumerate(dates):
            price = base_price + np.random.normal(0, 5)
            prices.append({
                'date': date.strftime('%Y-%m-%d'),
                'close': price,
                'symbol': 'AAPL'
            })
        return prices

    @patch('src.factor_lab.data.requests.get')
    def test_forward_fill_eliminates_gaps(self, mock_get, provider, mock_quarterly_data, mock_price_data):
        """Test that forward-fill eliminates gaps between quarterly reports"""
        mock_responses = [
            Mock(status_code=200, json=lambda: mock_price_data),      # Prices first
            Mock(status_code=200, json=lambda: mock_quarterly_data),  # Income statement
            Mock(status_code=200, json=lambda: []),                   # Balance sheet  
            Mock(status_code=200, json=lambda: []),                   # Cash flow
        ]
        mock_get.side_effect = mock_responses
        
        result = provider.get_fundamental_factors(
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31',
            frequency='daily'
        )
        
        factor_data = result['AAPL']
        
        # Should have no NaN values in ratios after forward-fill
        assert not factor_data['ROE'].isna().any(), "ROE should have no NaN values after forward-fill"
        assert not factor_data['debt_to_equity'].isna().any(), "Debt-to-equity should have no NaN values"
        
        # Should have data for most business days (allowing for some market holidays)
        expected_days = len(pd.bdate_range('2023-01-01', '2023-12-31'))
        actual_days = len(factor_data)
        assert actual_days >= expected_days * 0.95, f"Should have data for most business days, got {actual_days}/{expected_days}"

    @patch('src.factor_lab.data.requests.get')
    def test_forward_fill_preserves_quarterly_updates(self, mock_get, provider, mock_quarterly_data, mock_price_data):
        """Test that forward-fill properly updates when new quarterly data arrives"""
        mock_responses = [
            Mock(status_code=200, json=lambda: mock_price_data),
            Mock(status_code=200, json=lambda: mock_quarterly_data),
            Mock(status_code=200, json=lambda: []),
            Mock(status_code=200, json=lambda: []),
        ]
        mock_get.side_effect = mock_responses
        
        result = provider.get_fundamental_factors(
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31',
            frequency='daily'
        )
        
        factor_data = result['AAPL']
        
        # Find quarterly report dates
        quarterly_dates = pd.to_datetime(['2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31'])
        
        # Values should change at quarterly boundaries
        roe_values = factor_data['ROE'].values
        changes_detected = 0
        
        for i in range(1, len(factor_data)):
            if abs(roe_values[i] - roe_values[i-1]) > 0.001:  # Significant change
                changes_detected += 1
        
        # Should have at least some changes (when new quarterly data arrives)
        assert changes_detected > 0, "Should detect changes when new quarterly data arrives"

    @patch('src.factor_lab.data.requests.get')
    def test_different_calculation_frequencies(self, mock_get, provider, mock_quarterly_data, mock_price_data):
        """Test that different calculation frequencies work correctly"""
        mock_responses = [
            Mock(status_code=200, json=lambda: mock_price_data),
            Mock(status_code=200, json=lambda: mock_quarterly_data),
            Mock(status_code=200, json=lambda: []),
            Mock(status_code=200, json=lambda: []),
        ]
        mock_get.side_effect = mock_responses * 3  # Multiple calls for different frequencies
        
        frequencies = ['D', 'W', 'M']
        results = {}
        
        for freq in frequencies:
            result = provider.get_fundamental_factors(
                symbols='AAPL',
                start_date='2023-01-01',
                end_date='2023-12-31',
                frequency='daily',
                calculation_frequency=freq
            )
            results[freq] = result['AAPL']
        
        # All should have no NaN values
        for freq, data in results.items():
            assert not data['ROE'].isna().any(), f"Frequency {freq} should have no NaN ROE values"
            assert not data['debt_to_equity'].isna().any(), f"Frequency {freq} should have no NaN debt_to_equity values"
        
        # Daily should have most data points, monthly should have least
        assert len(results['D']) >= len(results['W']) >= len(results['M']), \
            "Daily should have most points, monthly least"

    @patch('src.factor_lab.data.requests.get')
    def test_trading_day_alignment(self, mock_get, provider, mock_quarterly_data, mock_price_data):
        """Test that fundamental factors align with trading days from price data"""
        mock_responses = [
            Mock(status_code=200, json=lambda: mock_price_data),
            Mock(status_code=200, json=lambda: mock_quarterly_data),
            Mock(status_code=200, json=lambda: []),
            Mock(status_code=200, json=lambda: []),
        ]
        mock_get.side_effect = mock_responses
        
        result = provider.get_fundamental_factors(
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31',
            frequency='daily'
        )
        
        factor_data = result['AAPL']
        
        # Check that index contains only business days (no weekends)
        dates = pd.to_datetime(factor_data.index)
        weekdays = dates.dayofweek
        
        # Should not have Saturday (5) or Sunday (6)
        assert not any(weekdays == 5), "Should not contain Saturdays"
        assert not any(weekdays == 6), "Should not contain Sundays"
        
        # Should be datetime index
        assert isinstance(factor_data.index, pd.DatetimeIndex), "Index should be DatetimeIndex"

    @patch('src.factor_lab.data.requests.get')
    def test_forward_fill_with_missing_quarters(self, mock_get, provider, mock_price_data):
        """Test forward-fill behavior when some quarterly data is missing"""
        # Create quarterly data with one missing quarter
        missing_quarter_data = [
            {
                'symbol': 'AAPL',
                'acceptedDate': '2023-03-31',
                'date': '2023-03-31',
                'totalAssets': 100000,
                'totalStockholdersEquity': 50000,
                'netIncome': 5000,
                'totalDebt': 20000,
                'weightedAverageShsOut': 1000000,
            },
            # Missing Q2 data
            {
                'symbol': 'AAPL',
                'acceptedDate': '2023-09-30',
                'date': '2023-09-30',
                'totalAssets': 105000,
                'totalStockholdersEquity': 52000,
                'netIncome': 5500,
                'totalDebt': 21000,
                'weightedAverageShsOut': 1000000,
            }
        ]
        
        mock_responses = [
            Mock(status_code=200, json=lambda: mock_price_data),
            Mock(status_code=200, json=lambda: missing_quarter_data),
            Mock(status_code=200, json=lambda: []),
            Mock(status_code=200, json=lambda: []),
        ]
        mock_get.side_effect = mock_responses
        
        result = provider.get_fundamental_factors(
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31',
            frequency='daily'
        )
        
        factor_data = result['AAPL']
        
        # Should still have no NaN values due to forward-fill
        assert not factor_data['ROE'].isna().any(), "Should handle missing quarters with forward-fill"
        
        # Values should be constant between Q1 and Q3 (missing Q2)
        q1_end = pd.to_datetime('2023-06-29')  # Before Q3
        q3_start = pd.to_datetime('2023-07-01')  # After Q1, before Q3
        
        q1_roe = factor_data.loc[factor_data.index <= q1_end, 'ROE'].iloc[-1]
        q3_roe = factor_data.loc[factor_data.index >= q3_start, 'ROE'].iloc[0]
        
        # Should be the same (Q1 value forward-filled until Q3)
        assert abs(q1_roe - q3_roe) < 0.001, "Q1 values should be forward-filled until Q3"

    @patch('src.factor_lab.data.requests.get')
    def test_forward_fill_date_range_coverage(self, mock_get, provider, mock_quarterly_data, mock_price_data):
        """Test that forward-fill covers the entire requested date range"""
        mock_responses = [
            Mock(status_code=200, json=lambda: mock_price_data),
            Mock(status_code=200, json=lambda: mock_quarterly_data),
            Mock(status_code=200, json=lambda: []),
            Mock(status_code=200, json=lambda: []),
        ]
        mock_get.side_effect = mock_responses
        
        start_date = '2023-01-15'  # Mid-month start
        end_date = '2023-11-15'   # Mid-month end
        
        result = provider.get_fundamental_factors(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date,
            frequency='daily'
        )
        
        factor_data = result['AAPL']
        
        # Should start on or after start_date and end on or before end_date
        first_date = factor_data.index[0]
        last_date = factor_data.index[-1]
        
        assert first_date >= pd.to_datetime(start_date), f"Data should start on/after {start_date}"
        assert last_date <= pd.to_datetime(end_date), f"Data should end on/before {end_date}"
        
        # Should have no gaps in the middle
        date_diff = (last_date - first_date).days
        expected_points = len(pd.bdate_range(first_date, last_date))
        actual_points = len(factor_data)
        
        # Allow for some market holidays
        assert actual_points >= expected_points * 0.9, \
            f"Should cover most of date range: {actual_points}/{expected_points} points"

    @patch('src.factor_lab.data.requests.get')
    def test_forward_fill_with_pe_pb_ratios(self, mock_get, provider, mock_quarterly_data, mock_price_data):
        """Test that PE/PB ratios are properly forward-filled (the original bug)"""
        mock_responses = [
            Mock(status_code=200, json=lambda: mock_price_data),
            Mock(status_code=200, json=lambda: mock_quarterly_data),
            Mock(status_code=200, json=lambda: []),
            Mock(status_code=200, json=lambda: []),
        ]
        mock_get.side_effect = mock_responses
        
        result = provider.get_fundamental_factors(
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31',
            frequency='daily'
        )
        
        factor_data = result['AAPL']
        
        # The original bug: PE/PB ratios showed as NaN after last quarterly report
        # These should now be forward-filled and have no NaN values
        assert 'PE' in factor_data.columns, "Should include PE ratio"
        assert 'PB' in factor_data.columns, "Should include PB ratio"
        
        # Critical test: no NaN values in PE/PB after forward-fill redesign
        pe_nans = factor_data['PE'].isna().sum()
        pb_nans = factor_data['PB'].isna().sum()
        
        assert pe_nans == 0, f"PE ratio should have no NaN values, found {pe_nans}"
        assert pb_nans == 0, f"PB ratio should have no NaN values, found {pb_nans}"
        
        # Values should be reasonable (positive for profitable companies)
        assert (factor_data['PE'] > 0).all(), "PE ratios should be positive"
        assert (factor_data['PB'] > 0).all(), "PB ratios should be positive"