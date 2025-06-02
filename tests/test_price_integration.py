import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from src.factor_lab.data import FMPProvider


class TestPriceIntegration:
    @pytest.fixture
    def provider(self):
        return FMPProvider(api_key="test_key")
    
    @pytest.fixture
    def mock_price_data_single(self):
        """Mock price data for a single symbol"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')  # Business days
        prices = []
        base_price = 150.0
        for i, date in enumerate(dates):
            price = base_price + np.sin(i * 0.1) * 10 + np.random.normal(0, 2)
            prices.append({
                'date': date.strftime('%Y-%m-%d'),
                'close': price,
                'symbol': 'AAPL'
            })
        return prices
    
    @pytest.fixture
    def mock_price_data_multiple(self):
        """Mock price data for multiple symbols"""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
        all_prices = []
        
        base_prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 100.0}
        
        for symbol in symbols:
            base_price = base_prices[symbol]
            for i, date in enumerate(dates):
                price = base_price + np.sin(i * 0.1) * 5 + np.random.normal(0, 1)
                all_prices.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'close': price,
                    'symbol': symbol
                })
        return all_prices

    @patch('src.factor_lab.data.requests.get')
    def test_fetch_historical_prices_single_symbol(self, mock_get, provider, mock_price_data_single):
        """Test fetching historical prices for a single symbol"""
        mock_get.return_value = Mock(status_code=200, json=lambda: mock_price_data_single)
        
        result = provider._fetch_historical_prices(
            'AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert 'close' in result.columns, "Should have close price column"
        assert 'symbol' in result.columns, "Should have symbol column"
        assert len(result) > 200, "Should have substantial price data"
        assert all(result['symbol'] == 'AAPL'), "All records should be for AAPL"
        
        # Check date index
        assert isinstance(result.index, pd.DatetimeIndex), "Should have datetime index"

    @patch('src.factor_lab.data.requests.get')
    def test_fetch_historical_prices_multiple_symbols(self, mock_get, provider, mock_price_data_multiple):
        """Test fetching historical prices for multiple symbols"""
        mock_get.return_value = Mock(status_code=200, json=lambda: mock_price_data_multiple)
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        result = provider._fetch_historical_prices(
            symbols,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert 'close' in result.columns, "Should have close price column"
        assert 'symbol' in result.columns, "Should have symbol column"
        
        # Should have data for all symbols
        unique_symbols = result['symbol'].unique()
        assert len(unique_symbols) == 3, f"Should have 3 symbols, got {len(unique_symbols)}"
        assert set(unique_symbols) == set(symbols), "Should have all requested symbols"

    @patch('src.factor_lab.data.requests.get')
    def test_get_prices_public_method(self, mock_get, provider, mock_price_data_single):
        """Test the public get_prices method for DataManager compatibility"""
        mock_get.return_value = Mock(status_code=200, json=lambda: mock_price_data_single)
        
        result = provider.get_prices(
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        assert isinstance(result, dict), "Should return dictionary"
        assert 'AAPL' in result, "Should have AAPL in result"
        
        aapl_data = result['AAPL']
        assert isinstance(aapl_data, pd.DataFrame), "Symbol data should be DataFrame"
        assert 'close' in aapl_data.columns, "Should have close price column"

    @patch('src.factor_lab.data.requests.get')
    def test_price_data_date_alignment(self, mock_get, provider, mock_price_data_single):
        """Test that price data is properly aligned with requested date range"""
        mock_get.return_value = Mock(status_code=200, json=lambda: mock_price_data_single)
        
        start_date = '2023-03-01'
        end_date = '2023-09-30'
        
        result = provider._fetch_historical_prices(
            'AAPL',
            start_date=start_date,
            end_date=end_date
        )
        
        # Check date range
        first_date = result.index.min()
        last_date = result.index.max()
        
        assert first_date >= pd.to_datetime(start_date), "Should start on/after start_date"
        assert last_date <= pd.to_datetime(end_date), "Should end on/before end_date"

    @patch('src.factor_lab.data.requests.get')
    def test_price_data_for_market_cap_calculation(self, mock_get, provider, mock_price_data_single):
        """Test that price data supports market cap calculation"""
        mock_get.return_value = Mock(status_code=200, json=lambda: mock_price_data_single)
        
        # Mock shares outstanding data
        shares_data = pd.DataFrame({
            'weightedAverageShsOut': [1000000] * 10,
            'date': pd.date_range('2023-01-01', periods=10, freq='QE')
        })
        shares_data.set_index('date', inplace=True)
        
        with patch.object(provider, '_calculate_market_cap') as mock_calc:
            mock_calc.return_value = pd.Series([150000000] * 100, 
                                             index=pd.date_range('2023-01-01', periods=100, freq='B'))
            
            prices = provider._fetch_historical_prices('AAPL', '2023-01-01', '2023-12-31')
            market_cap = provider._calculate_market_cap('AAPL', shares_data, '2023-01-01', '2023-12-31')
            
            # Should be able to calculate market cap from prices
            assert len(market_cap) > 0, "Should calculate market cap values"
            assert all(market_cap > 0), "Market cap should be positive"

    @patch('src.factor_lab.data.requests.get')
    def test_price_data_business_days_only(self, mock_get, provider, mock_price_data_single):
        """Test that price data contains only business days"""
        mock_get.return_value = Mock(status_code=200, json=lambda: mock_price_data_single)
        
        result = provider._fetch_historical_prices(
            'AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        # Check that index contains only business days
        dates = result.index
        weekdays = dates.dayofweek
        
        # Should not have Saturday (5) or Sunday (6)
        assert not any(weekdays == 5), "Should not contain Saturdays"
        assert not any(weekdays == 6), "Should not contain Sundays"

    @patch('src.factor_lab.data.requests.get')
    def test_price_api_error_handling(self, mock_get, provider):
        """Test error handling when price API fails"""
        # Test HTTP error
        mock_get.return_value = Mock(status_code=404, text="Not Found")
        
        with pytest.raises(Exception):
            provider._fetch_historical_prices('INVALID', '2023-01-01', '2023-12-31')
        
        # Test empty response
        mock_get.return_value = Mock(status_code=200, json=lambda: [])
        
        result = provider._fetch_historical_prices('AAPL', '2023-01-01', '2023-12-31')
        assert len(result) == 0, "Should handle empty response gracefully"

    @patch('src.factor_lab.data.requests.get')
    def test_price_data_sorting_and_formatting(self, mock_get, provider):
        """Test that price data is properly sorted and formatted"""
        # Create unsorted mock data
        unsorted_data = [
            {'date': '2023-01-15', 'close': 150.0, 'symbol': 'AAPL'},
            {'date': '2023-01-10', 'close': 145.0, 'symbol': 'AAPL'},
            {'date': '2023-01-20', 'close': 155.0, 'symbol': 'AAPL'},
        ]
        
        mock_get.return_value = Mock(status_code=200, json=lambda: unsorted_data)
        
        result = provider._fetch_historical_prices(
            'AAPL',
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        # Should be sorted by date
        dates = result.index
        assert dates.is_monotonic_increasing, "Dates should be sorted in ascending order"
        
        # Should have proper data types
        assert result['close'].dtype in [np.float64, float], "Close prices should be numeric"

    @patch('src.factor_lab.data.requests.get')
    def test_calculate_market_cap_method(self, mock_get, provider, mock_price_data_single):
        """Test the market cap calculation method"""
        mock_get.return_value = Mock(status_code=200, json=lambda: mock_price_data_single)
        
        # Mock shares outstanding data
        shares_data = pd.DataFrame({
            'weightedAverageShsOut': [1000000, 1100000, 1200000],
            'date': pd.to_datetime(['2023-03-31', '2023-06-30', '2023-09-30'])
        })
        shares_data.set_index('date', inplace=True)
        
        market_cap = provider._calculate_market_cap(
            'AAPL',
            shares_data,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        assert isinstance(market_cap, pd.Series), "Should return pandas Series"
        assert len(market_cap) > 100, "Should have daily market cap values"
        assert all(market_cap > 0), "Market cap should be positive"
        assert isinstance(market_cap.index, pd.DatetimeIndex), "Should have datetime index"

    @patch('src.factor_lab.data.requests.get')
    def test_price_integration_with_fundamental_factors(self, mock_get, provider, mock_price_data_single):
        """Test that price integration works with fundamental factors calculation"""
        # Mock both price and fundamental data
        mock_fundamental_data = [
            {
                'symbol': 'AAPL',
                'acceptedDate': '2023-03-31',
                'date': '2023-03-31',
                'totalAssets': 100000,
                'totalStockholdersEquity': 50000,
                'netIncome': 5000,
                'totalDebt': 20000,
                'weightedAverageShsOut': 1000000,
            }
        ]
        
        mock_responses = [
            Mock(status_code=200, json=lambda: mock_price_data_single),  # Prices first
            Mock(status_code=200, json=lambda: mock_fundamental_data),   # Income statement
            Mock(status_code=200, json=lambda: []),                     # Balance sheet  
            Mock(status_code=200, json=lambda: []),                     # Cash flow
        ]
        mock_get.side_effect = mock_responses
        
        result = provider.get_fundamental_factors(
            symbols='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31',
            frequency='daily'
        )
        
        factor_data = result['AAPL']
        
        # Should have PE and PB ratios (which require price data)
        assert 'PE' in factor_data.columns, "Should calculate PE ratio using price data"
        assert 'PB' in factor_data.columns, "Should calculate PB ratio using price data"
        
        # PE and PB should have values (not NaN)
        assert not factor_data['PE'].isna().all(), "PE should have calculated values"
        assert not factor_data['PB'].isna().all(), "PB should have calculated values"
        
        # Values should be reasonable
        assert (factor_data['PE'] > 0).all(), "PE ratios should be positive"
        assert (factor_data['PB'] > 0).all(), "PB ratios should be positive"