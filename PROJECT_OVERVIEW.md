# Factor Lab Project - AI Agent Handoff Documentation

**Last Updated**: June 1, 2025  
**Project Status**: Epic 1 (Core FMP Infrastructure) Complete - Ready for Epic 2 (Time-Aware Data Processing)

## 🎯 **MISSION CRITICAL CONTEXT**

This is a quantitative finance Factor Lab project focused on implementing **real fundamental data integration** using the Financial Modeling Prep (FMP) API. The core objective is to replace simulated fundamental data with real data while preventing look-ahead bias and maintaining performance.

### **IMMEDIATE NEXT TASK**: 
**Epic 2, Story 2.1** - Implement accepted date handling for look-ahead bias prevention in the FMPProvider class.

---

## 📊 **PROJECT ARCHITECTURE**

### **Core Structure**
```
factor-lab/
├── src/factor_lab/
│   ├── data/__init__.py          # 🔥 MAIN WORK FILE - FMPProvider class
│   ├── factors/                  # Factor calculation modules
│   ├── portfolio/                # Portfolio management
│   ├── backtesting/              # Strategy backtesting
│   ├── visualization/            # Plotting and charts
│   └── utils/                    # Utility functions
├── notebooks/
│   └── fundamental_factors.ipynb # 🎯 TARGET INTEGRATION FILE
├── tests/                        # Comprehensive test suite
├── config/environments.yaml      # API keys and configuration
└── data/                         # Data storage and cache
```

### **Key Dependencies**
- **Python 3.11+** (verified in pyproject.toml)
- **Poetry** for dependency management
- **pandas, numpy** for data manipulation
- **requests** for HTTP API calls
- **pytest** for testing
- **jupyter** for notebook environment

---

## 🔑 **CRITICAL FILES AND THEIR STATES**

### **1. `/src/factor_lab/data/__init__.py` - MAIN WORK FILE**
**Status**: Recently modified with Epic 1 complete
**Contains**:
- `FMPProvider` class inheriting from `DataProvider`
- Complete API integration with rate limiting (750 calls/minute)
- All 4 fundamental data endpoints implemented and tested
- Data validation and quality scoring systems
- Updated `DataManager` to include FMP provider

**Key Methods Implemented**:
```python
# Authentication & HTTP
_get_api_key() → str                           # ✅ Smart config loading
_make_request(url) → Optional[Dict]            # ✅ Robust HTTP with retries
_enforce_rate_limit() → None                   # ✅ 750 calls/min enforcement

# Data Fetching (All tested with real AAPL data)
_fetch_income_statement(symbol, limit) → Optional[List[Dict]]    # ✅
_fetch_balance_sheet(symbol, limit) → Optional[List[Dict]]       # ✅
_fetch_cash_flow(symbol, limit) → Optional[List[Dict]]           # ✅
_fetch_financial_ratios(symbol, limit) → Optional[List[Dict]]    # ✅

# Data Validation & Quality
_validate_financial_data(data, data_type) → List[Dict]          # ✅
_parse_date_safely(date_str) → Optional[datetime.date]          # ✅
_perform_data_quality_checks(data) → Tuple[float, Dict]         # ✅
```

**🚨 MISSING METHODS** (Epic 2 focus):
```python
# TODO: Story 2.1 - Look-ahead bias prevention
_filter_by_accepted_date(data, as_of_date) → List[Dict]

# TODO: Story 2.2 - Trailing 12-month calculations  
_get_trailing_12m_data(statements, as_of_date) → Dict

# TODO: Story 2.3 - Financial ratio calculations
_calculate_pe_ratio(income_data, market_cap) → Optional[float]
_calculate_pb_ratio(balance_sheet, market_cap) → Optional[float]
_calculate_roe(income_data, balance_sheet) → Optional[float]
_calculate_debt_equity(balance_sheet) → Optional[float]
```

**Critical Methods**:
- `_get_api_key()`: Multi-source API key loading with fallback
- `_enforce_rate_limit()`: Rate limiting implementation
- `_make_request(url, params)`: Robust HTTP request handler

#### **Story 1.2: Raw Data Fetching Methods** ✅
**Location**: `src/factor_lab/data/__init__.py` (FMPProvider methods)

**Implemented Endpoints**:
- `_fetch_income_statement(symbol, limit)`: Income statement data
- `_fetch_balance_sheet(symbol, limit)`: Balance sheet data  
- `_fetch_cash_flow(symbol, limit)`: Cash flow statement data
- `_fetch_financial_ratios(symbol, limit)`: Financial ratios data

**Performance Validated**:
- Response times: 0.04-0.15 seconds
- Real data testing with Apple (AAPL) Q4 2024 data
- Error handling tested with invalid symbols
- Rate limiting confirmed working

#### **Story 1.3: Data Validation and Cleaning** ✅
**Location**: `src/factor_lab/data/__init__.py` (validation methods)

**Key Methods**:
- `_validate_financial_data(data, data_type)`: Type conversion, null handling
- `_parse_date_safely(date_str)`: Consistent date parsing (strips time)
- `_perform_data_quality_checks(data, data_type)`: Quality scoring system

**Data Quality Features**:
- Empty string → None conversion
- Proper numeric type conversion
- Date consistency (strips time components)  
- Quality scoring algorithm (40% for problematic data, 100% for clean)
- Issue detection: missing acceptedDate, negative revenue, missing fields

### **🔬 TEST COVERAGE** ✅
**Location**: `tests/` directory

- **`test_fmp_methods.py`**: Real API endpoint testing with Apple data
- **`test_validation_methods.py`**: Data validation pipeline testing
- **`test_data_quality.py`**: Quality scoring validation
- All tests passing with real FMP API data

---

## 🚀 **Next Steps: Epic 2 - Time-Aware Data Processing**

### **📋 IMMEDIATE PRIORITY: Story 2.1 - Accepted Date Handling**

**Objective**: Implement look-ahead bias prevention using `acceptedDate` fields

**Tasks to Implement**:
1. **`_filter_by_accepted_date(data, as_of_date)` method**
   - Filter financial statements by acceptedDate <= as_of_date
   - Handle timezone issues in date comparison
   - Validate acceptedDate exists and is valid

2. **acceptedDate Parsing Enhancement**
   - Already implemented in `_validate_financial_data()` 
   - Need to ensure consistent timezone handling
   - Add validation that acceptedDate <= as_of_date

3. **Look-ahead Bias Prevention**
   - Critical for historical factor analysis
   - Must use acceptedDate (when data was publicly available)
   - NOT the date field (fiscal period end)

**Implementation Notes**:
- FMP API returns `acceptedDate` field in all statements
- Currently parsed to pandas Timestamp in validation
- Need point-in-time filtering for historical analysis

### **📊 Story 2.2: Trailing 12-Month Calculations**

**Objective**: Calculate trailing 12-month financial metrics

**Key Method to Implement**:
```python
def _get_trailing_12m_data(statements, as_of_date):
    """
    Sum 4 most recent quarters for income statement items.
    Use most recent balance sheet for point-in-time metrics.
    """
```

**Logic Requirements**:
- Income statement: Sum last 4 quarters
- Balance sheet: Use most recent quarter only
- Cash flow: Sum last 4 quarters
- Handle incomplete data (< 4 quarters available)
- Validate fiscal year calendars

### **📈 Story 2.3: Financial Ratio Calculations**

**Target Ratios** (matching `notebooks/fundamental_factors.ipynb`):
- **PE_ratio**: Trailing 12M net income / market cap
- **PB_ratio**: Book value / market cap  
- **ROE**: Trailing 12M net income / average equity
- **Debt_Equity**: Total debt / total equity

---

## 💾 **Configuration & Setup**

### **Required Configuration**
**File**: `config/environments.yaml`
```yaml
api_keys:
  financial_modeling_prep:
    api_key: "your_fmp_api_key_here"
```

**Alternative**: Set `FMP_API_KEY` environment variable

### **Dependencies**
**Key Packages** (from `pyproject.toml`):
- `requests`: HTTP client for FMP API
- `pandas`: Data manipulation and time series
- `numpy`: Numerical computations
- `yfinance`: Yahoo Finance data (existing)
- `openbb`: OpenBB Platform integration (existing)
- `yaml`: Configuration file parsing

### **Development Setup**
```bash
poetry install                    # Install dependencies
poetry run python -m pytest     # Run test suite
poetry run python tests/test_fmp_methods.py  # Test FMP specifically
```

---

## 📊 **Data Integration Architecture**

### **DataManager Integration** ✅
**Location**: `src/factor_lab/data/__init__.py`

```python
class DataManager:
    def __init__(self, primary_provider: str = "yahoo"):
        self.providers = {
            "yahoo": YahooFinanceProvider(),
            "openbb": OpenBBProvider(), 
            "fmp": FMPProvider(),  # ✅ INTEGRATED
        }
```

### **Data Flow**
1. **Raw Data**: FMP API → `_fetch_*()` methods
2. **Validation**: `_validate_financial_data()` → clean, typed data
3. **Quality Check**: `_perform_data_quality_checks()` → quality metrics
4. **Time Filtering**: `_filter_by_accepted_date()` → no look-ahead bias
5. **Calculations**: `_get_trailing_12m_data()` → financial metrics
6. **Factor Computation**: Public API → notebook integration

---

## 📝 **Critical Implementation Details**

### **FMP API Specifics**
- **Base URL**: `https://financialmodelingprep.com/api/v3`
- **Rate Limit**: 750 calls/minute (enforced in code)
- **Required Parameter**: `apikey` in all requests
- **Response Format**: JSON arrays of financial statement objects

### **Data Schema Knowledge**
**Income Statement Fields**:
- `revenue`, `netIncome`, `operatingIncome`, `grossProfit`
- `date` (fiscal period end), `acceptedDate` (filing date)
- `symbol`, `period` (Q1/Q2/Q3/FY), `fiscalYear`

**Balance Sheet Fields**:
- `totalAssets`, `totalEquity`, `totalDebt`
- Same metadata fields as income statement

**Critical Field Handling**:
- **CIK**: Keep as string (Securities identifier)
- **Numeric fields**: Convert to float, handle empty strings → None
- **Dates**: Strip time components for consistency
- **acceptedDate**: Convert to pandas Timestamp for timezone handling

### **Look-ahead Bias Prevention**
**CRITICAL CONCEPT**: 
- `date` field = fiscal period end (e.g., "2024-09-28" for Q4 FY2024)
- `acceptedDate` field = when data became publicly available (e.g., "2024-11-01")
- For historical analysis, use `acceptedDate` to filter available data
- Never use data where `acceptedDate > analysis_date`

---

## 🧪 **Testing Strategy**

### **Real API Testing**
- All tests use real FMP API calls with Apple (AAPL) data
- Tests validate both success and error scenarios
- Rate limiting tested and confirmed working

### **Test Data Patterns**
```python
# Good test record
{
    "date": "2024-09-28",
    "symbol": "AAPL", 
    "acceptedDate": "2024-11-01 06:01:36",
    "revenue": 391035000000.0,
    "netIncome": 94320000000.0
}

# Problematic test patterns
- Missing acceptedDate
- Negative revenue  
- Missing key fields
- Zero values
- Invalid date formats
```

### **Quality Score Benchmarks**
- **Real Apple data**: 100% quality score
- **Problematic synthetic data**: 40% quality score
- **Threshold**: <80% triggers quality warnings

---

## 🎯 **Integration Targets**

### **Notebook Integration**
**Target**: `notebooks/fundamental_factors.ipynb`

**Current State**: Uses simulated data
```python
# Simulated data generation (TO BE REPLACED)
fundamental_data = pd.DataFrame({
    'PE_ratio': np.random.normal(15, 5, len(dates)),
    'PB_ratio': np.random.normal(2, 0.5, len(dates)),
    'ROE': np.random.normal(0.15, 0.05, len(dates)),
    'Debt_Equity': np.random.normal(0.3, 0.1, len(dates))
})
```

**Target Integration**:
```python
# Real FMP data integration (TO BE IMPLEMENTED)
data_manager = DataManager()
fmp_provider = data_manager.providers['fmp']
fundamental_data = fmp_provider.get_fundamental_data(
    symbols=['AAPL', 'MSFT', 'GOOGL'], 
    start_date='2020-01-01',
    end_date='2024-01-01'
)
```

### **Expected Data Format**
**Notebook expects**:
- Daily frequency fundamental data (forward-filled from quarterly)
- Columns: `PE_ratio`, `PB_ratio`, `ROE`, `Debt_Equity`
- DatetimeIndex with trading dates
- Multi-symbol support via MultiIndex or separate DataFrames

---

## ⚠️ **Known Issues & Considerations**

### **Current Limitations**
1. **No Caching**: Every request hits FMP API (Epic 3 will add caching)
2. **No Batch Processing**: Single symbol requests only
3. **No Public API**: Methods are private (`_fetch_*`) - Epic 4 will add public interface
4. **Limited Error Recovery**: Basic retry logic only

### **Technical Debt**
1. **Missing Type Hints**: Some utility functions need type annotations
2. **Incomplete Logging**: Rate limiting logs are debug-level only
3. **No Configuration Validation**: API key validation happens at runtime
4. **Test Coverage Gaps**: No performance/load testing yet

### **FMP API Considerations**
- **Free Tier Limits**: 250 requests/day (upgrade required for production)
- **Data Delays**: Fundamental data may have 1-3 day delays
- **Historical Backfill**: Limited historical depth for some companies
- **International Coverage**: Primarily US markets

---

## 🔧 **Development Guidelines**

### **Code Style**
- **Type Hints**: Required for all public methods
- **Docstrings**: Google-style docstrings
- **Error Handling**: Graceful degradation, never crash
- **Logging**: Use module-level logger, appropriate levels

### **Testing Requirements**
- **Real API Tests**: Use actual FMP calls for integration tests
- **Mock Tests**: Use mocks for unit tests to avoid API limits
- **Error Scenarios**: Test invalid symbols, network failures, rate limits
- **Data Quality**: Validate all quality check scenarios

### **Git Workflow**
- **Feature Branches**: One branch per Epic/Story
- **Commit Messages**: Conventional commits with scope
- **Documentation**: Update implementation plan with progress

---

## 📚 **Key References**

### **Internal Documentation**
- `src/factor_lab/data/fmp_implementation_plan.md`: Detailed implementation roadmap
- `config/environments.yaml.example`: Configuration template
- `notebooks/fundamental_factors.ipynb`: Target integration notebook

### **External APIs**
- **FMP API Docs**: https://financialmodelingprep.com/developer/docs
- **Key Endpoints**:
  - Income Statement: `/api/v3/income-statement/{symbol}`
  - Balance Sheet: `/api/v3/balance-sheet-statement/{symbol}`
  - Cash Flow: `/api/v3/cash-flow-statement/{symbol}`
  - Ratios: `/api/v3/ratios/{symbol}`

### **Dependencies Documentation**
- **Pandas**: Time series handling, data manipulation
- **Requests**: HTTP client for API calls
- **OpenBB**: Existing market data integration

---

## 🎯 **Success Metrics**

### **Epic 2 Success Criteria**
- [ ] Look-ahead bias prevention implemented and tested
- [ ] Trailing 12-month calculations working for all metrics
- [ ] Financial ratios calculated correctly (PE, PB, ROE, Debt/Equity)
- [ ] Point-in-time data access with proper date filtering
- [ ] Performance maintained (<1 second for cached data when Epic 3 complete)

### **Overall Project Success**
- [ ] Real fundamental data replaces simulated data in notebooks
- [ ] No look-ahead bias in historical analysis
- [ ] Comprehensive test coverage (>90%)
- [ ] Production-ready error handling and monitoring
- [ ] Documentation complete for all public APIs

---

## 🚀 **AI Agent Pickup Instructions**

### **Immediate Next Steps**
1. **Review Epic 2 Tasks**: Check `fmp_implementation_plan.md` Story 2.1 requirements
2. **Implement acceptedDate Filtering**: Create `_filter_by_accepted_date()` method
3. **Test Look-ahead Bias Prevention**: Validate historical data filtering
4. **Create Test Cases**: Add tests for time-aware functionality

### **Development Environment**
```bash
cd /home/jcl/pythonCode/factor_lab_2/factor-lab
poetry shell                                    # Activate environment
poetry run python tests/test_fmp_methods.py    # Verify current functionality
poetry run python tests/test_data_quality.py   # Verify data quality
```

### **Code Locations**
- **Main Implementation**: `src/factor_lab/data/__init__.py` (FMPProvider class)
- **Test Suite**: `tests/test_*.py` files
- **Configuration**: `config/environments.yaml`
- **Progress Tracking**: `src/factor_lab/data/fmp_implementation_plan.md`

### **Critical Context**
- **Real API Testing**: All tests use actual FMP API calls
- **Rate Limiting**: Already implemented and tested
- **Data Quality**: Scoring system working correctly
- **Foundation Complete**: Epic 1 is solid, ready for Epic 2 time-aware features

**Happy coding! The foundation is solid - time to build the time-intelligence layer! 🚀**
