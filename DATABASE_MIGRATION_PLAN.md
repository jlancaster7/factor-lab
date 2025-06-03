# Database Migration Plan: From File Cache to Historical Data Store

## Executive Summary

The current file-based caching system is fundamentally misaligned with the nature of financial data. Historical financial statements are **immutable records**, not temporary cache entries. This migration plan outlines the transition to a proper database-based historical data store that will dramatically improve performance, scalability, and maintainability.

## Current State Analysis

### Problems with File-Based Cache

1. **Architectural Mismatch**
   - TTL-based expiration makes no sense for immutable historical data (Q3 2023 earnings never change)
   - Cache invalidation strategies don't apply to permanent historical records
   - File-per-statement-type-per-symbol creates thousands of small files

2. **Performance Issues**
   - Cannot query by date ranges efficiently
   - Must fetch entire quarterly history regardless of actual needs
   - No relational joins between income statements and balance sheets
   - Story 3.5 bottleneck: 2,000+ individual market cap calculations

3. **Scalability Limitations**
   - File system performance degrades with thousands of files
   - No concurrent access controls
   - Memory inefficient: must load full datasets for small queries
   - No indexing or query optimization

4. **Operational Complexity**
   - Cache corruption issues (recently fixed with atomic writes)
   - No backup/restore strategy
   - Difficult to inspect data or run analytics
   - No data integrity constraints

## Target Architecture: Database-Based Historical Data Store

### Database Selection: PostgreSQL with TimescaleDB

**Primary Choice: PostgreSQL + TimescaleDB**
- **TimescaleDB**: Purpose-built for time-series financial data
- **PostgreSQL**: Robust, ACID-compliant, excellent Python integration
- **Hybrid approach**: Time-series optimizations + relational capabilities
- **Scalability**: Handles massive time-series datasets efficiently

**Alternative: Snowflake (User has access)**
- Cloud-native data warehouse
- Excellent for analytics workloads
- Built-in time travel and versioning
- May be overkill for current scale but future-proof

### Schema Design

```sql
-- Core financial statements table
CREATE TABLE financial_statements (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    statement_type VARCHAR(20) NOT NULL, -- 'income', 'balance_sheet', 'cash_flow'
    fiscal_date DATE NOT NULL,
    fiscal_period VARCHAR(10) NOT NULL, -- 'Q1', 'Q2', 'Q3', 'Q4', 'FY'
    accepted_date DATE,
    filing_date DATE,
    metrics JSONB NOT NULL, -- All statement line items
    data_quality_score FLOAT,
    source_provider VARCHAR(20) DEFAULT 'fmp',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(symbol, statement_type, fiscal_date, fiscal_period)
);

-- Financial ratios (calculated metrics)
CREATE TABLE financial_ratios (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    fiscal_date DATE NOT NULL,
    accepted_date DATE,
    pe_ratio FLOAT,
    pb_ratio FLOAT,
    roe FLOAT,
    roa FLOAT,
    debt_to_equity FLOAT,
    current_ratio FLOAT,
    operating_margin FLOAT,
    net_margin FLOAT,
    market_cap BIGINT,
    shares_outstanding BIGINT,
    source_provider VARCHAR(20) DEFAULT 'fmp',
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(symbol, fiscal_date)
);

-- Price data (for market cap calculations)
CREATE TABLE price_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open_price FLOAT,
    high_price FLOAT,
    low_price FLOAT,
    close_price FLOAT,
    volume BIGINT,
    adjusted_close FLOAT,
    source_provider VARCHAR(20) DEFAULT 'fmp',
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(symbol, date)
);

-- Metadata tracking
CREATE TABLE data_fetch_log (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    data_type VARCHAR(20) NOT NULL,
    fetch_date TIMESTAMP DEFAULT NOW(),
    records_fetched INTEGER,
    api_calls_made INTEGER,
    success BOOLEAN,
    error_message TEXT
);
```

### Indexes for Performance

```sql
-- Primary lookup patterns
CREATE INDEX idx_statements_symbol_date ON financial_statements(symbol, fiscal_date);
CREATE INDEX idx_statements_type_date ON financial_statements(statement_type, fiscal_date);
CREATE INDEX idx_statements_accepted_date ON financial_statements(accepted_date);

CREATE INDEX idx_ratios_symbol_date ON financial_ratios(symbol, fiscal_date);
CREATE INDEX idx_ratios_accepted_date ON financial_ratios(accepted_date);

CREATE INDEX idx_prices_symbol_date ON price_data(symbol, date);

-- TimescaleDB hypertables (if using TimescaleDB)
SELECT create_hypertable('financial_statements', 'fiscal_date', chunk_time_interval => INTERVAL '1 year');
SELECT create_hypertable('financial_ratios', 'fiscal_date', chunk_time_interval => INTERVAL '1 year');
SELECT create_hypertable('price_data', 'date', chunk_time_interval => INTERVAL '1 month');
```

## Migration Strategy

### Phase 1: Database Setup & Infrastructure (Week 1)

1. **Environment Setup**
   ```bash
   # Option A: Local PostgreSQL + TimescaleDB
   docker run -d --name factor-lab-db \
     -p 5432:5432 \
     -e POSTGRES_DB=factor_lab \
     -e POSTGRES_USER=factor_lab \
     -e POSTGRES_PASSWORD=secure_password \
     timescale/timescaledb:latest-pg15
   
   # Option B: Snowflake (user has access)
   # Use existing Snowflake account
   ```

2. **Schema Creation**
   - Deploy SQL schema with tables and indexes
   - Set up connection configuration
   - Create database migration utilities

3. **Data Access Layer**
   ```python
   # New database provider
   class DatabaseProvider:
       def __init__(self, connection_string):
           self.engine = create_engine(connection_string)
       
       def get_financial_statements(self, symbol, start_date, end_date, statement_types):
           """Efficient time-range query"""
           return pd.read_sql("""
               SELECT * FROM financial_statements 
               WHERE symbol = %s 
               AND fiscal_date BETWEEN %s AND %s
               AND statement_type = ANY(%s)
               ORDER BY fiscal_date
           """, self.engine, params=[symbol, start_date, end_date, statement_types])
   ```

### Phase 2: Data Migration (Week 2)

1. **Cache Export Utility**
   ```python
   class CacheMigrationTool:
       def export_cache_to_database(self):
           """Migrate existing cache files to database"""
           for cache_file in self.find_cache_files():
               data = self.load_cache_file(cache_file)
               self.insert_to_database(data)
   ```

2. **Data Validation**
   - Compare cache vs database results
   - Verify data integrity and completeness
   - Performance benchmarking

3. **Parallel Operation**
   - Run both systems in parallel
   - Gradual migration of symbols
   - Fallback to cache if database issues

### Phase 3: Provider Refactoring (Week 3)

1. **Replace FMP Cache Logic**
   ```python
   # Before: File-based cache
   def get_fundamental_factors(self, symbols, start_date, end_date):
       # Complex cache management, fixed limits, inefficient queries
       
   # After: Database queries
   def get_fundamental_factors(self, symbols, start_date, end_date):
       return self.db.get_financial_ratios(
           symbols=symbols,
           start_date=start_date,
           end_date=end_date
       )
   ```

2. **Efficient Date Range Queries**
   ```sql
   -- Get only needed quarters for TTM calculations
   SELECT * FROM financial_statements 
   WHERE symbol = 'AAPL' 
   AND fiscal_date BETWEEN '2023-01-01' AND '2024-12-31'
   AND statement_type = 'income'
   ORDER BY fiscal_date;
   ```

3. **Solve Story 3.5 Performance Issue**
   ```python
   # Efficient market cap calculation with single query
   def calculate_market_caps_batch(self, symbols, dates):
       return pd.read_sql("""
           SELECT symbol, date, close_price * shares_outstanding as market_cap
           FROM price_data p
           JOIN financial_ratios r ON p.symbol = r.symbol 
           WHERE p.symbol = ANY(%s) AND p.date = ANY(%s)
       """, self.engine, params=[symbols, dates])
   ```

### Phase 4: Performance Optimization (Week 4)

1. **Query Optimization**
   - Analyze query patterns
   - Add specialized indexes
   - Implement query caching for repeated requests

2. **Batch Operations**
   - Bulk insert/update operations
   - Connection pooling
   - Async database operations

3. **Monitoring & Analytics**
   - Query performance monitoring
   - Data freshness tracking
   - Usage analytics

## Expected Performance Improvements

### Current State (File Cache)
- **Story 3.5 Issue**: 2,016 individual market cap calculations
- **Data Fetching**: Always fetch 20 quarters regardless of need
- **Time Complexity**: O(n) file reads per calculation
- **Typical Performance**: 50+ seconds for 16 symbols, 1 year, monthly

### Future State (Database)
- **Batch Operations**: Single query for all market caps
- **Time-Range Queries**: Fetch only needed date ranges
- **Relational Joins**: Combine statements efficiently
- **Expected Performance**: <5 seconds for same workload (10x+ improvement)

### Performance Benchmarks (Projected)

| Operation | Current (Cache) | Future (Database) | Improvement |
|-----------|----------------|-------------------|-------------|
| Single symbol, 1 year | 3.2 seconds | 0.3 seconds | 10x |
| 16 symbols, 1 year | 50 seconds | 4 seconds | 12x |
| 100 symbols, 5 years | 20+ minutes | 30 seconds | 40x+ |
| Market cap batch (Story 3.5) | 2,000+ operations | 1 query | 100x+ |

## Implementation Timeline

### Week 1: Infrastructure
- [ ] Database setup (PostgreSQL + TimescaleDB or Snowflake)
- [ ] Schema deployment
- [ ] Basic connection layer
- [ ] Configuration management

### Week 2: Migration
- [ ] Cache export utility
- [ ] Data migration scripts
- [ ] Validation tools
- [ ] Parallel operation setup

### Week 3: Provider Refactoring
- [ ] DatabaseProvider class
- [ ] Replace cache logic in FMPProvider
- [ ] Update get_fundamental_factors method
- [ ] Solve Story 3.5 bottleneck

### Week 4: Optimization & Testing
- [ ] Performance optimization
- [ ] Integration testing
- [ ] Notebook updates
- [ ] Documentation updates

## Risk Mitigation

### Data Loss Prevention
- **Backup Strategy**: Regular database backups
- **Migration Validation**: Comprehensive data comparison
- **Rollback Plan**: Keep cache files until migration proven

### Performance Risks
- **Gradual Migration**: Symbol-by-symbol migration
- **Load Testing**: Test with full data volumes
- **Monitoring**: Real-time performance tracking

### Operational Risks
- **Parallel Systems**: Run both cache and database initially
- **Feature Flags**: Toggle between cache and database
- **Monitoring**: Alert on database issues with cache fallback

## Configuration Management

### Database Connection
```yaml
# config/environments.yaml
database:
  provider: "postgresql"  # or "snowflake"
  host: "localhost"
  port: 5432
  database: "factor_lab"
  username: "factor_lab"
  password: "${DATABASE_PASSWORD}"
  pool_size: 10
  max_overflow: 20

# Feature flags
features:
  use_database: true
  fallback_to_cache: true
  migration_mode: false
```

### Migration Control
```python
class DataProvider:
    def __init__(self):
        if config.features.use_database:
            self.primary = DatabaseProvider()
            self.fallback = CacheProvider() if config.features.fallback_to_cache else None
        else:
            self.primary = CacheProvider()
```

## Success Metrics

### Performance Metrics
- [ ] Story 3.5 bottleneck resolved (10x+ speedup)
- [ ] Notebook execution time <10 seconds for demo scenarios
- [ ] Database query response time <100ms for typical queries
- [ ] Memory usage reduction >50%

### Operational Metrics
- [ ] Zero data loss during migration
- [ ] 99.9% uptime during transition
- [ ] All integration tests passing
- [ ] No cache corruption issues

### Developer Experience
- [ ] Simplified query interface
- [ ] Better debugging and inspection tools
- [ ] Reduced complexity in provider code
- [ ] Easier to add new data sources

## Long-term Benefits

### Scalability
- **Data Volume**: Handle 1000+ symbols, 10+ years of history
- **Query Performance**: Sub-second response for complex analytics
- **Concurrent Users**: Multiple researchers using same data store

### Analytics Capabilities
- **Cross-symbol Analysis**: Compare metrics across industries
- **Time-series Analytics**: Trend analysis, seasonality detection
- **Data Quality**: Better visibility into data completeness
- **Custom Queries**: Researchers can run ad-hoc SQL queries

### Operational Excellence
- **Backup/Restore**: Proper disaster recovery
- **Data Lineage**: Track data sources and transformations
- **Version Control**: Handle data corrections and restatements
- **Integration**: Easier to add new data providers

## Next Steps

1. **Immediate**: Get stakeholder approval for migration plan
2. **Week 1**: Begin database infrastructure setup
3. **Ongoing**: Update project documentation and handoff notes
4. **Future**: Consider additional data sources (Bloomberg, Refinitiv, etc.)

This migration represents a fundamental architectural improvement that will enable Factor Lab to scale to production-grade quantitative research workflows.