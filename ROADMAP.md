# Roadmap

Future enhancements for the superstore library.

## Up Next (Priority Order)

The following items are prioritized based on complexity, building blocks available, and user value.

---

### ✅ Priority 1: New Generator - `logs()` (COMPLETED)
Web server and application event logs generator fully implemented:

- [x] **Request timing** - Poisson-driven arrival times
- [x] **Status codes** - HTTP response sequences via MarkovChain
- [x] **Latency distributions** - Log-normal with configurable outliers
- [x] **Error bursts** - Configurable error burst simulation
- [x] **User agents** - Weighted categorical distribution
- [x] **IP addresses** - Realistic IP pool generation
- [x] **Request paths** - API and static paths with session patterns
- [x] **App logs** - Application event logs with trace/span IDs

**API**: `logs()` for web server logs, `app_logs()` for application logs

---

### ✅ Priority 2: New Generator - `finance()` (COMPLETED)
Financial data generator fully implemented:

- [x] **Stock prices** - Geometric Brownian Motion with optional jump diffusion
- [x] **Correlated assets** - Multi-asset returns via GaussianCopula
- [x] **Options data** - Black-Scholes pricing with full Greeks (delta, gamma, theta, vega)
- [x] **OHLCV bars** - Open/High/Low/Close/Volume with realistic intraday relationships
- [x] **Volume modeling** - Log-normal volume with price correlation

**API**:
- `stock_prices()` - Generate OHLCV price data for single or multiple assets
- `options_chain()` - Generate options chain with Black-Scholes pricing and Greeks
- `finance()` - Combined function returning (prices_df, options_df) tuple

**Configuration**: `FinanceConfig`, `StockConfig`, `OhlcvConfig`, `OptionsConfig`

---

### Priority 3: Crossfilter/IoT Generator Enhancements ✅ (COMPLETED)
Make machine telemetry more realistic for IoT demos. Patterns from weather apply directly:

- [x] **Anomaly injection** - CPU spikes, memory leaks, network saturation (similar to weather outliers)
- [x] **Sensor drift** - Gradual calibration drift via RandomWalk (already in weather)
- [x] **Temporal patterns** - Diurnal load cycles (use daily_adjustment from weather)
- [x] **Failure cascades** - Correlated failures via copulas
- [x] **Maintenance windows** - Scheduled downtime via MarkovChain states
- [x] **Degradation curves** - Gradual performance decrease before failure
- [x] **Recovery patterns** - Exponential cooldown after spikes

**API**:
- `telemetry()` - Generate IoT telemetry with realistic behaviors
- `TELEMETRY_SCHEMA` - Schema dictionary for telemetry output
- `TELEMETRY_SCENARIOS` - List of available preset scenarios

**Preset Scenarios** (pass `scenario=` to `telemetry()`):
- `normal` - Baseline healthy machines
- `cpu_spikes` - High CPU spike probability (15%)
- `memory_leak` - Gradual memory leak with low leak rate
- `network_congestion` - Network saturation events
- `disk_pressure` - Disk fill events leading to critical states
- `cascade_failure` - Zone-correlated failure cascades
- `maintenance_window` - Scheduled maintenance with recovery
- `sensor_drift` - Cumulative sensor drift with recalibration
- `degradation_cycle` - Progressive degradation and recovery
- `production` - Full realistic environment with all behaviors
- `chaos` - High anomaly rates for chaos engineering testing

**Configuration Structures**:
- `TelemetryConfig` - Main configuration with machine/reading counts, seed, nested configs
- `AnomalyConfig` - CPU spike, memory leak, network saturation, disk fill probabilities
- `SensorDriftConfig` - Drift rate, bias, recalibration probability
- `TemporalConfig` - Diurnal amplitude, peak hour, weekend reduction
- `FailureCascadeConfig` - Cascade probability, zone correlation
- `MaintenanceConfig` - Window probability, scheduled hours, duration
- `DegradationConfig` - Degradation rate, failure threshold, recovery rate

---

### Priority 4: Remaining Superstore Enhancements (COMPLETED)
Retail data enhancements for realistic simulation:

- [x] **Costco-style pricing** - Price endings indicate item status (IMPLEMENTED):
  - `.99` - Regular full price item
  - `.49` or `.79` - Manufacturer sale or deal
  - `.97` - Store clearance
  - `.88` - Returned or floor model
  
  **Implemented Features**:
  - `item_status` column with ItemStatus enum (Regular, ManufacturerSale, Clearance, ReturnedFloorModel)
  - `item_price` column with Costco-style price endings
  - Sale items have lower profit margins (0.4x multiplier)
  - Clearance items have very low margins (0.1x multiplier)
  - Returned/floor models have near-zero margins (0.05x multiplier)
  - Bimodal volume distribution for sale/clearance items

- [x] **Product bundling** - Related products purchased together (IMPLEMENTED):
  - `BundlingConfig` with bundle definitions and `bundle_probability`
  - New columns: `bundle_id`, `discount_multiplier`
  - Default bundles: Home Office, Entertainment Center, Back to School
  - Automatic bundle discount application to prices

- [x] **Regional variations** - Different product preferences by region (IMPLEMENTED):
  - `RegionalConfig` with per-region category weights
  - West: Technology heavy (1.4x weight)
  - East: Furniture/Office focused (1.2x/1.1x weight)
  - Central: Furniture heavy (1.3x weight)
  - South: Balanced distribution

- [x] **Inventory effects** - Stock-out simulation, backorder delays (IMPLEMENTED):
  - `InventoryConfig` with `stock_out_probability`, `backorder_delay_days`
  - New columns: `stock_status`, `backorder_days`
  - Low stock price premium (1.05x default)
  - Ship date adjustment for backordered items

- [x] **Payment methods** - Credit card, PayPal, gift cards with fraud rates (IMPLEMENTED):
  - `PaymentConfig` with `fraud_simulation` toggle
  - New columns: `payment_method`, `is_fraud`, `processing_fee`
  - Six payment methods with realistic fraud rates (0.1%-3%)
  - Processing fee calculation per payment method

---

### Priority 5: Remaining Timeseries Enhancements (COMPLETED)
Advanced financial features for realistic market simulation:

- [x] **GARCH-like volatility** - Volatility clustering (IMPLEMENTED):
  - `GarchConfig` with `alpha`, `beta`, `omega` parameters
  - GARCH(1,1) model: σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
  - Time-varying volatility with persistence

- [x] **Mean reversion** - Ornstein-Uhlenbeck process (IMPLEMENTED):
  - `MeanReversionConfig` with `theta` (speed), `mu` (mean), `sigma`
  - Euler-Maruyama discretization
  - Ideal for interest rates, spreads

- [x] **Intraday patterns** - U-shaped volatility (IMPLEMENTED):
  - `IntradayConfig` with opening/midday/closing multipliers
  - U-shaped pattern: high at open (1.5x), low midday (0.7x), high at close (1.3x)
  - Position-aware volatility scaling

- [x] **Event windows** - Abnormal returns around dates (IMPLEMENTED):
  - `EventWindowConfig` with `event_indices`, `pre/post_event_window`
  - Abnormal return injection with distance decay
  - Configurable return magnitude

- [x] **Financial metrics** - Beta, Sharpe ratio, alpha (IMPLEMENTED):
  - `FinancialMetrics` struct with alpha, beta, sharpe_ratio, volatility, max_drawdown
  - `compute_metrics` flag in TimeseriesConfig
  - First column used as market proxy for beta calculation

---

### Priority 6: Future Generators (WHEN NEEDED)

**Healthcare (`healthcare()`)** - Patient vitals, lab results
- Correlated vitals via GaussianCopula, disease progression via MarkovChain

**E-commerce (`ecommerce()`)** - Sessions, carts, conversions
- User journey via MarkovChain, cart abandonment, RFM metrics

**Social (`social()`)** - Users, connections, activity
- Power-law degree distribution, engagement cascades

**Geospatial (`geospatial()`)** - Points, trajectories
- GPS via RandomWalk, movement patterns, POI clustering

---

## ✅ Recently Completed

### Priority 4 & 5 Enhancements (v0.3.2)
- [x] **Product bundling** - BundlingConfig with bundle definitions, bundle_id column, discount multipliers
- [x] **Regional variations** - RegionalConfig with per-region category weights
- [x] **Inventory effects** - InventoryConfig with stock_out_probability, backorder_days, stock_status column
- [x] **Payment methods** - PaymentConfig with 6 payment types, fraud rates, processing_fee column
- [x] **GARCH volatility** - GarchConfig with alpha/beta/omega for volatility clustering
- [x] **Mean reversion** - MeanReversionConfig with Ornstein-Uhlenbeck process
- [x] **Intraday patterns** - IntradayConfig with U-shaped volatility
- [x] **Event windows** - EventWindowConfig with abnormal return injection
- [x] **Financial metrics** - FinancialMetrics struct with alpha, beta, sharpe_ratio, volatility, max_drawdown

### logs() Generator & Costco-Style Pricing (v0.3.1)
- [x] **logs() generator** - Web server access logs with Poisson timing, MarkovChain status codes, LogNormal latency
- [x] **app_logs() generator** - Application event logs with log levels, trace/span IDs, exceptions
- [x] **LogsConfig** - Full configuration for request rate, success rate, error bursts, latency
- [x] **Costco-style pricing** - ItemStatus enum with price endings (.99, .49/.79, .97, .88)
- [x] **Item status volume effects** - Bimodal distribution for sale/clearance items
- [x] **Item status profit margins** - Reduced margins for sale (0.4x), clearance (0.1x), returned (0.05x)

### Config Integration & Generator Enhancements (v0.3.0)
- [x] **SuperstoreConfig integration** - Full config parsing with nested seasonality, promotions, customers
- [x] **TimeseriesConfig integration** - Full config parsing with regimes, jumps, distributions
- [x] **Superstore seasonality** - Q4/holiday spikes (1.5x), summer slumps (0.9x), back-to-school (1.2x)
- [x] **Promotional effects** - Discounts correlate with quantity via discount_quantity_correlation
- [x] **Customer cohorts** - Repeat customer pools, VIP segments with vip_order_multiplier
- [x] **Realistic pricing** - Price points ($9.99, $14.99, ..., $299.99)
- [x] **Timeseries regime changes** - MarkovChain-driven volatility regimes with volatility_multipliers
- [x] **Jump diffusion** - Configurable jump_probability, jump_mean, jump_stddev
- [x] **Fat tails** - Student-t innovations via use_fat_tails and degrees_freedom
- [x] **Cross-correlation** - Correlated series via cross_correlation parameter
- [x] **Drift & cumulative** - Configurable drift and cumulative toggle

### Core Infrastructure (v0.2.x)
- [x] **Pure Rust implementation** - All data generators ported to Rust with PyO3 bindings
- [x] **Multiple output formats** - Support for pandas, polars, and dict outputs
- [x] **Seed support** - Added `seed` parameter to all generators for reproducible outputs
- [x] **Streaming data generation** - Iterator-based generators for memory-efficient processing of large datasets
- [x] **Parallel generation** - Rayon-powered multi-threaded data generation
- [x] **Distribution suite** - Normal, Log-normal, Exponential, Poisson, Pareto, Beta, Gamma, Weibull, Categorical, and Mixture distributions
- [x] **Noise models** - Gaussian noise and missing-at-random support
- [x] **Arrow memory format** - Native Apache Arrow IPC output for zero-copy interop with pandas/polars
- [x] **Direct Parquet/CSV export** - Write directly to files without DataFrame intermediary
- [x] **Correlation matrices** - Specify covariance structure between fields using Cholesky decomposition
- [x] **Type stubs** - Full `.pyi` type hints for IDE support with stubtest validation
- [x] **Deterministic mode** - Guarantee identical results across platforms with `setDeterministicMode()`
- [x] **Temporal dependencies** - AR(1), AR(p), Markov chains, random walks, exponential smoothing
- [x] **Copulas** - Gaussian, Clayton, Frank, Gumbel copulas for modeling complex dependency structures
- [x] **Benchmarking** - ASV (Airspeed Velocity) benchmark suite for performance tracking
- [x] **Weather generator** - Realistic outdoor sensor data with day/night cycles, seasons, weather events, outliers, sensor drift
- [x] **Pydantic configuration** - Structured config objects (`WeatherConfig`, `SuperstoreConfig`, etc.) with validation

---

## Future Enhancements

### Statistical Realism

- [x] **Configurable distributions** - Select from Normal, Log-normal, Exponential, Poisson, Pareto, Beta, Gamma, Weibull per field
- [x] **Mixture models** - Combine multiple distributions (e.g., bimodal income distributions)
- [x] **Copulas** - Model dependencies between correlated fields while preserving marginal distributions (Gaussian, Clayton, Frank, Gumbel)
- [ ] **Empirical distributions** - Fit distributions from real reference datasets
- [x] **Categorical sampling** - Weighted categorical distributions with configurable probabilities

### Noise Models

- [x] **Gaussian noise** - Additive white noise with configurable variance
- [ ] **Heteroscedastic noise** - Variance that changes with signal level
- [x] **Autocorrelated noise** - AR(1), AR(p) noise for time series
- [ ] **Measurement error** - Simulate instrument precision limits
- [ ] **Quantization noise** - Rounding/discretization effects
- [x] **Missing at random (MAR)** - Realistic missing data patterns
- [ ] **Censoring** - Left/right censoring for survival data

### Realistic Patterns

- [x] **Correlation matrices** - Specify covariance structure between fields
- [x] **Temporal dependencies** - Values that depend on previous observations (AR(1), AR(p), Markov chains, random walks, exponential smoothing)
- [ ] **Spatial correlation** - Geographically clustered data
- [ ] **Behavioral patterns** - Purchase cycles, session patterns, churn dynamics
- [ ] **Power law distributions** - Long-tail phenomena (popularity, wealth)
- [ ] **Zipf's law** - Realistic frequency distributions for text/categories

### Domain-Specific Realism

- [x] **Retail seasonality** - Holiday spikes, day-of-week effects, promotions ✅ Implemented
- [x] **Financial volatility** - Regime changes, fat tails, jump diffusion ✅ Implemented
- [ ] **Server metrics** - Diurnal patterns, load spikes, gradual degradation
- [ ] **User behavior** - Session lengths, click patterns, conversion funnels

### Validation Tools

- [ ] **Distribution fitting** - Fit real data and extract parameters
- [ ] **Statistical tests** - Kolmogorov-Smirnov, Chi-squared goodness-of-fit
- [ ] **Visual diagnostics** - Q-Q plots, histogram overlays
- [ ] **Moment matching** - Match mean, variance, skewness, kurtosis of reference data

## Schema & Customization

- [ ] **Custom schemas** - Define your own data schemas with field types and constraints
- [ ] **Field constraints** - Min/max values, regex patterns, enums, distributions
- [ ] **Relationships** - Foreign keys between generated tables (e.g., orders referencing customers)
- [ ] **Nullable fields** - Configurable null percentages per field
- [ ] **Localization** - Country-specific data (addresses, phone numbers, names)

## Geographic Data Consistency

### Real Location Data

- [ ] **US locations database** - Embedded dataset of real US cities, states, zip codes, counties
- [ ] **International locations** - Major countries with city/region/postal code mappings
- [ ] **Zip code validation** - Ensure zip codes match their corresponding state/city
- [ ] **Regional consistency** - Sales regions that align with actual geography
- [ ] **Coordinate generation** - Lat/long coordinates within city/state boundaries

### Coherent Fake Locations

- [ ] **Procedural city names** - Generate plausible-sounding city names
- [ ] **Consistent zip code ranges** - Fake zip codes that follow real patterns per state (e.g., CA starts with 9)
- [ ] **Area code matching** - Phone area codes that correspond to generated locations
- [ ] **Address formatting** - Country-appropriate address formats (US, UK, DE, etc.)

### Location Modes

- [ ] **Real mode** - Use actual location data from embedded databases
- [ ] **Fake mode** - Generate synthetic but internally consistent locations
- [ ] **Mixed mode** - Real states/countries with fake cities
- [ ] **Privacy mode** - Perturb real locations for anonymization

## Time Series Enhancements

- [ ] **Trend patterns** - Linear, exponential, polynomial trends
- [ ] **Seasonality** - Daily, weekly, monthly, yearly patterns
- [ ] **Anomaly injection** - Configurable outliers and anomalies
- [ ] **Missing data** - Configurable gaps in time series
- [ ] **Multiple frequencies** - Mixed-frequency data generation

## Output Formats

- [ ] **PyArrow Table** - Direct Arrow table output
- [ ] **DuckDB integration** - Direct insertion into DuckDB tables
- [ ] **JSON/JSONL** - Streaming JSON output
- [ ] **SQL inserts** - Generate SQL INSERT statements

## Developer Experience

- [ ] **CLI tool** - Command-line interface for quick data generation
- [ ] **WebAssembly bindings** - Use in browser/Node.js applications
- [x] **Type stubs** - Full `.pyi` type hints for IDE support with stubtest validation
- [ ] **Jupyter widgets** - Interactive data preview widgets
- [ ] **Data profiling** - Summary statistics of generated data
- [x] **Pydantic configs** - Structured configuration objects with validation

## Testing & Quality

- [ ] **Property-based testing** - Hypothesis integration for thorough testing
- [x] **Benchmarks** - ASV (Airspeed Velocity) benchmark suite for performance tracking
- [ ] **Data validation** - Built-in validation against schemas
- [ ] **Distribution testing** - Verify statistical properties of generated data

## Documentation

- [ ] **API documentation** - Comprehensive rustdoc and Sphinx docs
- [ ] **Tutorials** - Step-by-step guides for common use cases
- [ ] **Examples gallery** - Ready-to-use examples for various domains

## Community

- [ ] **Plugin system** - Allow third-party data generators
- [ ] **Schema registry** - Community-contributed schemas
- [ ] **Integration guides** - pytest fixtures, FastAPI, Django, etc.
