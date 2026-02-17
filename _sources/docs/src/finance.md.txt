# Financial Data Generation

Generate realistic market data including stock prices, OHLCV bars, and options chains.

## Overview

The finance generators create data suitable for:

- Quantitative finance research
- Trading strategy backtesting
- Options pricing demos
- Risk management analysis
- Portfolio analytics

## Stock Prices

The `stock_prices()` function generates realistic price series using Geometric Brownian Motion (GBM).

### Basic Usage

```python
from superstore import stock_prices

# Generate 252 days of stock prices (1 trading year)
df = stock_prices(nper=252, initial_price=150.0)

# Multiple tickers with correlation
df = stock_prices(nper=252, tickers=["AAPL", "GOOGL", "MSFT"])
```

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Trading date |
| `ticker` | str | Stock ticker symbol |
| `close` | float | Closing price |

---

## Options Chain

The `options_chain()` function generates options chain data with Black-Scholes pricing and Greeks.

### Basic Usage

```python
from superstore import options_chain

# Generate options chain for a $150 stock
df = options_chain(underlying_price=150.0, n_strikes=10)
```

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `strike` | float | Strike price |
| `expiration` | date | Expiration date |
| `type` | str | Option type (call/put) |
| `premium` | float | Option premium |
| `delta` | float | Delta |
| `gamma` | float | Gamma |
| `theta` | float | Theta |
| `vega` | float | Vega |
| `iv` | float | Implied volatility |

---

## Comprehensive Finance Data

The `finance()` function generates complete financial datasets including OHLCV bars, multi-asset returns, and options.

### Basic Usage

```python
from superstore import finance

# Generate full finance dataset
data = finance(tickers=["AAPL", "GOOGL"], nper=252)
```

---

## Configuration

Use `FinanceConfig` for detailed control:

```python
from superstore import finance, FinanceConfig

config = FinanceConfig(
    ndays=500,
    n_assets=3,
    seed=42,
)
data = finance(config=config)
```

### Basic Settings

```python
config = FinanceConfig(
    ndays=252,         # Trading days (252 = 1 year)
    n_assets=3,        # Number of assets
    tickers=["AAPL", "GOOGL", "MSFT"],
    start_date="2024-01-02",
    seed=42,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ndays` | `252` | Number of trading days |
| `n_assets` | `1` | Number of assets |
| `tickers` | `["AAPL"]` | Ticker symbols |
| `start_date` | `"2024-01-02"` | Start date (ISO format) |
| `asset_correlation` | `0.5` | Correlation between assets |

### Stock Configuration

Configure the underlying price process:

```python
config = FinanceConfig(
    ndays=252,
    stock={
        "annual_drift": 0.10,        # 10% expected annual return
        "annual_volatility": 0.25,   # 25% annual volatility
        "initial_price": 175.0,      # Starting price

        # Jump diffusion
        "enable_jumps": True,
        "jump_probability": 0.03,    # 3% daily jump probability
        "jump_mean": -0.01,          # Slight downward bias (crashes)
        "jump_stddev": 0.08,         # Jump size variability
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `annual_drift` | `0.08` | Annual expected return (mu) |
| `annual_volatility` | `0.20` | Annual volatility (sigma) |
| `initial_price` | `100.0` | Initial stock price |
| `enable_jumps` | `False` | Enable jump diffusion |
| `jump_probability` | `0.02` | Daily probability of jump |
| `jump_mean` | `0.0` | Mean jump size (log-normal) |
| `jump_stddev` | `0.05` | Jump size standard deviation |

### OHLCV Configuration

Configure OHLCV (Open-High-Low-Close-Volume) bars:

```python
config = FinanceConfig(
    ndays=252,
    ohlcv={
        "avg_volume": 2_000_000,       # Average daily volume
        "volume_volatility": 0.6,      # Volume variability
        "intraday_volatility": 0.03,   # Intraday price range
        "volume_price_correlation": 0.4, # Volume-return correlation
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `avg_volume` | `1,000,000` | Average daily trading volume |
| `volume_volatility` | `0.5` | Volatility of volume (log-normal) |
| `intraday_volatility` | `0.02` | Intraday price range volatility |
| `volume_price_correlation` | `0.3` | Volume-absolute return correlation |

### Options Configuration

Configure options chain generation with Black-Scholes pricing:

```python
config = FinanceConfig(
    ndays=252,
    options={
        "risk_free_rate": 0.045,       # Risk-free rate
        "dividend_yield": 0.015,        # Dividend yield
        "expirations": [7, 14, 30, 45, 60, 90],  # Days to expiration
        "strike_offsets": [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15],
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `risk_free_rate` | `0.05` | Annual risk-free rate |
| `dividend_yield` | `0.02` | Annual dividend yield |
| `expirations` | `[7, 14, 30, 60, 90]` | Days to expiration |
| `strike_offsets` | `[0.90, 0.95, ...]` | Strike as multiplier of spot |

### Complete Example

```python
from superstore import finance, FinanceConfig

config = FinanceConfig(
    ndays=500,
    n_assets=4,
    tickers=["AAPL", "GOOGL", "MSFT", "AMZN"],
    start_date="2023-01-03",
    asset_correlation=0.65,
    seed=42,

    # Stock process
    stock={
        "annual_drift": 0.12,
        "annual_volatility": 0.22,
        "initial_price": 150.0,
        "enable_jumps": True,
        "jump_probability": 0.02,
        "jump_mean": -0.005,
        "jump_stddev": 0.06,
    },

    # OHLCV bars
    ohlcv={
        "avg_volume": 5_000_000,
        "volume_volatility": 0.5,
        "intraday_volatility": 0.025,
        "volume_price_correlation": 0.35,
    },

    # Options chain
    options={
        "risk_free_rate": 0.05,
        "dividend_yield": 0.01,
        "expirations": [7, 14, 21, 30, 45, 60, 90, 120],
        "strike_offsets": [0.85, 0.90, 0.95, 0.97, 1.0, 1.03, 1.05, 1.10, 1.15],
    },
)

data = finance(config=config)
```

---

## API Reference

See the full API documentation:

- [stock_prices()](api.md)
- [options_chain()](api.md)
- [finance()](api.md)
- [FinanceConfig](api.md)
