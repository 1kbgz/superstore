# Time Series Generation

Generate financial-style time series with regime changes, volatility clustering, and jump diffusion.

## Overview

The time series generator creates synthetic data suitable for:

- Quantitative finance research
- Risk management backtesting
- Portfolio optimization demos
- Volatility modeling
- Algorithmic trading simulation

## Basic Usage

```python
from superstore import timeseries

# Generate 252 business days of 4 correlated series
df = timeseries(nper=252, ncol=4)

# Daily data with different output
df = timeseries(nper=100, ncol=3, freq="D", output="polars")
```

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Date index |
| `A`, `B`, `C`, ... | float | Time series values (up to 26 columns) |

## Configuration

The `TimeseriesConfig` class provides comprehensive control over the generated series:

```python
from superstore import timeseries, TimeseriesConfig

config = TimeseriesConfig(
    nper=500,           # 500 periods
    ncol=4,             # 4 columns (A, B, C, D)
    freq="B",           # Business days (B=business, D=daily, W=weekly, M=monthly)
    seed=42,            # Reproducible output
)
df = timeseries(config=config)
```

### Process Parameters

Control the statistical properties of the generated series:

```python
config = TimeseriesConfig(
    nper=252,

    # AR(1) parameters
    ar_phi=0.95,        # Persistence/mean reversion (0 to 1)
    sigma=1.0,          # Innovation standard deviation
    drift=0.0001,       # Drift/trend per period

    # Cumulative behavior
    cumulative=True,    # Apply cumsum (price-like behavior)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ar_phi` | `0.95` | AR(1) persistence parameter (-1 to 1) |
| `sigma` | `1.0` | Innovation standard deviation |
| `drift` | `0.0` | Drift/trend per period |
| `cumulative` | `True` | Apply cumulative sum for price-like behavior |

### Fat Tails (Student-t Distribution)

Financial returns often exhibit fat tails. Enable Student-t innovations:

```python
config = TimeseriesConfig(
    nper=252,
    use_fat_tails=True,      # Use Student-t instead of normal
    degrees_freedom=5.0,      # Lower = fatter tails (2.1 to 30)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_fat_tails` | `False` | Use Student-t distribution |
| `degrees_freedom` | `5.0` | Degrees of freedom (2.1-30) |

### Cross-Correlation

Generate correlated multi-asset series:

```python
config = TimeseriesConfig(
    nper=252,
    ncol=5,
    cross_correlation=0.6,   # 60% correlation between columns
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cross_correlation` | `0.0` | Correlation between columns (-1 to 1) |

### Regime Switching

Model different market regimes with varying volatility:

```python
config = TimeseriesConfig(
    nper=500,
    regimes={
        "enable": True,
        "n_regimes": 3,           # Number of regimes
        "regime_persistence": 0.97,  # Probability of staying in regime
        "volatility_multipliers": [0.5, 1.0, 2.5],  # Low, normal, high volatility
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable` | `False` | Enable regime switching |
| `n_regimes` | `2` | Number of distinct regimes (2-5) |
| `regime_persistence` | `0.95` | Probability of staying in current regime |
| `volatility_multipliers` | `[1.0, 2.5]` | Volatility multiplier per regime |

Example: A 3-regime market model with calm (0.5x), normal (1x), and turbulent (2.5x) volatility states.

### Jump Diffusion

Add discrete jumps for crash/rally modeling:

```python
config = TimeseriesConfig(
    nper=252,
    jumps={
        "enable": True,
        "jump_probability": 0.02,  # 2% chance per period
        "jump_mean": -0.02,        # Negative mean (crashes more likely)
        "jump_stddev": 0.05,       # Jump size variability
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable` | `False` | Enable jump diffusion |
| `jump_probability` | `0.01` | Jump probability per period |
| `jump_mean` | `0.0` | Mean jump size |
| `jump_stddev` | `0.05` | Jump size standard deviation |

### Complete Example

Realistic financial time series with all features:

```python
from superstore import timeseries, TimeseriesConfig

config = TimeseriesConfig(
    nper=1000,
    ncol=4,
    freq="B",
    seed=42,

    # Process
    ar_phi=0.98,
    sigma=0.015,
    drift=0.0003,
    cumulative=True,

    # Fat tails
    use_fat_tails=True,
    degrees_freedom=4.0,

    # Cross-correlation
    cross_correlation=0.5,

    # Regime switching
    regimes={
        "enable": True,
        "n_regimes": 3,
        "regime_persistence": 0.98,
        "volatility_multipliers": [0.6, 1.0, 2.0],
    },

    # Jump diffusion
    jumps={
        "enable": True,
        "jump_probability": 0.01,
        "jump_mean": -0.01,
        "jump_stddev": 0.03,
    },
)

df = timeseries(config=config)
```

---

## API Reference

See the full API documentation:

- [timeseries()](api.md)
- [TimeseriesConfig](api.md)
