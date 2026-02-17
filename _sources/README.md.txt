# superstore

High-performance synthetic data generation library for testing and development.

[![Build Status](https://github.com/1kbgz/superstore/actions/workflows/build.yaml/badge.svg?branch=main&event=push)](https://github.com/1kbgz/superstore/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/1kbgz/superstore/branch/main/graph/badge.svg)](https://codecov.io/gh/1kbgz/superstore)
[![License](https://img.shields.io/github/license/1kbgz/superstore)](https://github.com/1kbgz/superstore)
[![PyPI](https://img.shields.io/pypi/v/superstore.svg)](https://pypi.python.org/pypi/superstore)

## Overview

superstore is a Rust-powered Python library for generating realistic synthetic datasets. It provides:

### Data Generators

| Generator                                 | Description                                | Use Cases                        |
| ----------------------------------------- | ------------------------------------------ | -------------------------------- |
| **[Retail](docs/src/retail.md)**          | Sales transactions, employees              | BI dashboards, forecasting       |
| **[Time Series](docs/src/timeseries.md)** | Financial-style series with regimes, jumps | Quant research, backtesting      |
| **[Weather](docs/src/weather.md)**        | Sensor data with seasonal/diurnal patterns | IoT analytics, anomaly detection |
| **[Logs](docs/src/logs.md)**              | Web server & application logs              | Observability, alerting          |
| **[Finance](docs/src/finance.md)**        | Stock prices, OHLCV, options chains        | Trading systems, risk analysis   |
| **[Telemetry](docs/src/telemetry.md)**    | Machine metrics, anomalies, failures       | DevOps dashboards, ML training   |

### Statistical Tools

| Tool                                           | Description                           | Use Cases                         |
| ---------------------------------------------- | ------------------------------------- | --------------------------------- |
| **[Distributions](docs/src/distributions.md)** | Sample from statistical distributions | Simulation, Monte Carlo           |
| **[Copulas](docs/src/copulas.md)**             | Correlated multivariate data          | Risk modeling, portfolio analysis |
| **[Temporal Models](docs/src/temporal.md)**    | AR, Markov chains, random walks       | Time series simulation            |

### Key Features

- **Rust-powered**: High-performance generation, 10-100x faster than pure Python
- **Flexible output**: pandas DataFrame, polars DataFrame, or Python dicts
- **Configurable**: Pydantic config classes for validated, structured configuration
- **Reproducible**: Seed support for deterministic generation
- **Scalable**: Streaming and parallel generation for large datasets

## Installation

```bash
pip install superstore
```

For development with polars support:

```bash
pip install superstore[develop]
```

## Quick Start

```python
from superstore import superstore, employees, timeseries, weather

# Generate 1000 retail records as a pandas DataFrame
df = superstore(count=1000)

# Generate as polars DataFrame
df_polars = superstore(count=1000, output="polars")

# Generate as list of dicts
records = superstore(count=1000, output="dict")
```

## Reproducibility with Seeds

All data generators support an optional `seed` parameter for reproducible random data generation:

```python
from superstore import superstore, employees, getTimeSeries, machines

# Same seed produces identical data
df1 = superstore(count=100, seed=42)
df2 = superstore(count=100, seed=42)
assert df1.equals(df2)  # True

# Works with all generators
employees_df = employees(count=50, seed=123)
timeseries_df = timeseries(nper=30, seed=456)
weather_df = weather(count=100, seed=789)
machine_list = machines(count=10, seed=321)

# No seed means random data each time
df3 = superstore(count=100)  # Different each call
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/1kbgz/superstore.git
cd superstore

# Install development dependencies
make develop
```

### Building

```bash
# Build Python wheel
make build
```

### Testing

```bash
# Run all tests
make test
```

### Linting

```bash
# Run linters
make lint

# Fix formatting
make fix
```

## Architecture

superstore uses a hybrid Rust/Python architecture:

- **rust/**: Core Rust library with all data generation logic
- **src/**: PyO3 bindings exposing Rust functions to Python
- **superstore/**: Python package with native module

The core data generation is implemented in Rust for performance, with PyO3 providing seamless Python integration. Output format conversion (pandas/polars/dict) happens in the Rust bindings layer.

## License

This library is released under the [Apache 2.0 license](./LICENSE)

> [!NOTE]
> This library was generated using [copier](https://copier.readthedocs.io/en/stable/) from the [Base Python Project Template repository](https://github.com/python-project-templates/base).
