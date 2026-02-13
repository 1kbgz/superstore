# superstore

High-performance synthetic data generation library for testing and development.

[![Build Status](https://github.com/1kbgz/superstore/actions/workflows/build.yaml/badge.svg?branch=main&event=push)](https://github.com/1kbgz/superstore/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/1kbgz/superstore/branch/main/graph/badge.svg)](https://codecov.io/gh/1kbgz/superstore)
[![License](https://img.shields.io/github/license/1kbgz/superstore)](https://github.com/1kbgz/superstore)
[![PyPI](https://img.shields.io/pypi/v/superstore.svg)](https://pypi.python.org/pypi/superstore)

## Overview

superstore is a Rust-powered Python library for generating realistic synthetic datasets. It provides:

- **Retail Data**: Generate superstore-style sales data with orders, products, and customers
- **Employee Data**: Generate employee records with personal information
- **Time Series Data**: Generate random walk time series for financial/metric simulation
- **Machine Metrics**: Generate server/machine monitoring data with usage, status, and job information

All data generators are implemented in Rust for maximum performance and can output data as:

- **pandas DataFrame** (default)
- **polars DataFrame**
- **Python dict/list**

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

# Generate reproducible data with a seed
df = superstore(count=1000, seed=42)
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

## API Reference

### Retail Data Generation

#### `superstore(count=1000, output="pandas", seed=None)`

Generate synthetic superstore sales records.

**Parameters:**

- `count` (int, default=1000): Number of records to generate
- `output` (str, default="pandas"): Output format - "pandas", "polars", or "dict"
- `seed` (int, optional): Random seed for reproducible generation

**Returns:** DataFrame or list of dicts with columns:

- Row ID, Order ID, Order Date, Ship Date, Ship Mode
- Customer ID, Segment, Country, City, State, Postal Code, Region
- Product ID, Category, Sub-Category
- Sales, Quantity, Discount, Profit

```python
from superstore import superstore

# Pandas DataFrame
df = superstore(count=500)

# Polars DataFrame
df = superstore(count=500, output="polars")

# List of dicts
records = superstore(count=500, output="dict")
```

#### `employees(count=1000, output="pandas", seed=None)`

Generate synthetic employee records.

**Parameters:**

- `count` (int, default=1000): Number of employees to generate
- `output` (str, default="pandas"): Output format - "pandas", "polars", or "dict"
- `seed` (int, optional): Random seed for reproducible generation

**Returns:** DataFrame or list of dicts with columns:

- Row ID, Employee ID, First Name, Surname, Prefix, Suffix
- Phone Number, Email, SSN
- Street, City, Postal Code, Region, State, Country
- Start Date, Date of Birth

```python
from superstore import employees

# Generate 100 employees
df = employees(count=100)
```

### Time Series Data Generation

#### `timeseries(nper=30, freq="B", ncol=4, output="pandas", seed=None)`

Generate a random walk time series with multiple columns.

**Parameters:**

- `nper` (int, default=30): Number of periods
- `freq` (str, default="B"): Frequency - "B" (business days), "D" (daily), "W" (weekly), "M" (monthly)
- `ncol` (int, default=4): Number of columns (named A, B, C, D, ...)
- `output` (str, default="pandas"): Output format - "pandas", "polars", or "dict"
- `seed` (int, optional): Random seed for reproducible generation

**Returns:** DataFrame with DatetimeIndex and random walk columns

```python
from superstore import timeseries

# 30 business days, 4 columns
df = timeseries()

# 100 daily periods, 6 columns
df = timeseries(nper=100, freq="D", ncol=6)

# As polars DataFrame
df = timeseries(nper=50, output="polars")
```

#### `timeseriesData(nper=30, freq="B", ncol=4, output="pandas", seed=None)`

Generate multiple independent time series as a dict of Series/DataFrames.

**Parameters:** Same as `timeseries`

**Returns:** Dict mapping column names (A, B, C, ...) to pandas Series or polars DataFrames

```python
from superstore import timeseriesData

# Dict of pandas Series
series_dict = timeseriesData()

# Dict of polars DataFrames
series_dict = timeseriesData(output="polars")
```

### Weather Data Generation

#### `weather(count=1000, output="pandas", seed=None, **kwargs)` or `weather(config: WeatherConfig)`

Generate realistic weather sensor readings with day/night cycles, seasonal patterns, and weather events.

**Parameters:**

- `count` (int, default=1000): Number of readings to generate
- `output` (str, default="pandas"): Output format - "pandas", "polars", or "dict"
- `seed` (int, optional): Random seed for reproducible generation
- `climate_zone` (str, default="temperate"): Climate zone - "tropical", "subtropical", "temperate", "continental", "polar", "arid", "mediterranean"
- `latitude` (float, default=40.0): Latitude for day/night calculations (-90 to 90)
- `start_date` (str, optional): Start date (YYYY-MM-DD)
- `frequency_minutes` (int, default=15): Reading frequency in minutes
- `base_temp_celsius` (float, default=15.0): Annual average temperature
- `enable_weather_events` (bool, default=True): Enable weather event simulation
- `outlier_probability` (float, default=0.01): Probability of sensor errors
- `sensor_drift` (bool, default=False): Enable gradual sensor calibration drift

**Returns:** DataFrame or list of dicts with columns:

- timestamp, temperature_celsius, humidity_percent, precipitation_mm
- pressure_hpa, wind_speed_kmh, weather_event, is_outlier

```python
from superstore import weather
from superstore.config import WeatherConfig, ClimateZone

# Basic usage
df = weather(count=500, seed=42)

# With climate zone
df = weather(count=500, climate_zone="tropical", latitude=-5.0)

# Using Pydantic config for full control
config = WeatherConfig(
    count=1000,
    climate_zone=ClimateZone.MEDITERRANEAN,
    latitude=35.0,
    base_temp_celsius=20.0,
    enable_weather_events=True,
    outlier_probability=0.02,
    sensor_drift=True,
)
df = weather(config)
```

### Machine Metrics Generation

These functions simulate server/machine monitoring data for testing dashboards and alerting systems.

#### `machines(count=100, json=False, seed=None)`

Generate a list of machine definitions.

**Parameters:**

- `count` (int, default=100): Number of machines to generate
- `json` (bool, default=False): JSON format flag (reserved for future use)
- `seed` (int, optional): Random seed for reproducible generation

**Returns:** List of machine dicts with: machine_id, kind (edge/core/worker), cores, region, zone

```python
from superstore import machines

# Generate 50 machines
machine_list = machines(count=50)
```

#### `usage(machine, json=False, seed=None)`

Generate usage metrics for a machine.

**Parameters:**

- `machine` (dict): Machine dict from `machines()`
- `json` (bool, default=False): JSON format flag
- `seed` (int, optional): Random seed for reproducible generation

**Returns:** Dict with machine info plus: cpu, mem, free, network, disk (all 0-100 percentages)

```python
from superstore import machines, usage

machine = machines(1)[0]
metrics = usage(machine)
```

#### `status(machine, json=False)`

Get status classification for a machine based on usage.

**Parameters:**

- `machine` (dict): Usage dict from `usage()`
- `json` (bool, default=False): JSON format flag

**Returns:** Dict with machine info plus: status ("idle"/"active"/"capacity"), last_update

```python
from superstore import machines, usage, status

machine = machines(1)[0]
metrics = usage(machine)
s = status(metrics)
print(s["status"])  # "idle", "active", or "capacity"
```

#### `jobs(machine, json=False, seed=None)`

Generate active jobs for a machine.

**Parameters:**

- `machine` (dict): Machine dict
- `json` (bool, default=False): JSON format flag
- `seed` (int, optional): Random seed for reproducible generation

**Returns:** List of job dicts with: machine_id, job_id, name, units, start_time, end_time

```python
from superstore import machines, jobs

machine = machines(1)[0]
active_jobs = jobs(machine)
```

### Schema Constants

For type validation and schema definitions:

```python
from superstore import MACHINE_SCHEMA, USAGE_SCHEMA, STATUS_SCHEMA, JOBS_SCHEMA

print(MACHINE_SCHEMA)
# {'machine_id': 'string', 'kind': 'string', 'cores': 'integer', 'region': 'string', 'zone': 'string'}
```

### Pydantic Configuration Classes

For complex configurations with validation, use Pydantic config classes:

```python
from superstore import weather
from superstore.config import (
    WeatherConfig, ClimateZone, OutputFormat,
    # Future: SuperstoreConfig, TimeseriesConfig, CrossfilterConfig
)

# WeatherConfig provides full control with validation
config = WeatherConfig(
    count=2000,
    climate_zone=ClimateZone.TROPICAL,
    latitude=-5.0,
    base_temp_celsius=28.0,
    temp_daily_amplitude=8.0,
    temp_seasonal_amplitude=5.0,
    humidity_temp_correlation=-0.5,
    precipitation_probability=0.25,
    enable_weather_events=True,
    event_probability=0.1,
    outlier_probability=0.02,
    sensor_drift=True,
    sensor_drift_rate=0.002,
    output=OutputFormat.POLARS,
    seed=42,
)

# Pass config directly to generator
df = weather(config)

# Config classes validate parameters at construction time
try:
    bad_config = WeatherConfig(latitude=200.0)  # Raises ValidationError
except ValueError as e:
    print(e)  # "latitude must be between -90 and 90"
```

Available enums:

- `ClimateZone`: TROPICAL, SUBTROPICAL, TEMPERATE, CONTINENTAL, POLAR, ARID, MEDITERRANEAN
- `Season`: SPRING, SUMMER, FALL, WINTER
- `WeatherEvent`: CLEAR, CLOUDY, RAIN, HEAVY_RAIN, SNOW, STORM, HEATWAVE, COLD_SNAP, FOG
- `OutputFormat`: PANDAS, POLARS, DICT
- `MachineType`: CORE, EDGE, WORKER

## Output Format Examples

### Pandas (default)

```python
>>> superstore(count=3)
   Row ID    Order ID  Order Date  ... Quantity Discount  Profit
0       0  83-4132135  2026-01-01  ...      400    98.81  607.21
1       1  91-2252552  2026-01-04  ...      780    50.27  696.44
2       2  12-5610121  2026-01-13  ...      810    20.56  291.17
```

### Polars

```python
>>> superstore(count=3, output="polars")
shape: (3, 19)
┌────────┬─────────────┬────────────┬───────────┬───┐
│ Row ID ┆ Order ID    ┆ Order Date ┆ Ship Date ┆ … │
│ ---    ┆ ---         ┆ ---        ┆ ---       ┆   │
│ i64    ┆ str         ┆ str        ┆ str       ┆   │
╞════════╪═════════════╪════════════╪═══════════╪═══╡
│ 0      ┆ 85-8228070  ┆ 2026-04-29 ┆ 2026-02-11┆ … │
│ 1      ┆ 58-5681064  ┆ 2026-04-11 ┆ 2026-02-11┆ … │
│ 2      ┆ 66-7505483  ┆ 2026-05-23 ┆ 2026-02-11┆ … │
└────────┴─────────────┴────────────┴───────────┴───┘
```

### Dict

```python
>>> superstore(count=2, output="dict")
[{'Row ID': 0, 'Order ID': '52-3453664', 'Order Date': '2026-10-04', ...},
 {'Row ID': 1, 'Order ID': '88-1851475', 'Order Date': '2026-11-20', ...}]
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
python -m pytest superstore/tests -v
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
