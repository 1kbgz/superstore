# Weather Data Generation

Generate realistic outdoor weather sensor data with temporal patterns, seasonal variations, and weather events.

## Overview

The weather generator creates time-series sensor data suitable for:

- IoT sensor analytics
- Climate data visualization
- Anomaly detection training
- Time series forecasting demos
- Environmental monitoring dashboards

## Basic Usage

```python
from superstore import weather

# Generate 1000 weather readings
df = weather(count=1000)

# Different output formats
df = weather(count=1000, output="polars")
data = weather(count=1000, output="dict")
```

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Reading timestamp |
| `temperature` | float | Temperature in Celsius |
| `humidity` | float | Relative humidity (%) |
| `pressure` | float | Barometric pressure (hPa) |
| `wind_speed` | float | Wind speed (m/s) |
| `wind_direction` | float | Wind direction (degrees) |
| `precipitation` | float | Precipitation (mm) |
| `uv_index` | float | UV index |
| `weather_event` | str | Current weather event (if enabled) |

## Configuration

Use `WeatherConfig` for detailed control:

```python
from superstore import weather, WeatherConfig

config = WeatherConfig(
    count=5000,
    seed=42,
    output="pandas",
)
df = weather(config=config)
```

### Temporal Settings

Control the time span and frequency of readings:

```python
config = WeatherConfig(
    count=10000,
    start_date="2024-01-01",   # Start date (YYYY-MM-DD)
    frequency_minutes=15,       # Reading every 15 minutes
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_date` | (30 days ago) | Start date in YYYY-MM-DD format |
| `frequency_minutes` | `15` | Minutes between readings (1-1440) |

### Climate Zone

Set the climate zone for realistic regional patterns:

```python
from superstore import weather, WeatherConfig, ClimateZone

config = WeatherConfig(
    count=5000,
    climate_zone=ClimateZone.TROPICAL,  # or "tropical"
    latitude=25.0,  # Affects day/night calculations
)
```

Available climate zones:

| Zone | Description |
|------|-------------|
| `tropical` | Hot and humid year-round |
| `subtropical` | Hot summers, mild winters |
| `temperate` | Moderate with distinct seasons |
| `continental` | Large seasonal temperature swings |
| `polar` | Cold year-round |
| `arid` | Hot and dry |
| `mediterranean` | Dry summers, wet winters |

### Temperature Settings

Fine-tune temperature patterns:

```python
config = WeatherConfig(
    count=5000,

    # Base temperature
    base_temp_celsius=20.0,        # Annual average

    # Daily cycle
    temp_daily_amplitude=12.0,     # Day/night swing

    # Seasonal cycle
    temp_seasonal_amplitude=18.0,  # Summer/winter swing

    # Noise
    temp_noise_stddev=2.5,         # Random variation
)
```

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `base_temp_celsius` | `15.0` | -50 to 50 | Annual average temperature |
| `temp_daily_amplitude` | `10.0` | 0 to 30 | Day/night temperature swing |
| `temp_seasonal_amplitude` | `15.0` | 0 to 40 | Summer/winter temperature swing |
| `temp_noise_stddev` | `2.0` | 0 to 10 | Random noise standard deviation |

### Humidity Settings

Control humidity patterns:

```python
config = WeatherConfig(
    count=5000,
    base_humidity_percent=65.0,       # Average humidity
    humidity_temp_correlation=-0.4,   # Negative = high temp → low humidity
)
```

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `base_humidity_percent` | `60.0` | 0 to 100 | Average relative humidity |
| `humidity_temp_correlation` | `-0.3` | -1 to 1 | Temperature-humidity correlation |

### Precipitation

Configure precipitation probability:

```python
config = WeatherConfig(
    count=5000,
    precipitation_probability=0.20,  # 20% chance of precipitation
)
```

### Weather Events

Enable discrete weather events:

```python
config = WeatherConfig(
    count=5000,
    enable_weather_events=True,
    event_probability=0.08,  # 8% chance per reading
)
```

Weather events include: `clear`, `cloudy`, `rain`, `heavy_rain`, `snow`, `storm`, `heatwave`, `cold_snap`, `fog`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_weather_events` | `True` | Enable weather event simulation |
| `event_probability` | `0.05` | Probability of event per reading |

### Sensor Characteristics

Simulate sensor imperfections:

```python
config = WeatherConfig(
    count=5000,

    # Outliers (sensor errors)
    outlier_probability=0.02,    # 2% chance of bad readings

    # Sensor drift
    sensor_drift=True,           # Enable calibration drift
    sensor_drift_rate=0.002,     # Drift rate per reading
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `outlier_probability` | `0.01` | Probability of outlier readings |
| `sensor_drift` | `False` | Enable gradual sensor drift |
| `sensor_drift_rate` | `0.001` | Rate of drift per reading |

### Complete Example

```python
from superstore import weather, WeatherConfig

config = WeatherConfig(
    count=10000,
    seed=42,

    # Time settings
    start_date="2024-01-01",
    frequency_minutes=30,

    # Location
    climate_zone="mediterranean",
    latitude=37.0,

    # Temperature
    base_temp_celsius=18.0,
    temp_daily_amplitude=10.0,
    temp_seasonal_amplitude=12.0,

    # Humidity
    base_humidity_percent=55.0,
    humidity_temp_correlation=-0.4,

    # Precipitation
    precipitation_probability=0.10,

    # Events
    enable_weather_events=True,
    event_probability=0.05,

    # Sensor quality
    outlier_probability=0.01,
    sensor_drift=True,
)

df = weather(config=config)
```

---

## API Reference

See the full API documentation:

- [weather()](api.md)
- [WeatherConfig](api.md)
