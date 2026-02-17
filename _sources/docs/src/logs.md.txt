# Log Data Generation

Generate realistic web server access logs and application event logs with configurable traffic patterns.

## Overview

The log generators create data suitable for:

- Log analytics dashboards
- Error rate monitoring
- Latency analysis
- Security analysis and fraud detection
- Observability pipeline testing

## Web Server Logs

The `logs()` function generates HTTP access logs in Apache Combined or Common format.

### Basic Usage

```python
from superstore import logs

# Generate 10,000 log entries
df = logs(count=10000)

# Apache Combined format (default)
df = logs(count=10000, format="combined")

# JSON structured logs
df = logs(count=10000, format="json")
```

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Request timestamp |
| `ip_address` | str | Client IP address |
| `method` | str | HTTP method (GET, POST, etc.) |
| `path` | str | Request path |
| `status_code` | int | HTTP status code |
| `response_size` | int | Response size in bytes |
| `latency_ms` | float | Request latency in milliseconds |
| `user_agent` | str | User agent string |
| `referer` | str | Referrer URL |
| `user_id` | str | User identifier (if authenticated) |

---

## Application Logs

The `app_logs()` function generates application-level event logs with log levels, trace IDs, and exceptions.

### Basic Usage

```python
from superstore import app_logs

# Generate 5,000 application log entries
df = app_logs(count=5000)
```

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Event timestamp |
| `level` | str | Log level (DEBUG, INFO, WARN, ERROR) |
| `logger` | str | Logger name/component |
| `message` | str | Log message |
| `trace_id` | str | Distributed trace ID |
| `span_id` | str | Span ID |
| `exception` | str | Exception type (if error) |
| `stack_trace` | str | Stack trace (if error) |

---

## Configuration

Use `LogsConfig` for detailed control over log generation:

```python
from superstore import logs, LogsConfig

config = LogsConfig(
    count=50000,
    seed=42,
    format="combined",
)
df = logs(config=config)
```

### Traffic Patterns

Control the traffic rate and timing:

```python
config = LogsConfig(
    count=10000,
    start_time="2024-01-15T10:00:00",  # ISO format start time
    requests_per_second=250.0,          # Average RPS (Poisson arrival)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_time` | (current time) | Start timestamp in ISO format |
| `requests_per_second` | `100.0` | Average requests per second |

### Status Code Distribution

Configure success and error rates:

```python
config = LogsConfig(
    count=10000,
    success_rate=0.98,  # 98% success (2xx responses)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `success_rate` | `0.95` | Base success rate (2xx responses) |

### Error Bursts

Simulate error bursts for monitoring/alerting demos:

```python
config = LogsConfig(
    count=50000,
    error_burst={
        "enable": True,
        "burst_probability": 0.03,      # 3% chance of entering burst
        "burst_duration_seconds": 45,    # Average burst duration
        "burst_error_rate": 0.6,         # 60% errors during burst
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable` | `True` | Enable error burst simulation |
| `burst_probability` | `0.02` | Probability of entering burst state |
| `burst_duration_seconds` | `30` | Average burst duration |
| `burst_error_rate` | `0.5` | Error rate during bursts |

### Latency Distribution

Configure request latency behavior:

```python
config = LogsConfig(
    count=10000,
    latency={
        "base_latency_ms": 60.0,        # Median latency
        "latency_stddev": 0.9,          # Log-normal spread
        "slow_request_probability": 0.08,  # 8% slow requests
        "slow_request_multiplier": 15.0,   # Slow = 15x base
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_latency_ms` | `50.0` | Base/median latency in milliseconds |
| `latency_stddev` | `0.8` | Standard deviation (log-normal) |
| `slow_request_probability` | `0.05` | Probability of slow requests |
| `slow_request_multiplier` | `10.0` | Multiplier for slow request latency |

### Request Details

Customize request generation:

```python
config = LogsConfig(
    count=10000,
    include_user_agent=True,   # Include user agent strings
    unique_ips=2000,           # Number of unique client IPs
    unique_users=800,          # Number of unique user IDs
    api_path_ratio=0.8,        # 80% API paths, 20% static
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `include_user_agent` | `True` | Include user agent strings |
| `unique_ips` | `1000` | Number of unique IP addresses |
| `unique_users` | `500` | Number of unique user IDs |
| `api_path_ratio` | `0.7` | Ratio of API vs static paths |

### Complete Example

```python
from superstore import logs, LogsConfig

config = LogsConfig(
    count=100000,
    seed=42,
    format="json",

    # Traffic
    start_time="2024-06-01T00:00:00",
    requests_per_second=500.0,

    # Success rate
    success_rate=0.97,

    # Error bursts for monitoring demos
    error_burst={
        "enable": True,
        "burst_probability": 0.02,
        "burst_duration_seconds": 60,
        "burst_error_rate": 0.7,
    },

    # Latency
    latency={
        "base_latency_ms": 45.0,
        "latency_stddev": 0.7,
        "slow_request_probability": 0.05,
        "slow_request_multiplier": 20.0,
    },

    # Request details
    unique_ips=5000,
    unique_users=2000,
    api_path_ratio=0.85,
)

df = logs(config=config)
```

---

## API Reference

See the full API documentation:

- [logs()](api.md)
- [app_logs()](api.md)
- [LogsConfig](api.md)
