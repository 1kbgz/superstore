# Telemetry & IoT Data Generation

Generate machine telemetry and IoT sensor data for dashboard and observability demos.

## Overview

The telemetry generators create data suitable for:

- DevOps/SRE dashboards
- Anomaly detection systems
- Capacity planning demos
- Infrastructure monitoring
- Crossfilter visualization

## Telemetry

The `telemetry()` function generates comprehensive machine metrics with configurable scenarios.

### Basic Usage

```python
from superstore import telemetry

# Generate telemetry with default settings
df = telemetry(n_machines=50, n_readings=1000)

# Use a preset scenario
df = telemetry(scenario="anomaly_detection", n_machines=100)
```

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Reading timestamp |
| `machine_id` | str | Machine identifier |
| `machine_type` | str | Machine type (core, edge, worker) |
| `zone` | str | Zone identifier |
| `region` | str | Region identifier |
| `cpu_percent` | float | CPU utilization (0-100) |
| `memory_percent` | float | Memory utilization (0-100) |
| `disk_percent` | float | Disk utilization (0-100) |
| `network_in_mbps` | float | Network ingress (Mbps) |
| `network_out_mbps` | float | Network egress (Mbps) |
| `request_count` | int | Request count |
| `error_count` | int | Error count |
| `latency_p50` | float | 50th percentile latency (ms) |
| `latency_p99` | float | 99th percentile latency (ms) |
| `is_anomaly` | bool | Anomaly label (for ML training) |

### Preset Scenarios

Use preset scenarios for common use cases:

```python
from superstore import telemetry, TELEMETRY_SCENARIOS

# See available scenarios
print(TELEMETRY_SCENARIOS)

# Use a scenario
df = telemetry(scenario="production", n_machines=100)
```

| Scenario | Description |
|----------|-------------|
| `baseline` | Normal operating conditions |
| `maintenance_window` | Scheduled maintenance patterns |
| `capacity_planning` | Growth trend data |
| `anomaly_detection` | Training data with labeled anomalies |
| `multi_zone` | Multi-datacenter deployment |
| `cpu_bound` | High CPU workload |
| `memory_bound` | High memory workload |
| `network_heavy` | Network-intensive workload |
| `degradation_cycle` | Progressive degradation and recovery |
| `production` | Full realistic environment |
| `chaos` | High anomaly rates for chaos engineering |

---

## Crossfilter Data

For dashboard demos, use the individual crossfilter generators:

### machines()

Generate machine metadata:

```python
from superstore import machines

df = machines(n_machines=100)
```

| Column | Type | Description |
|--------|------|-------------|
| `machine_id` | str | Machine identifier |
| `machine_type` | str | Machine type |
| `cores` | int | CPU cores |
| `memory_gb` | int | Memory in GB |
| `zone` | str | Zone |
| `region` | str | Region |
| `created_at` | datetime | Provisioning date |

### usage()

Generate machine usage metrics:

```python
from superstore import usage

df = usage(n_machines=100, n_readings=1000)
```

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Reading timestamp |
| `machine_id` | str | Machine identifier |
| `cpu_percent` | float | CPU utilization |
| `memory_percent` | float | Memory utilization |
| `disk_percent` | float | Disk utilization |

### status()

Generate machine status records:

```python
from superstore import status

df = status(n_machines=100)
```

| Column | Type | Description |
|--------|------|-------------|
| `machine_id` | str | Machine identifier |
| `status` | str | Current status (healthy, degraded, down) |
| `last_heartbeat` | datetime | Last heartbeat time |

### jobs()

Generate job/task records:

```python
from superstore import jobs

df = jobs(n_machines=100, n_jobs=5000)
```

| Column | Type | Description |
|--------|------|-------------|
| `job_id` | str | Job identifier |
| `machine_id` | str | Executing machine |
| `job_type` | str | Job type |
| `status` | str | Job status |
| `start_time` | datetime | Job start time |
| `end_time` | datetime | Job end time |
| `duration_seconds` | float | Job duration |

---

## Configuration

Use `CrossfilterConfig` for detailed control:

```python
from superstore import telemetry, CrossfilterConfig

config = CrossfilterConfig(
    n_machines=200,
    n_readings=2000,
    seed=42,
)
df = telemetry(config=config)
```

### Machine Configuration

Configure the machine fleet:

```python
config = CrossfilterConfig(
    n_machines=100,

    # Machine types
    machine_types=["core", "edge", "worker"],

    # Hardware specs
    cores_range=(8, 128),  # CPU cores range

    # Topology
    zones=["zone-a", "zone-b", "zone-c", "zone-d"],
    regions=["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"],
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `machine_types` | `[core, edge, worker]` | Types of machines |
| `cores_range` | `(4, 64)` | CPU cores range |
| `zones` | `[zone-a, zone-b, zone-c]` | Available zones |
| `regions` | `[us-east-1, us-west-2, eu-west-1]` | Available regions |

### Usage Profiles

Configure baseline resource utilization:

```python
config = CrossfilterConfig(
    n_machines=100,
    n_readings=1000,

    base_cpu_load=0.4,        # 40% base CPU
    base_memory_load=0.6,     # 60% base memory
    load_variance=0.25,       # Load variability
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_cpu_load` | `0.3` | Base CPU utilization (0-1) |
| `base_memory_load` | `0.5` | Base memory utilization (0-1) |
| `load_variance` | `0.2` | Variance in load readings |

### Anomaly Injection

Inject anomalies for detection training:

```python
config = CrossfilterConfig(
    n_machines=100,
    n_readings=2000,

    anomalies={
        "enable": True,
        "cpu_spike_probability": 0.03,       # 3% CPU spikes
        "memory_leak_probability": 0.015,    # Memory leak starts
        "network_saturation_probability": 0.02,  # Network saturation
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable` | `False` | Enable anomaly injection |
| `cpu_spike_probability` | `0.02` | Probability of CPU spike |
| `memory_leak_probability` | `0.01` | Probability of memory leak |
| `network_saturation_probability` | `0.01` | Probability of network saturation |

### Temporal Patterns

Add realistic time-of-day and day-of-week patterns:

```python
config = CrossfilterConfig(
    n_machines=100,
    n_readings=5000,

    temporal_patterns={
        "enable_diurnal": True,    # Day/night patterns
        "enable_weekly": True,     # Weekday/weekend patterns
        "peak_hour": 14,           # Peak at 2 PM
        "night_load_factor": 0.25, # 25% load at night
        "weekend_load_factor": 0.4, # 40% load on weekends
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_diurnal` | `False` | Enable day/night load patterns |
| `enable_weekly` | `False` | Enable weekday/weekend patterns |
| `peak_hour` | `14` | Hour of peak load (0-23) |
| `night_load_factor` | `0.3` | Load factor during night |
| `weekend_load_factor` | `0.5` | Load factor during weekends |

### Failure Simulation

Simulate machine failures and cascades:

```python
config = CrossfilterConfig(
    n_machines=100,
    n_readings=2000,

    enable_failures=True,
    failure_probability=0.002,       # 0.2% failure rate per reading
    cascade_failure_probability=0.4, # 40% chance of cascade
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_failures` | `False` | Enable failure simulation |
| `failure_probability` | `0.001` | Per-reading failure probability |
| `cascade_failure_probability` | `0.3` | Cascade probability when dependent fails |

### Complete Example

```python
from superstore import telemetry, CrossfilterConfig

config = CrossfilterConfig(
    n_machines=200,
    n_readings=3000,
    seed=42,

    # Machine topology
    machine_types=["core", "edge", "worker"],
    cores_range=(8, 96),
    zones=["zone-a", "zone-b", "zone-c"],
    regions=["us-east-1", "us-west-2", "eu-west-1"],

    # Usage profiles
    base_cpu_load=0.35,
    base_memory_load=0.55,
    load_variance=0.2,

    # Anomalies
    anomalies={
        "enable": True,
        "cpu_spike_probability": 0.02,
        "memory_leak_probability": 0.01,
        "network_saturation_probability": 0.01,
    },

    # Temporal patterns
    temporal_patterns={
        "enable_diurnal": True,
        "enable_weekly": True,
        "peak_hour": 15,
        "night_load_factor": 0.2,
        "weekend_load_factor": 0.35,
    },

    # Failures
    enable_failures=True,
    failure_probability=0.001,
    cascade_failure_probability=0.25,
)

df = telemetry(config=config)
```

---

## Schemas

Access schema constants for validation:

```python
from superstore import (
    MACHINE_SCHEMA,
    USAGE_SCHEMA,
    STATUS_SCHEMA,
    JOBS_SCHEMA,
    TELEMETRY_SCHEMA,
    TELEMETRY_SCENARIOS,
)
```

---

## API Reference

See the full API documentation:

- [telemetry()](api.md)
- [machines()](api.md)
- [usage()](api.md)
- [status()](api.md)
- [jobs()](api.md)
- [CrossfilterConfig](api.md)
