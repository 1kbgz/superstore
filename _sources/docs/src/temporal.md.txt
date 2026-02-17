# Temporal Models

Time-dependent data generation with autoregressive processes, Markov chains, and random walks.

## Overview

Temporal models provide:

- Autoregressive (AR) processes for correlated time series
- Markov chains for state-based dynamics
- Random walks for price-like processes
- Exponential smoothing for trend/seasonality

---

## AR(1) Process

The `AR1` class generates a first-order autoregressive process:

$$X_t = \phi X_{t-1} + \epsilon_t$$

where $\phi$ is the persistence parameter and $\epsilon_t$ is white noise.

```python
from superstore import AR1

# Create AR(1) with high persistence
ar = AR1(phi=0.95, sigma=1.0)

# Generate time series
values = ar.generate(n=500)

# Generate with initial value
values = ar.generate(n=500, x0=10.0)
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `phi` | Persistence parameter (-1 < φ < 1 for stationarity) |
| `sigma` | Standard deviation of innovations |

**Properties:**

- `phi` close to 1: Highly persistent, slow mean reversion
- `phi` close to 0: Low persistence, rapid mean reversion
- `phi` negative: Oscillating behavior
- `|phi| >= 1`: Non-stationary (explosive or unit root)

---

## AR(p) Process

The `ARp` class generates a higher-order autoregressive process:

$$X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + ... + \phi_p X_{t-p} + \epsilon_t$$

```python
from superstore import ARp

# Create AR(3) process
ar = ARp(phi=[0.5, 0.3, 0.1], sigma=1.0)

# Generate time series
values = ar.generate(n=500)
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `phi` | List of AR coefficients [φ₁, φ₂, ..., φₚ] |
| `sigma` | Standard deviation of innovations |

**Usage:**

```python
# AR(2) with cyclical behavior
ar = ARp(phi=[0.6, -0.3], sigma=1.0)
values = ar.generate(n=1000)
```

---

## Markov Chain

The `MarkovChain` class generates sequences from a discrete-state Markov chain:

```python
from superstore import MarkovChain

# Define transition matrix (rows must sum to 1)
# State 0: Bull market, State 1: Bear market
transition_matrix = [
    [0.95, 0.05],  # Bull → Bull 95%, Bull → Bear 5%
    [0.10, 0.90],  # Bear → Bull 10%, Bear → Bear 90%
]

mc = MarkovChain(transition_matrix=transition_matrix)

# Generate state sequence
states = mc.generate(n=1000, initial_state=0)
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `transition_matrix` | Square matrix P where P[i,j] = P(state j \| state i) |

**Named states:**

```python
# Three-state regime model
states = ["low_vol", "normal", "high_vol"]
transition_matrix = [
    [0.90, 0.08, 0.02],  # low → low, normal, high
    [0.05, 0.90, 0.05],  # normal → ...
    [0.03, 0.17, 0.80],  # high → ...
]

mc = MarkovChain(
    transition_matrix=transition_matrix,
    state_names=states,
)

# Returns strings instead of integers
regimes = mc.generate(n=500, initial_state="normal")
```

**Stationary distribution:**

```python
# Get long-run state probabilities
stationary = mc.stationary_distribution()
```

---

## Random Walk

The `RandomWalk` class generates random walk processes (cumulative sum of innovations):

$$X_t = X_{t-1} + \mu + \epsilon_t$$

```python
from superstore import RandomWalk

# Simple random walk
rw = RandomWalk(drift=0.0, sigma=1.0)
values = rw.generate(n=500)

# Random walk with positive drift (trending up)
rw = RandomWalk(drift=0.001, sigma=0.02)
prices = rw.generate(n=252, x0=100.0)
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `drift` | Drift/trend per step (μ) |
| `sigma` | Standard deviation of innovations |

**Geometric random walk (for prices):**

```python
import numpy as np

# Generate log-price random walk
rw = RandomWalk(drift=0.0003, sigma=0.015)
log_prices = rw.generate(n=252, x0=0.0)

# Convert to prices
prices = 100 * np.exp(log_prices)
```

---

## Exponential Smoothing

The `ExponentialSmoothing` class generates smoothed time series with optional trend and seasonality:

```python
from superstore import ExponentialSmoothing

# Simple exponential smoothing
es = ExponentialSmoothing(alpha=0.3)
smoothed = es.smooth(values)

# Holt-Winters with trend and seasonality
es = ExponentialSmoothing(
    alpha=0.3,      # Level smoothing
    beta=0.1,       # Trend smoothing
    gamma=0.1,      # Seasonal smoothing
    period=12,      # Seasonal period
)
smoothed = es.smooth(values)
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `alpha` | Level smoothing (0 < α < 1) |
| `beta` | Trend smoothing (0 < β < 1), optional |
| `gamma` | Seasonal smoothing (0 < γ < 1), optional |
| `period` | Seasonal period (required if gamma set) |

---

## Examples

### Mean-Reverting Price

```python
from superstore import AR1
import numpy as np

# Price that mean-reverts to 100
ar = AR1(phi=0.98, sigma=2.0)
deviations = ar.generate(n=500)
prices = 100 + deviations
```

### Regime-Switching Volatility

```python
from superstore import MarkovChain, sampleNormal

# Two volatility regimes
mc = MarkovChain(
    transition_matrix=[[0.95, 0.05], [0.10, 0.90]],
    state_names=["low_vol", "high_vol"],
)
regimes = mc.generate(n=500, initial_state="low_vol")

# Generate returns with regime-dependent volatility
volatilities = {"low_vol": 0.01, "high_vol": 0.03}
returns = [sampleNormal(n=1, std=volatilities[r])[0] for r in regimes]
```

### Stock Price with Drift

```python
from superstore import RandomWalk
import numpy as np

# Geometric Brownian Motion approximation
# Annual drift 8%, volatility 20%
daily_drift = 0.08 / 252
daily_vol = 0.20 / np.sqrt(252)

rw = RandomWalk(drift=daily_drift, sigma=daily_vol)
log_returns = rw.generate(n=252, x0=0.0)
prices = 100 * np.exp(log_returns)
```

### Seasonal Time Series

```python
from superstore import ExponentialSmoothing, sampleNormal
import numpy as np

# Create noisy seasonal data
t = np.arange(365)
seasonal = 10 * np.sin(2 * np.pi * t / 365)
trend = 0.05 * t
noise = sampleNormal(n=365, std=3.0)
values = 50 + trend + seasonal + noise

# Apply Holt-Winters smoothing
es = ExponentialSmoothing(alpha=0.2, beta=0.05, gamma=0.1, period=365)
smoothed = es.smooth(values)
```

---

## API Reference

See the full [API Reference](api.md) for all temporal model classes.
