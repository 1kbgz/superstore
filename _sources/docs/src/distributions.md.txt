# Statistical Distributions

Sample from various probability distributions for statistical modeling and simulation.

## Overview

The distribution functions provide:

- Standard probability distributions
- Mixture distributions
- Noise injection utilities
- Missing data simulation

## Continuous Distributions

### Uniform Distribution

Sample from a uniform distribution over `[low, high)`:

```python
from superstore import sampleUniform

# 1000 samples between 0 and 1
values = sampleUniform(n=1000)

# Custom range
values = sampleUniform(n=1000, low=10.0, high=50.0)
```

### Normal (Gaussian) Distribution

Sample from a normal distribution:

```python
from superstore import sampleNormal

# Standard normal (mean=0, std=1)
values = sampleNormal(n=1000)

# Custom mean and standard deviation
values = sampleNormal(n=1000, mean=100.0, std=15.0)
```

### Log-Normal Distribution

Sample from a log-normal distribution (useful for prices, sizes, durations):

```python
from superstore import sampleLogNormal

# Default parameters
values = sampleLogNormal(n=1000)

# Custom parameters
values = sampleLogNormal(n=1000, mean=2.0, sigma=0.5)
```

### Exponential Distribution

Sample from an exponential distribution (useful for waiting times):

```python
from superstore import sampleExponential

# Rate parameter lambda=1.0
values = sampleExponential(n=1000, rate=1.0)

# Mean = 1/rate, so rate=0.1 gives mean=10
values = sampleExponential(n=1000, rate=0.1)
```

### Beta Distribution

Sample from a beta distribution (useful for probabilities, proportions):

```python
from superstore import sampleBeta

# Uniform on [0,1]
values = sampleBeta(n=1000, alpha=1.0, beta=1.0)

# Skewed toward 1
values = sampleBeta(n=1000, alpha=5.0, beta=1.0)

# Bell-shaped around 0.5
values = sampleBeta(n=1000, alpha=5.0, beta=5.0)
```

### Gamma Distribution

Sample from a gamma distribution (useful for waiting times, amounts):

```python
from superstore import sampleGamma

values = sampleGamma(n=1000, shape=2.0, scale=1.0)
```

### Weibull Distribution

Sample from a Weibull distribution (useful for failure times, survival analysis):

```python
from superstore import sampleWeibull

values = sampleWeibull(n=1000, shape=1.5, scale=1.0)
```

### Pareto Distribution

Sample from a Pareto distribution (useful for heavy-tailed phenomena):

```python
from superstore import samplePareto

# Classic 80/20 distribution
values = samplePareto(n=1000, alpha=1.16, x_min=1.0)
```

---

## Discrete Distributions

### Poisson Distribution

Sample from a Poisson distribution (useful for count data):

```python
from superstore import samplePoisson

# Mean = 5 events
values = samplePoisson(n=1000, rate=5.0)
```

### Categorical Distribution

Sample from a categorical distribution with specified probabilities:

```python
from superstore import sampleCategorical

# Equal probabilities
categories = ["A", "B", "C", "D"]
values = sampleCategorical(n=1000, categories=categories)

# Custom probabilities
probs = [0.5, 0.3, 0.15, 0.05]
values = sampleCategorical(n=1000, categories=categories, probabilities=probs)
```

---

## Mixture Distributions

### Gaussian Mixture

Sample from a mixture of Gaussian distributions:

```python
from superstore import sampleMixture

# Two-component mixture
values = sampleMixture(
    n=1000,
    means=[0.0, 5.0],
    stds=[1.0, 0.5],
    weights=[0.7, 0.3],  # 70% from first, 30% from second
)

# Three-component mixture (bimodal with outliers)
values = sampleMixture(
    n=1000,
    means=[-2.0, 2.0, 10.0],
    stds=[0.5, 0.5, 0.3],
    weights=[0.45, 0.45, 0.10],
)
```

---

## Noise & Missing Data

### Gaussian Noise

Add Gaussian noise to existing data:

```python
from superstore import addGaussianNoise

# Add noise with std=0.1
noisy_values = addGaussianNoise(values, std=0.1)

# Add proportional noise (10% of value)
noisy_values = addGaussianNoise(values, std=0.1, proportional=True)
```

### Missing Data

Apply missing values (NaN) randomly:

```python
from superstore import applyMissing

# 5% missing values
data_with_missing = applyMissing(values, missing_rate=0.05)

# Different missing rates for multiple columns
df = applyMissing(df, missing_rate={"col1": 0.1, "col2": 0.05})
```

---

## Examples

### Quality Score Distribution

```python
from superstore import sampleBeta

# Quality scores skewed toward high values
scores = sampleBeta(n=10000, alpha=8.0, beta=2.0)
# Mode ≈ 0.78, most values between 0.6-1.0
```

### Customer Lifetime Value

```python
from superstore import sampleLogNormal

# CLV with long tail
clv = sampleLogNormal(n=10000, mean=5.0, sigma=1.0)
# Median ≈ exp(5) ≈ 148, with some very high values
```

### Service Response Times

```python
from superstore import sampleMixture

# Bimodal: fast responses + slow (queued) responses
response_ms = sampleMixture(
    n=10000,
    means=[50.0, 500.0],
    stds=[20.0, 100.0],
    weights=[0.9, 0.1],
)
```

### Event Counts with Noise

```python
from superstore import samplePoisson, addGaussianNoise

# Poisson counts with measurement noise
counts = samplePoisson(n=1000, rate=20.0)
noisy_counts = addGaussianNoise(counts, std=2.0)
```

---

## API Reference

See the full [API Reference](api.md) for all distribution functions.
