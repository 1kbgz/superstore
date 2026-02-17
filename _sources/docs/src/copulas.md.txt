# Correlation & Copulas

Generate correlated multivariate data using copula models.

## Overview

Copulas allow you to:

- Generate correlated random variables
- Control the dependence structure independently from marginals
- Model different types of dependence (tail dependence, asymmetric)
- Create realistic multivariate distributions

## Correlation Functions

### Pearson Correlation

Compute Pearson correlation between two arrays:

```python
from superstore import pearsonCorrelation

correlation = pearsonCorrelation(x, y)
```

### Bivariate Sampling

Generate correlated bivariate samples:

```python
from superstore import sampleBivariate

# Generate correlated pairs with rho=0.7
x, y = sampleBivariate(n=1000, rho=0.7)
```

---

## Copula Models

### Gaussian Copula

The Gaussian copula creates correlation through a multivariate normal distribution. It has **no tail dependence** - extreme observations are not more likely to occur together.

```python
from superstore import GaussianCopula

# Create a Gaussian copula with correlation matrix
copula = GaussianCopula(
    correlation=[[1.0, 0.7, 0.3],
                 [0.7, 1.0, 0.5],
                 [0.3, 0.5, 1.0]]
)

# Generate 1000 correlated uniform samples
u = copula.sample(n=1000)
# u has shape (1000, 3), each column is marginally Uniform(0,1)
```

**Properties:**
- Symmetric dependence
- No tail dependence
- Easy to parameterize with correlation matrix
- Good for "normal" dependencies

### Clayton Copula

The Clayton copula has **lower tail dependence** - extreme low values are more likely to occur together. Useful for:

- Credit risk (joint defaults)
- Insurance (correlated claims)
- Portfolio risk (market crashes)

```python
from superstore import ClaytonCopula

# Create a Clayton copula with theta=2.0
# Higher theta = stronger dependence
copula = ClaytonCopula(theta=2.0, dim=3)

# Generate samples
u = copula.sample(n=1000)
```

**Properties:**
- Asymmetric dependence
- Lower tail dependence (crashes happen together)
- No upper tail dependence
- theta > 0 for positive dependence

### Frank Copula

The Frank copula has **no tail dependence** but can model both positive and negative dependence:

```python
from superstore import FrankCopula

# Positive dependence
copula = FrankCopula(theta=5.0, dim=2)

# Negative dependence
copula = FrankCopula(theta=-5.0, dim=2)

u = copula.sample(n=1000)
```

**Properties:**
- Symmetric dependence
- No tail dependence
- Can model negative dependence (theta < 0)
- Good for weak dependencies

### Gumbel Copula

The Gumbel copula has **upper tail dependence** - extreme high values are more likely to occur together. Useful for:

- Flood modeling (extreme rainfall)
- Insurance (extreme losses)
- Finance (market bubbles)

```python
from superstore import GumbelCopula

# Create a Gumbel copula with theta=3.0
# theta >= 1, higher = stronger dependence
copula = GumbelCopula(theta=3.0, dim=2)

u = copula.sample(n=1000)
```

**Properties:**
- Asymmetric dependence
- Upper tail dependence (booms happen together)
- No lower tail dependence
- theta >= 1

---

## Choosing a Copula

| Copula | Lower Tail | Upper Tail | Use Case |
|--------|------------|------------|----------|
| Gaussian | No | No | General correlation |
| Clayton | Yes | No | Joint crashes, defaults |
| Frank | No | No | Weak/negative dependence |
| Gumbel | No | Yes | Joint extremes (high) |

---

## Combining Copulas with Marginals

Copulas generate uniform marginals. Transform to desired distributions:

```python
from superstore import GaussianCopula, sampleNormal
from scipy.stats import norm, lognorm

# Generate correlated uniforms
copula = GaussianCopula(correlation=[[1.0, 0.8], [0.8, 1.0]])
u = copula.sample(n=10000)

# Transform to different marginal distributions
x = norm.ppf(u[:, 0], loc=0, scale=1)      # Standard normal
y = lognorm.ppf(u[:, 1], s=0.5, scale=100)  # Log-normal

# x and y are now correlated with different marginals
```

---

## Examples

### Correlated Asset Returns

```python
from superstore import GaussianCopula
from scipy.stats import t

# Create correlated returns for 4 assets
correlation = [
    [1.0, 0.6, 0.3, 0.2],
    [0.6, 1.0, 0.4, 0.3],
    [0.3, 0.4, 1.0, 0.5],
    [0.2, 0.3, 0.5, 1.0],
]
copula = GaussianCopula(correlation=correlation)
u = copula.sample(n=252)

# Transform to Student-t returns (fat tails)
import numpy as np
returns = t.ppf(u, df=5) * 0.02  # 2% daily vol
```

### Credit Risk Defaults

```python
from superstore import ClaytonCopula

# Strong lower tail dependence for joint defaults
copula = ClaytonCopula(theta=3.0, dim=5)
u = copula.sample(n=10000)

# Transform to default indicators
default_threshold = 0.03  # 3% default probability
defaults = u < default_threshold  # Boolean array
```

### Insurance Claims

```python
from superstore import GumbelCopula
from scipy.stats import pareto

# Upper tail dependence for extreme claims
copula = GumbelCopula(theta=2.5, dim=3)
u = copula.sample(n=5000)

# Transform to Pareto claims
claims = pareto.ppf(u, b=2.0, scale=10000)
```

---

## API Reference

See the full [API Reference](api.md) for all copula classes.
