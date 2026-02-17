# API Reference

Full API reference for all public functions and classes.

For detailed guides with examples, see:

- [Retail Data](retail.md) - `superstore()`, `employees()`
- [Time Series](timeseries.md) - `timeseries()`
- [Weather](weather.md) - `weather()`
- [Logs](logs.md) - `logs()`, `app_logs()`
- [Finance](finance.md) - `stock_prices()`, `options_chain()`, `finance()`
- [Telemetry](telemetry.md) - `telemetry()`, crossfilter functions
- [Distributions](distributions.md) - `sample*()` functions
- [Copulas](copulas.md) - copula classes
- [Temporal Models](temporal.md) - `AR1`, `MarkovChain`, `RandomWalk`

---

## Data Generators

```{eval-rst}
.. autofunction:: superstore.superstore
.. autofunction:: superstore.employees
.. autofunction:: superstore.timeseries
.. autofunction:: superstore.weather
.. autofunction:: superstore.logs
.. autofunction:: superstore.app_logs
.. autofunction:: superstore.stock_prices
.. autofunction:: superstore.options_chain
.. autofunction:: superstore.finance
.. autofunction:: superstore.telemetry
.. autofunction:: superstore.machines
.. autofunction:: superstore.usage
.. autofunction:: superstore.status
.. autofunction:: superstore.jobs
```

---

## Streaming & Parallel

```{eval-rst}
.. autofunction:: superstore.superstoreStream
.. autofunction:: superstore.employeesStream
.. autofunction:: superstore.superstoreParallel
.. autofunction:: superstore.employeesParallel
.. autofunction:: superstore.numThreads
.. autofunction:: superstore.setNumThreads
```

---

## Export Functions

```{eval-rst}
.. autofunction:: superstore.superstoreArrowIpc
.. autofunction:: superstore.employeesArrowIpc
.. autofunction:: superstore.superstoreToParquet
.. autofunction:: superstore.employeesToParquet
.. autofunction:: superstore.superstoreToCsv
.. autofunction:: superstore.employeesToCsv
```

---

## Distributions

```{eval-rst}
.. autofunction:: superstore.sampleUniform
.. autofunction:: superstore.sampleNormal
.. autofunction:: superstore.sampleLogNormal
.. autofunction:: superstore.sampleExponential
.. autofunction:: superstore.sampleBeta
.. autofunction:: superstore.sampleGamma
.. autofunction:: superstore.sampleWeibull
.. autofunction:: superstore.samplePareto
.. autofunction:: superstore.samplePoisson
.. autofunction:: superstore.sampleCategorical
.. autofunction:: superstore.sampleMixture
.. autofunction:: superstore.addGaussianNoise
.. autofunction:: superstore.applyMissing
```

---

## Correlation & Copulas

```{eval-rst}
.. autofunction:: superstore.pearsonCorrelation
.. autofunction:: superstore.sampleBivariate

.. autoclass:: superstore.GaussianCopula
   :members:

.. autoclass:: superstore.ClaytonCopula
   :members:

.. autoclass:: superstore.FrankCopula
   :members:

.. autoclass:: superstore.GumbelCopula
   :members:
```

---

## Temporal Models

```{eval-rst}
.. autoclass:: superstore.AR1
   :members:

.. autoclass:: superstore.ARp
   :members:

.. autoclass:: superstore.MarkovChain
   :members:

.. autoclass:: superstore.RandomWalk
   :members:

.. autoclass:: superstore.ExponentialSmoothing
   :members:
```

---

## Configuration Classes

```{eval-rst}
.. autopydantic_model:: superstore.SuperstoreConfig
   :members:

.. autopydantic_model:: superstore.TimeseriesConfig
   :members:

.. autopydantic_model:: superstore.WeatherConfig
   :members:

.. autopydantic_model:: superstore.LogsConfig
   :members:

.. autopydantic_model:: superstore.FinanceConfig
   :members:

.. autopydantic_model:: superstore.CrossfilterConfig
   :members:
```

---

## Enums

```{eval-rst}
.. autoclass:: superstore.ClimateZone
   :members:
   :undoc-members:

.. autoclass:: superstore.OutputFormat
   :members:
   :undoc-members:

.. autoclass:: superstore.LogLevel
   :members:
   :undoc-members:

.. autoclass:: superstore.LogFormat
   :members:
   :undoc-members:
```
