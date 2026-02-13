"""Type stubs for superstore - a library for realistic data generation."""

from typing import Any, Literal, final, overload

import pandas as pd
import polars as pl

# =============================================================================
# Schema constants
# =============================================================================

MACHINE_SCHEMA: dict[str, str]
USAGE_SCHEMA: dict[str, str]
STATUS_SCHEMA: dict[str, str]
JOBS_SCHEMA: dict[str, str]

# =============================================================================
# Core generators
# =============================================================================

@overload
def superstore(
    count: int = ...,
    output: Literal["pandas"] = ...,
    seed: int | None = ...,
) -> pd.DataFrame: ...
@overload
def superstore(
    count: int = ...,
    *,
    output: Literal["polars"],
    seed: int | None = ...,
) -> pl.DataFrame: ...
@overload
def superstore(
    count: int = ...,
    *,
    output: Literal["dict"],
    seed: int | None = ...,
) -> list[dict[str, Any]]: ...
@overload
def employees(
    count: int = ...,
    output: Literal["pandas"] = ...,
    seed: int | None = ...,
) -> pd.DataFrame: ...
@overload
def employees(
    count: int = ...,
    *,
    output: Literal["polars"],
    seed: int | None = ...,
) -> pl.DataFrame: ...
@overload
def employees(
    count: int = ...,
    *,
    output: Literal["dict"],
    seed: int | None = ...,
) -> list[dict[str, Any]]: ...

# =============================================================================
# Time series generators
# =============================================================================

@overload
def timeseries(
    nper: int = ...,
    freq: str = ...,
    ncol: int = ...,
    output: Literal["pandas"] = ...,
    seed: int | None = ...,
) -> pd.DataFrame: ...
@overload
def timeseries(
    nper: int = ...,
    freq: str = ...,
    ncol: int = ...,
    *,
    output: Literal["polars"],
    seed: int | None = ...,
) -> pl.DataFrame: ...
@overload
def timeseries(
    nper: int = ...,
    freq: str = ...,
    ncol: int = ...,
    *,
    output: Literal["dict"],
    seed: int | None = ...,
) -> dict[str, Any]: ...
@overload
def timeseriesData(
    nper: int = ...,
    freq: str = ...,
    ncol: int = ...,
    output: Literal["pandas"] = ...,
    seed: int | None = ...,
) -> dict[str, pd.Series]: ...
@overload
def timeseriesData(
    nper: int = ...,
    freq: str = ...,
    ncol: int = ...,
    *,
    output: Literal["polars"],
    seed: int | None = ...,
) -> dict[str, pl.DataFrame]: ...
@overload
def timeseriesData(
    nper: int = ...,
    freq: str = ...,
    ncol: int = ...,
    *,
    output: Literal["dict"],
    seed: int | None = ...,
) -> dict[str, dict[str, Any]]: ...

# =============================================================================
# Crossfilter generators
# =============================================================================

def machines(
    count: int = ...,
    json: bool = ...,
    seed: int | None = ...,
) -> list[dict[str, Any]]: ...
def usage(
    machine: dict[str, Any],
    json: bool = ...,
    seed: int | None = ...,
) -> dict[str, Any]: ...
def status(
    machine: dict[str, Any],
    json: bool = ...,
) -> dict[str, Any]: ...
def jobs(
    machine: dict[str, Any],
    json: bool = ...,
    seed: int | None = ...,
) -> dict[str, Any] | None: ...

# =============================================================================
# Weather generator
# =============================================================================

@overload
def weather(
    config: Any | None = ...,
    count: int | None = ...,
    output: Literal["pandas"] = ...,
    seed: int | None = ...,
) -> pd.DataFrame: ...
@overload
def weather(
    config: Any | None = ...,
    count: int | None = ...,
    *,
    output: Literal["polars"],
    seed: int | None = ...,
) -> pl.DataFrame: ...
@overload
def weather(
    config: Any | None = ...,
    count: int | None = ...,
    *,
    output: Literal["dict"],
    seed: int | None = ...,
) -> list[dict[str, Any]]: ...

# =============================================================================
# Streaming generators
# =============================================================================

@final
class SuperstoreStream:
    """Iterator for streaming superstore data generation."""

    def __iter__(self) -> SuperstoreStream: ...
    def __next__(self) -> list[dict[str, Any]]: ...

@final
class EmployeeStream:
    """Iterator for streaming employee data generation."""

    def __iter__(self) -> EmployeeStream: ...
    def __next__(self) -> list[dict[str, Any]]: ...

def superstoreStream(
    total_count: int,
    chunk_size: int = ...,
    seed: int | None = ...,
) -> SuperstoreStream: ...
def employeesStream(
    total_count: int,
    chunk_size: int = ...,
    seed: int | None = ...,
) -> EmployeeStream: ...

# =============================================================================
# Parallel generators
# =============================================================================

@overload
def superstoreParallel(
    count: int = ...,
    output: Literal["pandas"] = ...,
    seed: int | None = ...,
) -> pd.DataFrame: ...
@overload
def superstoreParallel(
    count: int = ...,
    *,
    output: Literal["polars"],
    seed: int | None = ...,
) -> pl.DataFrame: ...
@overload
def superstoreParallel(
    count: int = ...,
    *,
    output: Literal["dict"],
    seed: int | None = ...,
) -> list[dict[str, Any]]: ...
@overload
def employeesParallel(
    count: int = ...,
    output: Literal["pandas"] = ...,
    seed: int | None = ...,
) -> pd.DataFrame: ...
@overload
def employeesParallel(
    count: int = ...,
    *,
    output: Literal["polars"],
    seed: int | None = ...,
) -> pl.DataFrame: ...
@overload
def employeesParallel(
    count: int = ...,
    *,
    output: Literal["dict"],
    seed: int | None = ...,
) -> list[dict[str, Any]]: ...
def numThreads() -> int: ...
def setNumThreads(num_threads: int) -> None: ...
def setDeterministicMode(num_threads: int = ...) -> None: ...

# =============================================================================
# Statistical distributions
# =============================================================================

@overload
def sampleUniform(
    min: float,
    max: float,
) -> float: ...
@overload
def sampleUniform(
    min: float,
    max: float,
    n: int,
    seed: int | None = ...,
) -> float | list[float]: ...
@overload
def sampleNormal(
    mean: float,
    std_dev: float,
) -> float: ...
@overload
def sampleNormal(
    mean: float,
    std_dev: float,
    n: int,
    seed: int | None = ...,
) -> float | list[float]: ...
@overload
def sampleLogNormal(
    mu: float,
    sigma: float,
) -> float: ...
@overload
def sampleLogNormal(
    mu: float,
    sigma: float,
    n: int,
    seed: int | None = ...,
) -> float | list[float]: ...
@overload
def sampleExponential(
    lambda_: float,
) -> float: ...
@overload
def sampleExponential(
    lambda_: float,
    n: int,
    seed: int | None = ...,
) -> float | list[float]: ...
@overload
def samplePoisson(
    lambda_: float,
) -> int: ...
@overload
def samplePoisson(
    lambda_: float,
    n: int,
    seed: int | None = ...,
) -> int | list[int]: ...
@overload
def samplePareto(
    scale: float,
    shape: float,
) -> float: ...
@overload
def samplePareto(
    scale: float,
    shape: float,
    n: int,
    seed: int | None = ...,
) -> float | list[float]: ...
@overload
def sampleBeta(
    alpha: float,
    beta: float,
) -> float: ...
@overload
def sampleBeta(
    alpha: float,
    beta: float,
    n: int,
    seed: int | None = ...,
) -> float | list[float]: ...
@overload
def sampleGamma(
    shape: float,
    scale: float,
) -> float: ...
@overload
def sampleGamma(
    shape: float,
    scale: float,
    n: int,
    seed: int | None = ...,
) -> float | list[float]: ...
@overload
def sampleWeibull(
    shape: float,
    scale: float,
) -> float: ...
@overload
def sampleWeibull(
    shape: float,
    scale: float,
    n: int,
    seed: int | None = ...,
) -> float | list[float]: ...
@overload
def sampleCategorical(
    weights: list[float],
) -> int: ...
@overload
def sampleCategorical(
    weights: list[float],
    n: int,
    seed: int | None = ...,
) -> int | list[int]: ...
@overload
def sampleMixture(
    means: list[float],
    std_devs: list[float],
    weights: list[float],
) -> float: ...
@overload
def sampleMixture(
    means: list[float],
    std_devs: list[float],
    weights: list[float],
    n: int,
    seed: int | None = ...,
) -> float | list[float]: ...

# =============================================================================
# Noise models
# =============================================================================

def addGaussianNoise(
    values: list[float],
    std_dev: float,
    seed: int | None = ...,
) -> list[float]: ...
def applyMissing(
    values: list[float],
    probability: float,
    seed: int | None = ...,
) -> list[float | None]: ...

# =============================================================================
# Arrow IPC export
# =============================================================================

def superstoreArrowIpc(
    count: int,
    seed: int | None = ...,
) -> bytes: ...
def employeesArrowIpc(
    count: int,
    seed: int | None = ...,
) -> bytes: ...

# =============================================================================
# File export
# =============================================================================

def superstoreToParquet(
    path: str,
    count: int,
    seed: int | None = ...,
    compression: Literal["none", "snappy", "zstd"] | None = ...,
) -> int: ...
def employeesToParquet(
    path: str,
    count: int,
    seed: int | None = ...,
    compression: Literal["none", "snappy", "zstd"] | None = ...,
) -> int: ...
def superstoreToCsv(
    path: str,
    count: int,
    seed: int | None = ...,
) -> int: ...
def employeesToCsv(
    path: str,
    count: int,
    seed: int | None = ...,
) -> int: ...

# =============================================================================
# Correlation matrix support
# =============================================================================

@final
class CorrelationMatrix:
    """A correlation matrix for generating correlated multivariate normal data."""

    def __init__(
        self,
        data: list[float],
        names: list[str] | None = ...,
    ) -> None: ...
    @staticmethod
    def identity(
        dim: int,
        names: list[str] | None = ...,
    ) -> CorrelationMatrix: ...
    @staticmethod
    def uniform(
        dim: int,
        rho: float,
        names: list[str] | None = ...,
    ) -> CorrelationMatrix: ...
    @staticmethod
    def ar1(
        dim: int,
        rho: float,
        names: list[str] | None = ...,
    ) -> CorrelationMatrix: ...
    @property
    def dim(self) -> int: ...
    @property
    def names(self) -> list[str] | None: ...
    def sample(
        self,
        n: int,
        means: list[float],
        std_devs: list[float],
        seed: int | None = ...,
    ) -> list[list[float]]: ...
    def sample_columns(
        self,
        n: int,
        means: list[float],
        std_devs: list[float],
        seed: int | None = ...,
    ) -> list[list[float]]: ...

def sampleBivariate(
    n: int,
    rho: float,
    mean1: float = ...,
    std1: float = ...,
    mean2: float = ...,
    std2: float = ...,
    seed: int | None = ...,
) -> tuple[list[float], list[float]]: ...
def pearsonCorrelation(
    x: list[float],
    y: list[float],
) -> float: ...

# =============================================================================
# Temporal dependency models
# =============================================================================

@final
class AR1:
    """AR(1) autoregressive model.

    Generates temporally correlated data using x[t] = phi * x[t-1] + mean * (1 - phi) + e[t],
    where e[t] ~ N(0, sigma^2).
    """

    def __init__(
        self,
        phi: float,
        sigma: float,
        mean: float = ...,
    ) -> None: ...
    @property
    def phi(self) -> float: ...
    @property
    def sigma(self) -> float: ...
    @property
    def mean(self) -> float: ...
    @property
    def state(self) -> float: ...
    @state.setter
    def state(self, value: float) -> None: ...
    def reset(self) -> None: ...
    def stationary_variance(self) -> float: ...
    def sample(
        self,
        n: int,
        seed: int | None = ...,
    ) -> list[float]: ...

@final
class ARp:
    """AR(p) autoregressive model of order p.

    Generates temporally correlated data using x[t] = sum(phi[i] * x[t-i-1]) + e[t],
    where e[t] ~ N(0, sigma^2).
    """

    def __init__(
        self,
        coefficients: list[float],
        sigma: float,
        mean: float = ...,
    ) -> None: ...
    @staticmethod
    def ar2(
        phi1: float,
        phi2: float,
        sigma: float,
        mean: float = ...,
    ) -> ARp: ...
    def order(self) -> int: ...
    def reset(self) -> None: ...
    def sample(
        self,
        n: int,
        seed: int | None = ...,
    ) -> list[float]: ...

@final
class MarkovChain:
    """Discrete state Markov chain.

    Generates categorical sequences with transition probabilities.
    """

    def __init__(
        self,
        transition_matrix: list[list[float]],
        states: list[str],
    ) -> None: ...
    @staticmethod
    def two_state(
        state_a: str,
        state_b: str,
        prob_a_to_b: float,
        prob_b_to_a: float,
    ) -> MarkovChain: ...
    def states(self) -> list[str]: ...
    @property
    def current_state(self) -> str: ...
    def set_state(self, state: str) -> None: ...
    def stationary_distribution(self) -> list[float]: ...
    def sample(
        self,
        n: int,
        seed: int | None = ...,
    ) -> list[str]: ...
    def sample_indices(
        self,
        n: int,
        seed: int | None = ...,
    ) -> list[int]: ...

@final
class RandomWalk:
    """Random walk model.

    Generates cumulative sums of random steps.
    """

    def __init__(
        self,
        sigma: float,
        start: float = ...,
        drift: float = ...,
    ) -> None: ...
    @property
    def position(self) -> float: ...
    @position.setter
    def position(self, value: float) -> None: ...
    def sample(
        self,
        n: int,
        seed: int | None = ...,
    ) -> list[float]: ...

@final
class ExponentialSmoothing:
    """Exponential smoothing (EMA-style) model.

    Generates data with exponentially weighted moving average characteristics.
    """

    def __init__(
        self,
        alpha: float,
        sigma: float,
        initial: float = ...,
    ) -> None: ...
    @property
    def smoothed(self) -> float: ...
    def sample(
        self,
        n: int,
        seed: int | None = ...,
    ) -> list[float]: ...

# =============================================================================
# Copula models
# =============================================================================

@final
class GaussianCopula:
    """Gaussian (Normal) copula.

    Uses multivariate normal distribution to model dependencies.
    """

    def __init__(
        self,
        correlation_matrix: list[list[float]],
    ) -> None: ...
    @property
    def dim(self) -> int: ...
    def sample(
        self,
        n: int,
        seed: int | None = ...,
    ) -> list[list[float]]: ...

@final
class ClaytonCopula:
    """Clayton copula with lower tail dependence.

    Good for modeling dependencies where extreme low values tend to occur together.
    """

    def __init__(
        self,
        theta: float,
        dim: int,
    ) -> None: ...
    @property
    def theta(self) -> float: ...
    @property
    def dim(self) -> int: ...
    def kendalls_tau(self) -> float: ...
    def sample(
        self,
        n: int,
        seed: int | None = ...,
    ) -> list[list[float]]: ...

@final
class FrankCopula:
    """Frank copula with symmetric tail dependence.

    Good for modeling overall dependence without tail asymmetry.
    """

    def __init__(
        self,
        theta: float,
    ) -> None: ...
    @property
    def theta(self) -> float: ...
    def kendalls_tau(self) -> float: ...
    def sample(
        self,
        n: int,
        seed: int | None = ...,
    ) -> list[list[float]]: ...

@final
class GumbelCopula:
    """Gumbel copula with upper tail dependence.

    Good for modeling dependencies where extreme high values tend to occur together.
    """

    def __init__(
        self,
        theta: float,
    ) -> None: ...
    @property
    def theta(self) -> float: ...
    def kendalls_tau(self) -> float: ...
    def upper_tail_dependence(self) -> float: ...
    def sample(
        self,
        n: int,
        seed: int | None = ...,
    ) -> list[list[float]]: ...
