__version__ = "0.3.0"

# Import directly from the native Rust module
# Import configuration classes from pydantic models
from .config import (
    # Enums
    ClimateZone,
    # Config classes
    CrossfilterConfig,
    FinanceConfig,
    LogFormat,
    LogLevel,
    LogsConfig,
    MachineType,
    OhlcvConfig,
    OptionsConfig,
    OutputFormat,
    Season,
    StockConfig,
    SuperstoreConfig,
    TimeseriesConfig,
    WeatherConfig,
    WeatherEvent,
    # Factory functions
    crossfilter_config,
    finance_config,
    logs_config,
    superstore_config,
    timeseries_config,
    weather_config,
)
from .superstore import (
    # Temporal dependency models
    AR1,
    JOBS_SCHEMA,
    MACHINE_SCHEMA,
    STATUS_SCHEMA,
    TELEMETRY_SCENARIOS,
    TELEMETRY_SCHEMA,
    USAGE_SCHEMA,
    ARp,
    # Copula models
    ClaytonCopula,
    # Correlation matrix support
    CorrelationMatrix,
    EmployeeStream,
    ExponentialSmoothing,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    MarkovChain,
    RandomWalk,
    SuperstoreStream,
    addGaussianNoise,
    app_logs,
    applyMissing,
    # Core generators
    employees,
    # Arrow IPC export
    employeesArrowIpc,
    employeesParallel,
    employeesStream,
    # File export
    employeesToCsv,
    employeesToParquet,
    finance,
    jobs,
    # Logs generators
    logs,
    machines,
    numThreads,
    options_chain,
    pearsonCorrelation,
    sampleBeta,
    sampleBivariate,
    sampleCategorical,
    sampleExponential,
    sampleGamma,
    sampleLogNormal,
    sampleMixture,
    sampleNormal,
    samplePareto,
    samplePoisson,
    # Statistical distributions
    sampleUniform,
    sampleWeibull,
    # Deterministic mode
    setDeterministicMode,
    setNumThreads,
    status,
    # Finance generators
    stock_prices,
    superstore,
    superstoreArrowIpc,
    # Parallel generators
    superstoreParallel,
    # Streaming generators
    superstoreStream,
    superstoreToCsv,
    superstoreToParquet,
    telemetry,
    timeseries,
    timeseriesData,
    usage,
    # Weather generator
    weather,
)

__all__ = (
    # Schemas
    "JOBS_SCHEMA",
    "MACHINE_SCHEMA",
    "STATUS_SCHEMA",
    "TELEMETRY_SCHEMA",
    "TELEMETRY_SCENARIOS",
    "USAGE_SCHEMA",
    # Core generators
    "employees",
    "timeseries",
    "timeseriesData",
    "jobs",
    "machines",
    "status",
    "superstore",
    "telemetry",
    "usage",
    "weather",
    "logs",
    "app_logs",
    # Streaming generators
    "superstoreStream",
    "employeesStream",
    "SuperstoreStream",
    "EmployeeStream",
    # Parallel generators
    "superstoreParallel",
    "employeesParallel",
    "numThreads",
    "setNumThreads",
    "setDeterministicMode",
    # Statistical distributions
    "sampleUniform",
    "sampleNormal",
    "sampleLogNormal",
    "sampleExponential",
    "samplePoisson",
    "samplePareto",
    "sampleBeta",
    "sampleGamma",
    "sampleWeibull",
    "sampleCategorical",
    "sampleMixture",
    "addGaussianNoise",
    "applyMissing",
    # Arrow IPC export
    "superstoreArrowIpc",
    "employeesArrowIpc",
    # File export
    "superstoreToParquet",
    "employeesToParquet",
    "superstoreToCsv",
    "employeesToCsv",
    # Correlation matrix support
    "CorrelationMatrix",
    "sampleBivariate",
    "pearsonCorrelation",
    # Temporal dependency models
    "AR1",
    "ARp",
    "MarkovChain",
    "RandomWalk",
    "ExponentialSmoothing",
    # Copula models
    "GaussianCopula",
    "ClaytonCopula",
    "FrankCopula",
    "GumbelCopula",
    # Configuration classes
    "WeatherConfig",
    "SuperstoreConfig",
    "TimeseriesConfig",
    "CrossfilterConfig",
    # Config enums
    "ClimateZone",
    "Season",
    "WeatherEvent",
    "MachineType",
    "OutputFormat",
    # Config factory functions
    "weather_config",
    "superstore_config",
    "timeseries_config",
    "crossfilter_config",
    # Logs config
    "LogsConfig",
    "LogFormat",
    "LogLevel",
    "logs_config",
    # Finance generators
    "stock_prices",
    "options_chain",
    "finance",
    # Finance config
    "FinanceConfig",
    "StockConfig",
    "OhlcvConfig",
    "OptionsConfig",
    "finance_config",
)
