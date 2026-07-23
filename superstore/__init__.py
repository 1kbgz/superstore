__version__ = "0.3.2"

# Import directly from the native Rust module
# Import configuration classes from pydantic models
from .config import (
    # Config classes
    CartConfig,
    CatalogConfig,
    # Enums
    ClimateZone,
    CrossfilterConfig,
    EcommerceConfig,
    FinanceConfig,
    FunnelConfig,
    LogFormat,
    LogLevel,
    LogsConfig,
    MachineType,
    OhlcvConfig,
    OptionsConfig,
    OutputFormat,
    RfmConfig,
    Season,
    SessionConfig,
    StockConfig,
    SuperstoreConfig,
    TimeseriesConfig,
    WeatherConfig,
    WeatherEvent,
    # Factory functions
    crossfilter_config,
    ecommerce_config,
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
    # E-commerce generators
    ecommerce_data,
    ecommerce_products,
    ecommerce_sessions,
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
    # Temporal dependency models
    "AR1",
    # Schemas
    "JOBS_SCHEMA",
    "MACHINE_SCHEMA",
    "STATUS_SCHEMA",
    "TELEMETRY_SCENARIOS",
    "TELEMETRY_SCHEMA",
    "USAGE_SCHEMA",
    "ARp",
    "CartConfig",
    "CatalogConfig",
    "ClaytonCopula",
    # Config enums
    "ClimateZone",
    # Correlation matrix support
    "CorrelationMatrix",
    "CrossfilterConfig",
    # E-commerce config
    "EcommerceConfig",
    "EmployeeStream",
    "ExponentialSmoothing",
    # Finance config
    "FinanceConfig",
    "FrankCopula",
    "FunnelConfig",
    # Copula models
    "GaussianCopula",
    "GumbelCopula",
    "LogFormat",
    "LogLevel",
    # Logs config
    "LogsConfig",
    "MachineType",
    "MarkovChain",
    "OhlcvConfig",
    "OptionsConfig",
    "OutputFormat",
    "RandomWalk",
    "RfmConfig",
    "Season",
    "SessionConfig",
    "StockConfig",
    "SuperstoreConfig",
    "SuperstoreStream",
    "TimeseriesConfig",
    # Configuration classes
    "WeatherConfig",
    "WeatherEvent",
    "addGaussianNoise",
    "app_logs",
    "applyMissing",
    "crossfilter_config",
    "ecommerce_config",
    # E-commerce generators
    "ecommerce_data",
    "ecommerce_products",
    "ecommerce_sessions",
    # Core generators
    "employees",
    "employeesArrowIpc",
    "employeesParallel",
    "employeesStream",
    "employeesToCsv",
    "employeesToParquet",
    "finance",
    "finance_config",
    "jobs",
    "logs",
    "logs_config",
    "machines",
    "numThreads",
    "options_chain",
    "pearsonCorrelation",
    "sampleBeta",
    "sampleBivariate",
    "sampleCategorical",
    "sampleExponential",
    "sampleGamma",
    "sampleLogNormal",
    "sampleMixture",
    "sampleNormal",
    "samplePareto",
    "samplePoisson",
    # Statistical distributions
    "sampleUniform",
    "sampleWeibull",
    "setDeterministicMode",
    "setNumThreads",
    "status",
    # Finance generators
    "stock_prices",
    "superstore",
    # Arrow IPC export
    "superstoreArrowIpc",
    # Parallel generators
    "superstoreParallel",
    # Streaming generators
    "superstoreStream",
    "superstoreToCsv",
    # File export
    "superstoreToParquet",
    "superstore_config",
    "telemetry",
    "timeseries",
    "timeseriesData",
    "timeseries_config",
    "usage",
    "weather",
    # Config factory functions
    "weather_config",
)
