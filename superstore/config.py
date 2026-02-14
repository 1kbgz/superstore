"""Configuration models for superstore generators using Pydantic.

These config classes provide structured, validated configuration for data generators.
They serialize to JSON/dict for passing to Rust via PyO3.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# =============================================================================
# Enums for configuration options
# =============================================================================


class Season(str, Enum):
    """Season for weather patterns."""

    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"


class ClimateZone(str, Enum):
    """Climate zone affecting weather patterns."""

    TROPICAL = "tropical"
    SUBTROPICAL = "subtropical"
    TEMPERATE = "temperate"
    CONTINENTAL = "continental"
    POLAR = "polar"
    ARID = "arid"
    MEDITERRANEAN = "mediterranean"


class WeatherEvent(str, Enum):
    """Types of weather events."""

    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    SNOW = "snow"
    STORM = "storm"
    HEATWAVE = "heatwave"
    COLD_SNAP = "cold_snap"
    FOG = "fog"


class MachineType(str, Enum):
    """Types of machines for crossfilter."""

    CORE = "core"
    EDGE = "edge"
    WORKER = "worker"


class OutputFormat(str, Enum):
    """Output format for generators."""

    PANDAS = "pandas"
    POLARS = "polars"
    DICT = "dict"


class LogFormat(str, Enum):
    """Log output format styles."""

    COMBINED = "combined"  # Apache Combined Log Format
    COMMON = "common"  # Apache Common Log Format
    JSON = "json"  # JSON structured logs
    APPLICATION = "application"  # Application event logs


class LogLevel(str, Enum):
    """Log severity levels."""

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


# =============================================================================
# Weather Generator Configuration
# =============================================================================


class WeatherConfig(BaseModel):
    """Configuration for the weather data generator.

    Generates realistic outdoor sensor data with temporal patterns,
    seasonal variations, and weather events.
    """

    # Basic parameters
    count: int = Field(default=1000, ge=1, description="Number of readings to generate")
    output: OutputFormat = Field(default=OutputFormat.PANDAS, description="Output format")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")

    # Temporal settings
    start_date: str | None = Field(
        default=None,
        description="Start date (YYYY-MM-DD). Defaults to 30 days ago.",
    )
    frequency_minutes: int = Field(default=15, ge=1, le=1440, description="Reading frequency in minutes")

    # Location settings
    climate_zone: ClimateZone = Field(default=ClimateZone.TEMPERATE, description="Climate zone for realistic patterns")
    latitude: float = Field(default=40.0, ge=-90.0, le=90.0, description="Latitude for day/night calculations")

    # Temperature settings
    base_temp_celsius: float = Field(
        default=15.0,
        ge=-50.0,
        le=50.0,
        description="Annual average temperature in Celsius",
    )
    temp_daily_amplitude: float = Field(
        default=10.0,
        ge=0.0,
        le=30.0,
        description="Day/night temperature swing in Celsius",
    )
    temp_seasonal_amplitude: float = Field(
        default=15.0,
        ge=0.0,
        le=40.0,
        description="Summer/winter temperature swing in Celsius",
    )
    temp_noise_stddev: float = Field(default=2.0, ge=0.0, le=10.0, description="Random noise standard deviation")

    # Humidity settings
    base_humidity_percent: float = Field(default=60.0, ge=0.0, le=100.0, description="Average humidity percentage")
    humidity_temp_correlation: float = Field(
        default=-0.3,
        ge=-1.0,
        le=1.0,
        description="Correlation between temp and humidity (-1 to 1)",
    )

    # Precipitation settings
    precipitation_probability: float = Field(default=0.15, ge=0.0, le=1.0, description="Base probability of precipitation")

    # Weather events
    enable_weather_events: bool = Field(default=True, description="Enable weather event simulation")
    event_probability: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Probability of weather event occurring",
    )

    # Outliers
    outlier_probability: float = Field(
        default=0.01,
        ge=0.0,
        le=0.1,
        description="Probability of outlier readings (sensor errors)",
    )

    # Sensor characteristics
    sensor_drift: bool = Field(default=False, description="Enable gradual sensor calibration drift")
    sensor_drift_rate: float = Field(
        default=0.001,
        ge=0.0,
        le=0.1,
        description="Rate of sensor drift per reading",
    )

    model_config = {"use_enum_values": True}


# =============================================================================
# Superstore Generator Configuration
# =============================================================================


class SeasonalityConfig(BaseModel):
    """Configuration for seasonal patterns in sales data."""

    enable: bool = Field(default=True, description="Enable seasonal effects")
    q4_multiplier: float = Field(default=1.5, ge=1.0, le=3.0, description="Q4 (holiday) sales multiplier")
    summer_multiplier: float = Field(default=0.9, ge=0.5, le=1.5, description="Summer sales multiplier")
    back_to_school_multiplier: float = Field(
        default=1.2,
        ge=1.0,
        le=2.0,
        description="August/September sales multiplier",
    )


class PromotionalConfig(BaseModel):
    """Configuration for promotional effects."""

    enable: bool = Field(default=True, description="Enable promotional patterns")
    discount_quantity_correlation: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How much discounts increase quantity (correlation factor)",
    )
    price_elasticity: float = Field(default=-0.8, ge=-2.0, le=0.0, description="Price elasticity of demand")


class CustomerConfig(BaseModel):
    """Configuration for customer behavior patterns."""

    enable_cohorts: bool = Field(default=True, description="Enable customer cohort modeling")
    repeat_customer_rate: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Fraction of orders from repeat customers",
    )
    vip_segment_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Fraction of customers in VIP segment",
    )
    vip_order_multiplier: float = Field(default=2.0, ge=1.0, le=5.0, description="VIP customer order value multiplier")


class SuperstoreConfig(BaseModel):
    """Configuration for the superstore data generator.

    Generates realistic retail transaction data with correlations
    between sales, quantity, discount, and profit.
    """

    # Basic parameters
    count: int = Field(default=1000, ge=1, description="Number of rows to generate")
    output: OutputFormat = Field(default=OutputFormat.PANDAS, description="Output format")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    pool_size: int = Field(default=1000, ge=1, le=100000, description="Size of pre-generated data pools for performance")

    # Correlation settings
    sales_quantity_correlation: float = Field(default=0.8, ge=-1.0, le=1.0, description="Sales-quantity correlation")
    sales_profit_correlation: float = Field(default=0.9, ge=-1.0, le=1.0, description="Sales-profit correlation")
    discount_profit_correlation: float = Field(default=-0.6, ge=-1.0, le=1.0, description="Discount-profit correlation")

    # Pricing
    enable_price_points: bool = Field(default=True, description="Round prices to realistic $X.99 values")

    # Advanced features
    seasonality: SeasonalityConfig = Field(default_factory=SeasonalityConfig, description="Seasonal patterns")
    promotions: PromotionalConfig = Field(default_factory=PromotionalConfig, description="Promotional effects")
    customers: CustomerConfig = Field(default_factory=CustomerConfig, description="Customer behavior")

    model_config = {"use_enum_values": True}


# =============================================================================
# Time Series Generator Configuration
# =============================================================================


class RegimeConfig(BaseModel):
    """Configuration for regime-switching behavior."""

    enable: bool = Field(default=False, description="Enable regime switching")
    n_regimes: int = Field(default=2, ge=2, le=5, description="Number of regimes")
    regime_persistence: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Probability of staying in current regime",
    )
    volatility_multipliers: list[float] = Field(
        default_factory=lambda: [1.0, 2.5],
        description="Volatility multiplier for each regime",
    )


class JumpConfig(BaseModel):
    """Configuration for jump diffusion."""

    enable: bool = Field(default=False, description="Enable jump diffusion")
    jump_probability: float = Field(default=0.01, ge=0.0, le=0.1, description="Probability of jump per period")
    jump_mean: float = Field(default=0.0, description="Mean jump size")
    jump_stddev: float = Field(default=0.05, ge=0.0, description="Standard deviation of jump size")


class TimeseriesConfig(BaseModel):
    """Configuration for the time series generator.

    Generates financial-style time series with optional regime changes,
    volatility clustering, and jump diffusion.
    """

    # Basic parameters
    nper: int = Field(default=30, ge=1, description="Number of periods")
    ncol: int = Field(default=4, ge=1, le=26, description="Number of columns (max 26)")
    freq: Literal["B", "D", "W", "M"] = Field(default="B", description="Frequency: B=business, D=daily, W=weekly, M=monthly")
    output: OutputFormat = Field(default=OutputFormat.PANDAS, description="Output format")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")

    # Process parameters
    ar_phi: float = Field(
        default=0.95,
        ge=-1.0,
        le=1.0,
        description="AR(1) persistence parameter",
    )
    sigma: float = Field(default=1.0, ge=0.0, description="Innovation standard deviation")
    drift: float = Field(default=0.0, description="Drift/trend per period")
    cumulative: bool = Field(default=True, description="Apply cumulative sum (price-like behavior)")

    # Distribution
    use_fat_tails: bool = Field(default=False, description="Use Student-t instead of normal innovations")
    degrees_freedom: float = Field(default=5.0, ge=2.1, le=30.0, description="Degrees of freedom for Student-t")

    # Correlation
    cross_correlation: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Correlation between columns (0 = independent)",
    )

    # Advanced features
    regimes: RegimeConfig = Field(default_factory=RegimeConfig, description="Regime switching configuration")
    jumps: JumpConfig = Field(default_factory=JumpConfig, description="Jump diffusion configuration")

    model_config = {"use_enum_values": True}


# =============================================================================
# Crossfilter Generator Configuration
# =============================================================================


class AnomalyConfig(BaseModel):
    """Configuration for anomaly injection."""

    enable: bool = Field(default=False, description="Enable anomaly injection")
    cpu_spike_probability: float = Field(default=0.02, ge=0.0, le=0.1, description="Probability of CPU spike")
    memory_leak_probability: float = Field(default=0.01, ge=0.0, le=0.1, description="Probability of memory leak start")
    network_saturation_probability: float = Field(default=0.01, ge=0.0, le=0.1, description="Probability of network saturation")


class TemporalPatternConfig(BaseModel):
    """Configuration for temporal patterns in IoT data."""

    enable_diurnal: bool = Field(default=False, description="Enable day/night load patterns")
    enable_weekly: bool = Field(default=False, description="Enable weekday/weekend patterns")
    peak_hour: int = Field(default=14, ge=0, le=23, description="Hour of peak load (0-23)")
    night_load_factor: float = Field(default=0.3, ge=0.0, le=1.0, description="Load factor during night hours")
    weekend_load_factor: float = Field(default=0.5, ge=0.0, le=1.0, description="Load factor during weekends")


class CrossfilterConfig(BaseModel):
    """Configuration for crossfilter IoT data generator.

    Generates machine telemetry data suitable for dashboard demos
    with optional anomalies and temporal patterns.
    """

    # Basic parameters
    n_machines: int = Field(default=10, ge=1, description="Number of machines")
    n_readings: int = Field(default=1000, ge=1, description="Number of usage readings per machine")
    output: OutputFormat = Field(default=OutputFormat.PANDAS, description="Output format")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")

    # Machine configuration
    machine_types: list[MachineType] = Field(
        default_factory=lambda: [MachineType.CORE, MachineType.EDGE, MachineType.WORKER],
        description="Types of machines to generate",
    )
    cores_range: tuple[int, int] = Field(default=(4, 64), description="Range of CPU cores per machine")
    zones: list[str] = Field(
        default_factory=lambda: ["zone-a", "zone-b", "zone-c"],
        description="Available zones",
    )
    regions: list[str] = Field(
        default_factory=lambda: ["us-east-1", "us-west-2", "eu-west-1"],
        description="Available regions",
    )

    # Usage profiles
    base_cpu_load: float = Field(default=0.3, ge=0.0, le=1.0, description="Base CPU utilization")
    base_memory_load: float = Field(default=0.5, ge=0.0, le=1.0, description="Base memory utilization")
    load_variance: float = Field(default=0.2, ge=0.0, le=0.5, description="Variance in load readings")

    # Advanced features
    anomalies: AnomalyConfig = Field(default_factory=AnomalyConfig, description="Anomaly injection settings")
    temporal_patterns: TemporalPatternConfig = Field(default_factory=TemporalPatternConfig, description="Temporal pattern settings")

    # Failure simulation
    enable_failures: bool = Field(default=False, description="Enable machine failure simulation")
    failure_probability: float = Field(default=0.001, ge=0.0, le=0.1, description="Probability of failure per reading")
    cascade_failure_probability: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Probability of cascade failure when dependent machine fails",
    )

    model_config = {"use_enum_values": True}


# =============================================================================
# Logs Generator Configuration
# =============================================================================


class ErrorBurstConfig(BaseModel):
    """Configuration for error burst behavior in logs."""

    enable: bool = Field(default=True, description="Enable error burst simulation")
    burst_probability: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Probability of entering a burst state per second",
    )
    burst_duration_seconds: int = Field(
        default=30,
        ge=1,
        description="Average duration of error bursts in seconds",
    )
    burst_error_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Error rate during burst periods",
    )


class LatencyConfig(BaseModel):
    """Configuration for request latency distribution."""

    base_latency_ms: float = Field(
        default=50.0,
        ge=1.0,
        description="Base latency in milliseconds (median)",
    )
    latency_stddev: float = Field(
        default=0.8,
        ge=0.1,
        description="Standard deviation for log-normal distribution",
    )
    slow_request_probability: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Probability of a slow request",
    )
    slow_request_multiplier: float = Field(
        default=10.0,
        ge=1.0,
        description="Multiplier for slow request latency",
    )


class LogsConfig(BaseModel):
    """Configuration for the logs data generator.

    Generates realistic web server access logs and application event logs
    with configurable traffic patterns, error rates, and latency distributions.
    """

    # Basic settings
    count: int = Field(default=1000, ge=1, description="Number of log entries to generate")
    output: OutputFormat = Field(
        default=OutputFormat.PANDAS,
        description="Output format (pandas, polars, or dict)",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    format: LogFormat = Field(
        default=LogFormat.COMBINED,
        description="Log format style",
    )

    # Traffic patterns
    start_time: str | None = Field(
        default=None,
        description="Start timestamp (ISO format). Defaults to current time.",
    )
    requests_per_second: float = Field(
        default=100.0,
        ge=0.1,
        description="Average requests per second (Poisson rate)",
    )

    # Status code distribution
    success_rate: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Base success rate (2xx responses)",
    )

    # Error bursts
    error_burst: ErrorBurstConfig = Field(
        default_factory=ErrorBurstConfig,
        description="Error burst configuration",
    )

    # Latency
    latency: LatencyConfig = Field(
        default_factory=LatencyConfig,
        description="Latency distribution configuration",
    )

    # Request details
    include_user_agent: bool = Field(
        default=True,
        description="Include user agent strings",
    )
    unique_ips: int = Field(
        default=1000,
        ge=1,
        description="Number of unique IP addresses to generate",
    )
    unique_users: int = Field(
        default=500,
        ge=1,
        description="Number of unique user IDs",
    )
    api_path_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Ratio of API paths vs static paths",
    )

    model_config = {"use_enum_values": True}


# =============================================================================
# Finance Config Classes
# =============================================================================


class StockConfig(BaseModel):
    """Configuration for stock price generation using Geometric Brownian Motion."""

    annual_drift: float = Field(
        default=0.08,
        description="Annual expected return (mu). E.g., 0.08 = 8% annual return",
    )
    annual_volatility: float = Field(
        default=0.20,
        ge=0.0,
        description="Annual volatility (sigma). E.g., 0.20 = 20% annual volatility",
    )
    initial_price: float = Field(
        default=100.0,
        gt=0.0,
        description="Initial stock price",
    )
    enable_jumps: bool = Field(
        default=False,
        description="Enable jump diffusion for more realistic price movements",
    )
    jump_probability: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Daily probability of a jump event",
    )
    jump_mean: float = Field(
        default=0.0,
        description="Mean of jump size (log-normal)",
    )
    jump_stddev: float = Field(
        default=0.05,
        ge=0.0,
        description="Standard deviation of jump size",
    )


class OhlcvConfig(BaseModel):
    """Configuration for OHLCV (Open-High-Low-Close-Volume) bar generation."""

    avg_volume: int = Field(
        default=1_000_000,
        ge=1,
        description="Average daily trading volume",
    )
    volume_volatility: float = Field(
        default=0.5,
        ge=0.0,
        description="Volatility of volume (log-normal sigma)",
    )
    intraday_volatility: float = Field(
        default=0.02,
        ge=0.0,
        description="Intraday price range volatility",
    )
    volume_price_correlation: float = Field(
        default=0.3,
        ge=-1.0,
        le=1.0,
        description="Correlation between volume and absolute returns",
    )


class OptionsConfig(BaseModel):
    """Configuration for options chain generation with Black-Scholes pricing."""

    risk_free_rate: float = Field(
        default=0.05,
        description="Annual risk-free interest rate",
    )
    dividend_yield: float = Field(
        default=0.02,
        ge=0.0,
        description="Annual dividend yield",
    )
    expirations: list[int] = Field(
        default_factory=lambda: [7, 14, 30, 60, 90],
        description="Days to expiration for option contracts",
    )
    strike_offsets: list[float] = Field(
        default_factory=lambda: [0.90, 0.95, 0.97, 1.0, 1.03, 1.05, 1.10],
        description="Strike prices as multipliers of spot price",
    )


class FinanceConfig(BaseModel):
    """Configuration for the finance data generator.

    Generates realistic financial market data including OHLCV stock prices,
    multi-asset correlated returns, and options chains with Black-Scholes pricing.
    """

    # Basic settings
    ndays: int = Field(
        default=252,
        ge=1,
        description="Number of trading days to generate (252 = 1 year)",
    )
    n_assets: int = Field(
        default=1,
        ge=1,
        description="Number of assets (1 = single stock, >1 = correlated multi-asset)",
    )
    output: OutputFormat = Field(
        default=OutputFormat.PANDAS,
        description="Output format (pandas, polars, or dict)",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    start_date: str | None = Field(
        default=None,
        description="Start date (ISO format YYYY-MM-DD). Defaults to 2024-01-02.",
    )
    tickers: list[str] = Field(
        default_factory=lambda: ["AAPL"],
        description="Ticker symbols for the assets",
    )
    asset_correlation: float = Field(
        default=0.5,
        ge=-1.0,
        le=1.0,
        description="Correlation between assets (for multi-asset generation)",
    )

    # Nested configurations
    stock: StockConfig = Field(
        default_factory=StockConfig,
        description="Stock price generation configuration",
    )
    ohlcv: OhlcvConfig = Field(
        default_factory=OhlcvConfig,
        description="OHLCV bar configuration",
    )
    options: OptionsConfig = Field(
        default_factory=OptionsConfig,
        description="Options chain configuration",
    )

    model_config = {"use_enum_values": True}


# =============================================================================
# Convenience factory functions
# =============================================================================


def weather_config(**kwargs) -> WeatherConfig:
    """Create a weather configuration with the given parameters."""
    return WeatherConfig(**kwargs)


def superstore_config(**kwargs) -> SuperstoreConfig:
    """Create a superstore configuration with the given parameters."""
    return SuperstoreConfig(**kwargs)


def timeseries_config(**kwargs) -> TimeseriesConfig:
    """Create a time series configuration with the given parameters."""
    return TimeseriesConfig(**kwargs)


def crossfilter_config(**kwargs) -> CrossfilterConfig:
    """Create a crossfilter configuration with the given parameters."""
    return CrossfilterConfig(**kwargs)


def logs_config(**kwargs) -> LogsConfig:
    """Create a logs configuration with the given parameters."""
    return LogsConfig(**kwargs)


def finance_config(**kwargs) -> FinanceConfig:
    """Create a finance configuration with the given parameters."""
    return FinanceConfig(**kwargs)
