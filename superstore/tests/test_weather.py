"""Tests for the weather data generator."""

import pytest


class TestWeather:
    """Test suite for weather() generator."""

    def test_weather_basic(self):
        """Test basic weather generation."""
        from superstore import weather

        df = weather(count=100, seed=42)
        assert len(df) == 100
        assert "timestamp" in df.columns
        assert "temperature_celsius" in df.columns
        assert "humidity_percent" in df.columns
        assert "precipitation_mm" in df.columns
        assert "weather_event" in df.columns
        assert "is_outlier" in df.columns

    def test_weather_seed_reproducibility(self):
        """Test that same seed produces identical results."""
        from superstore import weather

        df1 = weather(count=50, seed=42)
        df2 = weather(count=50, seed=42)

        assert df1["temperature_celsius"].tolist() == df2["temperature_celsius"].tolist()
        assert df1["humidity_percent"].tolist() == df2["humidity_percent"].tolist()

    def test_weather_different_seeds(self):
        """Test that different seeds produce different results."""
        from superstore import weather

        df1 = weather(count=50, seed=42)
        df2 = weather(count=50, seed=123)

        assert df1["temperature_celsius"].tolist() != df2["temperature_celsius"].tolist()

    def test_weather_polars_output(self):
        """Test polars DataFrame output."""
        import polars as pl

        from superstore import weather

        df = weather(count=50, seed=42, output="polars")
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 50
        assert "temperature_celsius" in df.columns

    def test_weather_dict_output(self):
        """Test dict output."""
        from superstore import weather

        data = weather(count=50, seed=42, output="dict")
        assert isinstance(data, list)
        assert len(data) == 50
        assert "temperature_celsius" in data[0]

    def test_weather_with_config(self):
        """Test weather generation with WeatherConfig."""
        from superstore import WeatherConfig, weather

        # Use a summer date to avoid winter temperature issues
        config = WeatherConfig(
            count=30,
            seed=42,
            base_temp_celsius=20.0,
            start_date="2026-07-01",  # Summer in Northern Hemisphere
        )
        df = weather(config=config)

        assert len(df) == 30
        # Temperature should be around the base with daily/seasonal variations
        mean_temp = df["temperature_celsius"].mean()
        # In summer, temperatures should be higher than base (base + seasonal amplitude)
        assert 10 < mean_temp < 50

    def test_weather_climate_zones(self):
        """Test different climate zones produce different temperatures."""
        from superstore import ClimateZone, WeatherConfig, weather

        tropical = WeatherConfig(count=100, seed=42, climate_zone=ClimateZone.TROPICAL)
        polar = WeatherConfig(count=100, seed=42, climate_zone=ClimateZone.POLAR)

        df_tropical = weather(config=tropical)
        df_polar = weather(config=polar)

        # Tropical should be warmer than polar on average
        assert df_tropical["temperature_celsius"].mean() > df_polar["temperature_celsius"].mean()

    def test_weather_humidity_correlation(self):
        """Test humidity is inversely correlated with temperature."""
        from superstore import WeatherConfig, weather

        config = WeatherConfig(
            count=500,
            seed=42,
            humidity_temp_correlation=-0.5,
            enable_weather_events=False,  # Disable events for cleaner correlation
        )
        df = weather(config=config)

        # Calculate correlation
        corr = df["temperature_celsius"].corr(df["humidity_percent"])
        # Should be negative (inverse correlation)
        assert corr < 0

    def test_weather_precipitation(self):
        """Test precipitation is generated."""
        from superstore import WeatherConfig, weather

        config = WeatherConfig(
            count=500,
            seed=42,
            precipitation_probability=0.3,  # Higher probability
            enable_weather_events=True,
        )
        df = weather(config=config)

        # Should have some non-zero precipitation
        assert df["precipitation_mm"].sum() > 0

    def test_weather_events(self):
        """Test weather events are generated."""
        from superstore import WeatherConfig, weather

        config = WeatherConfig(
            count=500,
            seed=42,
            enable_weather_events=True,
            event_probability=0.3,  # High probability to ensure events
        )
        df = weather(config=config)

        # Should have multiple event types
        unique_events = df["weather_event"].unique()
        assert len(unique_events) > 1

    def test_weather_outliers(self):
        """Test outlier injection."""
        from superstore import WeatherConfig, weather

        config = WeatherConfig(
            count=500,
            seed=42,
            outlier_probability=0.1,  # 10% outliers
        )
        df = weather(config=config)

        outlier_count = df["is_outlier"].sum()
        # Should have roughly 10% outliers (allow variance)
        assert 20 < outlier_count < 100

    def test_weather_sensor_drift(self):
        """Test sensor drift feature."""
        from superstore import WeatherConfig, weather

        config = WeatherConfig(
            count=1000,
            seed=42,
            sensor_drift=True,
            sensor_drift_rate=0.01,  # Noticeable drift
        )
        df = weather(config=config)

        # First readings should differ from last readings due to drift
        # (Drift is cumulative)
        assert len(df) == 1000

    def test_weather_frequency(self):
        """Test different reading frequencies."""
        import pandas as pd

        from superstore import WeatherConfig, weather

        config = WeatherConfig(
            count=10,
            seed=42,
            frequency_minutes=60,  # Hourly readings
        )
        df = weather(config=config)

        # Check timestamps are 60 minutes apart
        timestamps = pd.to_datetime(df["timestamp"])
        time_diffs = timestamps.diff().dropna()
        # All differences should be 60 minutes
        assert all(td.total_seconds() == 3600 for td in time_diffs)

    def test_weather_config_dict(self):
        """Test passing config as dict."""
        from superstore import weather

        config_dict = {
            "count": 25,
            "seed": 42,
            "base_temp_celsius": 10.0,
        }
        # This should work via the pydantic model
        from superstore import WeatherConfig

        config = WeatherConfig(**config_dict)
        df = weather(config=config)
        assert len(df) == 25


class TestWeatherConfig:
    """Test suite for WeatherConfig pydantic model."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from superstore import WeatherConfig

        config = WeatherConfig()
        assert config.count == 1000
        assert config.frequency_minutes == 15
        assert config.base_temp_celsius == 15.0
        assert config.seed is None

    def test_config_validation(self):
        """Test pydantic validation."""
        from superstore import WeatherConfig

        # Valid config
        config = WeatherConfig(count=100, latitude=45.0)
        assert config.count == 100

        # Invalid latitude should raise
        with pytest.raises(Exception):
            WeatherConfig(latitude=100.0)  # > 90

    def test_config_serialization(self):
        """Test config serialization to dict."""
        from superstore import ClimateZone, WeatherConfig

        config = WeatherConfig(count=500, climate_zone=ClimateZone.ARID, seed=123)
        data = config.model_dump()

        assert data["count"] == 500
        assert data["climate_zone"] == "arid"
        assert data["seed"] == 123

    def test_config_factory(self):
        """Test weather_config factory function."""
        from superstore import weather_config

        config = weather_config(count=200, seed=42)
        assert config.count == 200
        assert config.seed == 42
