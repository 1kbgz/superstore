"""Benchmarks for core data generators."""

import superstore as ss


def _set_deterministic():
    """Set deterministic mode if available."""
    if hasattr(ss, "setDeterministicMode"):
        ss.setDeterministicMode(1)


class SuperstoreSuite:
    """Benchmark suite for superstore() generator."""

    params = [1_000, 10_000, 100_000]
    param_names = ["n_rows"]
    timeout = 120  # seconds

    def setup(self, n_rows):
        # Ensure deterministic mode for consistent benchmarks (1 thread)
        _set_deterministic()

    def time_superstore(self, n_rows):
        """Time the superstore() generator."""
        ss.superstore(n_rows)

    def peakmem_superstore(self, n_rows):
        """Measure peak memory of superstore()."""
        ss.superstore(n_rows)


class EmployeesSuite:
    """Benchmark suite for employees() generator."""

    params = [1_000, 10_000, 100_000]
    param_names = ["n_rows"]
    timeout = 120

    def setup(self, n_rows):
        _set_deterministic()

    def time_employees(self, n_rows):
        """Time the employees() generator."""
        ss.employees(n_rows)

    def peakmem_employees(self, n_rows):
        """Measure peak memory of employees()."""
        ss.employees(n_rows)


class TimeSeriesSuite:
    """Benchmark suite for time series generator."""

    params = [1_000, 10_000, 100_000, 1_000_000]
    param_names = ["n_rows"]
    timeout = 120

    def setup(self, n_rows):
        _set_deterministic()

    def time_timeseries(self, n_rows):
        """Time the timeseries() generator."""
        ss.timeseries(n_rows)

    def peakmem_timeseries(self, n_rows):
        """Measure peak memory of timeseries()."""
        ss.timeseries(n_rows)


class WeatherSuite:
    """Benchmark suite for weather() generator."""

    params = [1_000, 10_000, 100_000]
    param_names = ["n_rows"]
    timeout = 120

    def setup(self, n_rows):
        _set_deterministic()

    def time_weather(self, n_rows):
        """Time the weather() generator."""
        ss.weather(count=n_rows, seed=42)

    def peakmem_weather(self, n_rows):
        """Measure peak memory of weather()."""
        ss.weather(count=n_rows, seed=42)
