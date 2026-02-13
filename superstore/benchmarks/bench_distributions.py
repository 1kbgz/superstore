"""Benchmarks for distribution sampling functions."""

import superstore as ss


def _set_deterministic():
    """Set deterministic mode if available."""
    if hasattr(ss, "setDeterministicMode"):
        ss.setDeterministicMode(1)


class NormalDistributionSuite:
    """Benchmark suite for normal distribution sampling."""

    params = [1_000, 10_000, 100_000, 1_000_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        _set_deterministic()

    def time_sampleNormal(self, n_samples):
        """Time normal distribution sampling."""
        ss.sampleNormal(0.0, 1.0, n=n_samples)


class ExponentialDistributionSuite:
    """Benchmark suite for exponential distribution sampling."""

    params = [1_000, 10_000, 100_000, 1_000_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        _set_deterministic()

    def time_sampleExponential(self, n_samples):
        """Time exponential distribution sampling."""
        ss.sampleExponential(1.0, n=n_samples)


class UniformDistributionSuite:
    """Benchmark suite for uniform distribution sampling."""

    params = [1_000, 10_000, 100_000, 1_000_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        _set_deterministic()

    def time_sampleUniform(self, n_samples):
        """Time uniform distribution sampling."""
        ss.sampleUniform(0.0, 1.0, n=n_samples)


class PoissonDistributionSuite:
    """Benchmark suite for Poisson distribution sampling."""

    params = [1_000, 10_000, 100_000, 1_000_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        _set_deterministic()

    def time_samplePoisson(self, n_samples):
        """Time Poisson distribution sampling."""
        ss.samplePoisson(5.0, n=n_samples)


class LogNormalDistributionSuite:
    """Benchmark suite for log-normal distribution sampling."""

    params = [1_000, 10_000, 100_000, 1_000_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        _set_deterministic()

    def time_sampleLogNormal(self, n_samples):
        """Time log-normal distribution sampling."""
        ss.sampleLogNormal(0.0, 1.0, n=n_samples)


class GammaDistributionSuite:
    """Benchmark suite for gamma distribution sampling."""

    params = [1_000, 10_000, 100_000, 1_000_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        _set_deterministic()

    def time_sampleGamma(self, n_samples):
        """Time gamma distribution sampling."""
        ss.sampleGamma(2.0, 1.0, n=n_samples)


class BetaDistributionSuite:
    """Benchmark suite for beta distribution sampling."""

    params = [1_000, 10_000, 100_000, 1_000_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        _set_deterministic()

    def time_sampleBeta(self, n_samples):
        """Time beta distribution sampling."""
        ss.sampleBeta(2.0, 5.0, n=n_samples)
