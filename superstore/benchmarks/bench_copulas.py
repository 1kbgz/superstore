"""Benchmarks for copula models."""

import superstore as ss


def _set_deterministic():
    """Set deterministic mode if available."""
    if hasattr(ss, "setDeterministicMode"):
        ss.setDeterministicMode(1)


class GaussianCopulaSuite:
    """Benchmark suite for Gaussian copula."""

    params = [1_000, 10_000, 100_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        if not hasattr(ss, "GaussianCopula"):
            raise NotImplementedError("GaussianCopula not available")
        _set_deterministic()
        # 4-dimensional correlation matrix
        self.model = ss.GaussianCopula(
            correlation_matrix=[
                [1.0, 0.7, 0.3, 0.1],
                [0.7, 1.0, 0.5, 0.2],
                [0.3, 0.5, 1.0, 0.4],
                [0.1, 0.2, 0.4, 1.0],
            ]
        )

    def time_gaussian_copula_sample(self, n_samples):
        """Time Gaussian copula sample generation."""
        self.model.sample(n_samples)


class HighDimensionalGaussianCopulaSuite:
    """Benchmark suite for high-dimensional Gaussian copula."""

    params = [1_000, 10_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        if not hasattr(ss, "GaussianCopula"):
            raise NotImplementedError("GaussianCopula not available")
        _set_deterministic()
        # 10-dimensional identity correlation (independent)
        dim = 10
        self.model = ss.GaussianCopula(correlation_matrix=[[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)])

    def time_10d_gaussian_copula_sample(self, n_samples):
        """Time 10-dimensional Gaussian copula sample generation."""
        self.model.sample(n_samples)


class ClaytonCopulaSuite:
    """Benchmark suite for Clayton copula."""

    params = [1_000, 10_000, 100_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        if not hasattr(ss, "ClaytonCopula"):
            raise NotImplementedError("ClaytonCopula not available")
        _set_deterministic()
        self.model = ss.ClaytonCopula(theta=2.0, dim=2)

    def time_clayton_copula_sample(self, n_samples):
        """Time Clayton copula sample generation."""
        self.model.sample(n_samples)


class FrankCopulaSuite:
    """Benchmark suite for Frank copula."""

    params = [1_000, 10_000, 100_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        if not hasattr(ss, "FrankCopula"):
            raise NotImplementedError("FrankCopula not available")
        _set_deterministic()
        self.model = ss.FrankCopula(theta=5.0)

    def time_frank_copula_sample(self, n_samples):
        """Time Frank copula sample generation."""
        self.model.sample(n_samples)


class GumbelCopulaSuite:
    """Benchmark suite for Gumbel copula."""

    params = [1_000, 10_000, 100_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        if not hasattr(ss, "GumbelCopula"):
            raise NotImplementedError("GumbelCopula not available")
        _set_deterministic()
        self.model = ss.GumbelCopula(theta=2.0)

    def time_gumbel_copula_sample(self, n_samples):
        """Time Gumbel copula sample generation."""
        self.model.sample(n_samples)
