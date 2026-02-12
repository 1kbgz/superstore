"""Benchmarks for temporal dependency models."""

import superstore as ss


def _set_deterministic():
    """Set deterministic mode if available."""
    if hasattr(ss, "setDeterministicMode"):
        ss.setDeterministicMode(1)


class AR1Suite:
    """Benchmark suite for AR(1) model."""

    params = [1_000, 10_000, 100_000, 1_000_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        if not hasattr(ss, "AR1"):
            raise NotImplementedError("AR1 not available")
        _set_deterministic()
        self.model = ss.AR1(phi=0.95, sigma=1.0)

    def time_ar1_sample(self, n_samples):
        """Time AR(1) sample generation."""
        self.model.sample(n_samples)


class ARpSuite:
    """Benchmark suite for AR(p) model."""

    params = [1_000, 10_000, 100_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        if not hasattr(ss, "ARp"):
            raise NotImplementedError("ARp not available")
        _set_deterministic()
        self.model = ss.ARp(coefficients=[0.5, 0.3, 0.1], sigma=1.0)

    def time_arp_sample(self, n_samples):
        """Time AR(p) sample generation."""
        self.model.sample(n_samples)


class MarkovChainSuite:
    """Benchmark suite for Markov chain."""

    params = [1_000, 10_000, 100_000, 1_000_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        if not hasattr(ss, "MarkovChain"):
            raise NotImplementedError("MarkovChain not available")
        _set_deterministic()
        # 3-state Markov chain
        self.model = ss.MarkovChain(
            transition_matrix=[
                [0.7, 0.2, 0.1],
                [0.3, 0.5, 0.2],
                [0.2, 0.3, 0.5],
            ],
            states=["A", "B", "C"],
        )

    def time_markov_sample(self, n_samples):
        """Time Markov chain sample generation."""
        self.model.sample(n_samples)


class RandomWalkSuite:
    """Benchmark suite for random walk."""

    params = [1_000, 10_000, 100_000, 1_000_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        if not hasattr(ss, "RandomWalk"):
            raise NotImplementedError("RandomWalk not available")
        _set_deterministic()
        self.model = ss.RandomWalk(sigma=1.0, drift=0.1)

    def time_random_walk_sample(self, n_samples):
        """Time random walk sample generation."""
        self.model.sample(n_samples)


class ExponentialSmoothingSuite:
    """Benchmark suite for exponential smoothing."""

    params = [1_000, 10_000, 100_000, 1_000_000]
    param_names = ["n_samples"]
    timeout = 120

    def setup(self, n_samples):
        if not hasattr(ss, "ExponentialSmoothing"):
            raise NotImplementedError("ExponentialSmoothing not available")
        _set_deterministic()
        self.model = ss.ExponentialSmoothing(alpha=0.3, sigma=1.0)

    def time_exponential_smoothing_sample(self, n_samples):
        """Time exponential smoothing sample generation."""
        self.model.sample(n_samples)
