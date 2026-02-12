"""Tests for statistical distributions."""


class TestDistributions:
    def test_sample_uniform(self):
        from superstore import sampleUniform

        # Single sample
        value = sampleUniform(0, 100, seed=42)
        assert isinstance(value, float)
        assert 0 <= value <= 100

        # Multiple samples
        samples = sampleUniform(0, 100, n=100, seed=42)
        assert len(samples) == 100
        assert all(0 <= s <= 100 for s in samples)

    def test_sample_uniform_reproducible(self):
        from superstore import sampleUniform

        s1 = sampleUniform(0, 100, n=10, seed=42)
        s2 = sampleUniform(0, 100, n=10, seed=42)
        assert s1 == s2

    def test_sample_normal(self):
        from superstore import sampleNormal

        # Single sample
        value = sampleNormal(50, 10, seed=42)
        assert isinstance(value, float)

        # Multiple samples
        samples = sampleNormal(50, 10, n=1000, seed=42)
        mean = sum(samples) / len(samples)
        # Mean should be close to 50
        assert abs(mean - 50) < 5

    def test_sample_normal_reproducible(self):
        from superstore import sampleNormal

        s1 = sampleNormal(0, 1, n=10, seed=42)
        s2 = sampleNormal(0, 1, n=10, seed=42)
        assert s1 == s2

    def test_sample_log_normal(self):
        from superstore import sampleLogNormal

        samples = sampleLogNormal(0, 1, n=100, seed=42)
        assert len(samples) == 100
        # All values should be positive
        assert all(s > 0 for s in samples)

    def test_sample_exponential(self):
        from superstore import sampleExponential

        samples = sampleExponential(1.0, n=100, seed=42)
        assert len(samples) == 100
        assert all(s >= 0 for s in samples)

    def test_sample_poisson(self):
        from superstore import samplePoisson

        samples = samplePoisson(10.0, n=100, seed=42)
        assert len(samples) == 100
        # Poisson returns integers
        assert all(isinstance(s, int) for s in samples)
        assert all(s >= 0 for s in samples)

    def test_sample_pareto(self):
        from superstore import samplePareto

        samples = samplePareto(1.0, 2.0, n=100, seed=42)
        assert len(samples) == 100
        # Pareto has minimum value of scale
        assert all(s >= 1.0 for s in samples)

    def test_sample_beta(self):
        from superstore import sampleBeta

        samples = sampleBeta(2.0, 5.0, n=100, seed=42)
        assert len(samples) == 100
        # Beta is bounded [0, 1]
        assert all(0 <= s <= 1 for s in samples)

    def test_sample_gamma(self):
        from superstore import sampleGamma

        samples = sampleGamma(2.0, 1.0, n=100, seed=42)
        assert len(samples) == 100
        assert all(s >= 0 for s in samples)

    def test_sample_weibull(self):
        from superstore import sampleWeibull

        samples = sampleWeibull(2.0, 1.0, n=100, seed=42)
        assert len(samples) == 100
        assert all(s >= 0 for s in samples)

    def test_sample_categorical(self):
        from superstore import sampleCategorical

        weights = [0.5, 0.3, 0.2]

        # Single sample
        idx = sampleCategorical(weights, seed=42)
        assert isinstance(idx, int)
        assert 0 <= idx <= 2

        # Multiple samples - check distribution roughly matches weights
        samples = sampleCategorical(weights, n=1000, seed=42)
        counts = [samples.count(i) for i in range(3)]
        # Category 0 (weight 0.5) should be most common
        assert counts[0] > counts[1] > counts[2]

    def test_sample_mixture(self):
        from superstore import sampleMixture

        # Bimodal distribution
        samples = sampleMixture(
            means=[30000, 80000],
            std_devs=[10000, 20000],
            weights=[0.6, 0.4],
            n=1000,
            seed=42,
        )
        assert len(samples) == 1000

        # Should have values near both modes
        low_count = sum(1 for s in samples if s < 50000)
        high_count = sum(1 for s in samples if s > 60000)
        assert low_count > 200
        assert high_count > 200

    def test_add_gaussian_noise(self):
        from superstore import addGaussianNoise

        values = [100.0] * 100
        noisy = addGaussianNoise(values, std_dev=5.0, seed=42)
        assert len(noisy) == 100

        # Mean should still be close to 100
        mean = sum(noisy) / len(noisy)
        assert abs(mean - 100) < 3

        # Values should vary
        assert min(noisy) != max(noisy)

    def test_apply_missing(self):
        from superstore import applyMissing

        values = [1.0, 2.0, 3.0, 4.0, 5.0] * 200  # 1000 values
        result = applyMissing(values, probability=0.3, seed=42)
        assert len(result) == 1000

        # Should have some None values (approximately 30%)
        none_count = sum(1 for v in result if v is None)
        assert 200 < none_count < 400

    def test_distributions_reproducible(self):
        from superstore import sampleBeta, sampleGamma, sampleMixture

        # All distributions should be reproducible with same seed
        assert sampleBeta(2, 5, n=10, seed=42) == sampleBeta(2, 5, n=10, seed=42)
        assert sampleGamma(2, 1, n=10, seed=42) == sampleGamma(2, 1, n=10, seed=42)
        assert sampleMixture([0, 100], [10, 10], [0.5, 0.5], n=10, seed=42) == sampleMixture([0, 100], [10, 10], [0.5, 0.5], n=10, seed=42)
