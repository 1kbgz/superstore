//! Statistical distributions for realistic data generation.
//!
//! This module provides configurable probability distributions that can be
//! used to generate more realistic synthetic data. It includes common
//! distributions like Normal, Log-normal, Exponential, and more.

use rand::Rng;
use rand_distr::{
    Beta, Distribution, Exp, Gamma, LogNormal, Normal, Pareto, Poisson, Uniform, Weibull,
};
use serde::{Deserialize, Serialize};

/// Error type for distribution operations.
#[derive(Debug, Clone)]
pub enum DistributionError {
    InvalidParameters(String),
}

impl std::fmt::Display for DistributionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistributionError::InvalidParameters(msg) => write!(f, "Invalid parameters: {}", msg),
        }
    }
}

impl std::error::Error for DistributionError {}

/// Enumeration of available probability distributions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DistributionType {
    /// Uniform distribution between min and max.
    Uniform { min: f64, max: f64 },

    /// Normal (Gaussian) distribution with mean and standard deviation.
    Normal { mean: f64, std_dev: f64 },

    /// Log-normal distribution (exp of normal).
    LogNormal { mu: f64, sigma: f64 },

    /// Exponential distribution with rate parameter lambda.
    Exponential { lambda: f64 },

    /// Poisson distribution with rate parameter lambda.
    Poisson { lambda: f64 },

    /// Pareto distribution (power law) with scale and shape.
    Pareto { scale: f64, shape: f64 },

    /// Beta distribution for values in [0, 1].
    Beta { alpha: f64, beta: f64 },

    /// Gamma distribution.
    Gamma { shape: f64, scale: f64 },

    /// Weibull distribution for reliability/survival analysis.
    Weibull { shape: f64, scale: f64 },

    /// Categorical distribution with weighted categories.
    Categorical { weights: Vec<f64> },

    /// Mixture of multiple distributions.
    Mixture {
        distributions: Vec<DistributionType>,
        weights: Vec<f64>,
    },
}

impl DistributionType {
    /// Create a uniform distribution.
    pub fn uniform(min: f64, max: f64) -> Self {
        DistributionType::Uniform { min, max }
    }

    /// Create a normal distribution.
    pub fn normal(mean: f64, std_dev: f64) -> Self {
        DistributionType::Normal { mean, std_dev }
    }

    /// Create a log-normal distribution.
    pub fn log_normal(mu: f64, sigma: f64) -> Self {
        DistributionType::LogNormal { mu, sigma }
    }

    /// Create an exponential distribution.
    pub fn exponential(lambda: f64) -> Self {
        DistributionType::Exponential { lambda }
    }

    /// Create a Poisson distribution.
    pub fn poisson(lambda: f64) -> Self {
        DistributionType::Poisson { lambda }
    }

    /// Create a Pareto (power law) distribution.
    pub fn pareto(scale: f64, shape: f64) -> Self {
        DistributionType::Pareto { scale, shape }
    }

    /// Create a Beta distribution.
    pub fn beta(alpha: f64, beta: f64) -> Self {
        DistributionType::Beta { alpha, beta }
    }

    /// Create a Gamma distribution.
    pub fn gamma(shape: f64, scale: f64) -> Self {
        DistributionType::Gamma { shape, scale }
    }

    /// Create a Weibull distribution.
    pub fn weibull(shape: f64, scale: f64) -> Self {
        DistributionType::Weibull { shape, scale }
    }

    /// Create a categorical distribution with weights.
    pub fn categorical(weights: Vec<f64>) -> Self {
        DistributionType::Categorical { weights }
    }

    /// Create a mixture of distributions.
    pub fn mixture(distributions: Vec<DistributionType>, weights: Vec<f64>) -> Self {
        DistributionType::Mixture {
            distributions,
            weights,
        }
    }

    /// Sample a value from this distribution.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        match self {
            DistributionType::Uniform { min, max } => {
                let dist = Uniform::new(*min, *max);
                dist.sample(rng)
            }
            DistributionType::Normal { mean, std_dev } => {
                let dist =
                    Normal::new(*mean, *std_dev).unwrap_or_else(|_| Normal::new(0.0, 1.0).unwrap());
                dist.sample(rng)
            }
            DistributionType::LogNormal { mu, sigma } => {
                let dist = LogNormal::new(*mu, *sigma)
                    .unwrap_or_else(|_| LogNormal::new(0.0, 1.0).unwrap());
                dist.sample(rng)
            }
            DistributionType::Exponential { lambda } => {
                let dist = Exp::new(*lambda).unwrap_or_else(|_| Exp::new(1.0).unwrap());
                dist.sample(rng)
            }
            DistributionType::Poisson { lambda } => {
                let dist = Poisson::new(*lambda).unwrap_or_else(|_| Poisson::new(1.0).unwrap());
                dist.sample(rng) as f64
            }
            DistributionType::Pareto { scale, shape } => {
                let dist =
                    Pareto::new(*scale, *shape).unwrap_or_else(|_| Pareto::new(1.0, 1.0).unwrap());
                dist.sample(rng)
            }
            DistributionType::Beta { alpha, beta } => {
                let dist =
                    Beta::new(*alpha, *beta).unwrap_or_else(|_| Beta::new(1.0, 1.0).unwrap());
                dist.sample(rng)
            }
            DistributionType::Gamma { shape, scale } => {
                let dist =
                    Gamma::new(*shape, *scale).unwrap_or_else(|_| Gamma::new(1.0, 1.0).unwrap());
                dist.sample(rng)
            }
            DistributionType::Weibull { shape, scale } => {
                let dist = Weibull::new(*scale, *shape)
                    .unwrap_or_else(|_| Weibull::new(1.0, 1.0).unwrap());
                dist.sample(rng)
            }
            DistributionType::Categorical { weights } => {
                let total: f64 = weights.iter().sum();
                let mut r = rng.gen::<f64>() * total;
                for (i, w) in weights.iter().enumerate() {
                    r -= w;
                    if r <= 0.0 {
                        return i as f64;
                    }
                }
                (weights.len() - 1) as f64
            }
            DistributionType::Mixture {
                distributions,
                weights,
            } => {
                // Select which distribution to sample from based on weights
                let total: f64 = weights.iter().sum();
                let mut r = rng.gen::<f64>() * total;
                for (i, w) in weights.iter().enumerate() {
                    r -= w;
                    if r <= 0.0 {
                        return distributions[i].sample(rng);
                    }
                }
                distributions.last().unwrap().sample(rng)
            }
        }
    }

    /// Sample multiple values from this distribution.
    pub fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample(rng)).collect()
    }

    /// Sample and apply optional transformations (clipping, rounding).
    pub fn sample_with_constraints<R: Rng>(
        &self,
        rng: &mut R,
        min: Option<f64>,
        max: Option<f64>,
        round_to: Option<u32>,
    ) -> f64 {
        let mut value = self.sample(rng);

        // Apply min/max constraints
        if let Some(min_val) = min {
            value = value.max(min_val);
        }
        if let Some(max_val) = max {
            value = value.min(max_val);
        }

        // Apply rounding
        if let Some(decimals) = round_to {
            let factor = 10_f64.powi(decimals as i32);
            value = (value * factor).round() / factor;
        }

        value
    }
}

/// Sample from a categorical distribution and return the category index.
pub fn sample_categorical<R: Rng>(rng: &mut R, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    let mut r = rng.gen::<f64>() * total;
    for (i, w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

/// Sample from a categorical distribution and return the selected item.
pub fn sample_weighted<'a, T, R: Rng>(
    rng: &mut R,
    items: &'a [T],
    weights: &[f64],
) -> Option<&'a T> {
    if items.is_empty() || weights.is_empty() || items.len() != weights.len() {
        return None;
    }
    let idx = sample_categorical(rng, weights);
    Some(&items[idx])
}

/// Noise models for adding realistic noise to data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NoiseModel {
    /// Gaussian (white) noise with given standard deviation.
    Gaussian { std_dev: f64 },

    /// Heteroscedastic noise where variance scales with signal.
    Heteroscedastic { base_std: f64, scale_factor: f64 },

    /// Quantization noise (rounding to specified precision).
    Quantization { step_size: f64 },

    /// Missing at random with given probability.
    MissingAtRandom { probability: f64 },
}

impl NoiseModel {
    /// Apply noise to a value.
    pub fn apply<R: Rng>(&self, rng: &mut R, value: f64) -> Option<f64> {
        match self {
            NoiseModel::Gaussian { std_dev } => {
                let noise = Normal::new(0.0, *std_dev).unwrap();
                Some(value + noise.sample(rng))
            }
            NoiseModel::Heteroscedastic {
                base_std,
                scale_factor,
            } => {
                let std = base_std + scale_factor * value.abs();
                let noise = Normal::new(0.0, std).unwrap();
                Some(value + noise.sample(rng))
            }
            NoiseModel::Quantization { step_size } => Some((value / step_size).round() * step_size),
            NoiseModel::MissingAtRandom { probability } => {
                if rng.gen::<f64>() < *probability {
                    None
                } else {
                    Some(value)
                }
            }
        }
    }
}

/// Field distribution configuration for customizing generated data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldConfig {
    /// The probability distribution to use.
    pub distribution: DistributionType,

    /// Optional minimum value (clipping).
    pub min: Option<f64>,

    /// Optional maximum value (clipping).
    pub max: Option<f64>,

    /// Optional decimal places for rounding.
    pub round_to: Option<u32>,

    /// Optional noise model to apply.
    pub noise: Option<NoiseModel>,
}

impl FieldConfig {
    /// Create a new field configuration.
    pub fn new(distribution: DistributionType) -> Self {
        Self {
            distribution,
            min: None,
            max: None,
            round_to: None,
            noise: None,
        }
    }

    /// Set minimum value constraint.
    pub fn with_min(mut self, min: f64) -> Self {
        self.min = Some(min);
        self
    }

    /// Set maximum value constraint.
    pub fn with_max(mut self, max: f64) -> Self {
        self.max = Some(max);
        self
    }

    /// Set rounding precision.
    pub fn with_round_to(mut self, decimals: u32) -> Self {
        self.round_to = Some(decimals);
        self
    }

    /// Set noise model.
    pub fn with_noise(mut self, noise: NoiseModel) -> Self {
        self.noise = Some(noise);
        self
    }

    /// Sample a value according to this configuration.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Option<f64> {
        let mut value =
            self.distribution
                .sample_with_constraints(rng, self.min, self.max, self.round_to);

        // Apply noise if configured
        if let Some(ref noise) = self.noise {
            match noise.apply(rng, value) {
                Some(v) => value = v,
                None => return None, // Missing value
            }
        }

        // Re-apply constraints after noise
        if let Some(min_val) = self.min {
            value = value.max(min_val);
        }
        if let Some(max_val) = self.max {
            value = value.min(max_val);
        }

        Some(value)
    }
}

// Common distribution presets for realistic data generation

/// Sales amount distribution (log-normal, typically right-skewed).
pub fn sales_distribution() -> DistributionType {
    DistributionType::log_normal(4.0, 1.0) // Mean ~$55, skewed
}

/// Quantity distribution (Poisson for count data).
pub fn quantity_distribution() -> DistributionType {
    DistributionType::poisson(10.0)
}

/// Discount distribution (Beta for 0-1 range, skewed toward lower discounts).
pub fn discount_distribution() -> DistributionType {
    DistributionType::beta(2.0, 5.0)
}

/// Profit margin distribution (Normal, can be negative).
pub fn profit_margin_distribution() -> DistributionType {
    DistributionType::normal(0.15, 0.10) // Mean 15% margin
}

/// Customer lifetime value distribution (Pareto/power law).
pub fn clv_distribution() -> DistributionType {
    DistributionType::pareto(100.0, 1.5)
}

/// Inter-arrival time distribution (Exponential for Poisson process).
pub fn arrival_time_distribution() -> DistributionType {
    DistributionType::exponential(0.1)
}

/// Age distribution (Normal, constrained to valid range).
pub fn age_distribution() -> DistributionType {
    DistributionType::normal(40.0, 15.0)
}

/// Salary distribution (Log-normal).
pub fn salary_distribution() -> DistributionType {
    DistributionType::log_normal(10.8, 0.5) // Median ~$50k
}

/// Product rating distribution (Beta, slightly skewed positive).
pub fn rating_distribution() -> DistributionType {
    DistributionType::beta(5.0, 2.0) // Skewed toward higher ratings
}

/// Bimodal income distribution (mixture of two normals).
pub fn bimodal_income_distribution() -> DistributionType {
    DistributionType::mixture(
        vec![
            DistributionType::normal(35000.0, 10000.0),
            DistributionType::normal(85000.0, 20000.0),
        ],
        vec![0.6, 0.4],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_uniform_distribution() {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = DistributionType::uniform(0.0, 100.0);

        for _ in 0..100 {
            let value = dist.sample(&mut rng);
            assert!(value >= 0.0 && value <= 100.0);
        }
    }

    #[test]
    fn test_normal_distribution() {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = DistributionType::normal(50.0, 10.0);

        let samples: Vec<f64> = dist.sample_n(&mut rng, 1000);
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;

        // Mean should be approximately 50
        assert!((mean - 50.0).abs() < 5.0);
    }

    #[test]
    fn test_categorical_distribution() {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = DistributionType::categorical(vec![0.7, 0.2, 0.1]);

        let mut counts = [0; 3];
        for _ in 0..1000 {
            let idx = dist.sample(&mut rng) as usize;
            counts[idx] += 1;
        }

        // Category 0 should be most common
        assert!(counts[0] > counts[1]);
        assert!(counts[1] > counts[2]);
    }

    #[test]
    fn test_mixture_distribution() {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = bimodal_income_distribution();

        let samples: Vec<f64> = dist.sample_n(&mut rng, 1000);

        // Should have values near both modes
        let low_count = samples.iter().filter(|&&x| x < 50000.0).count();
        let high_count = samples.iter().filter(|&&x| x > 70000.0).count();

        assert!(low_count > 200);
        assert!(high_count > 200);
    }

    #[test]
    fn test_sample_with_constraints() {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = DistributionType::normal(50.0, 30.0);

        for _ in 0..100 {
            let value = dist.sample_with_constraints(&mut rng, Some(0.0), Some(100.0), Some(2));
            assert!(value >= 0.0 && value <= 100.0);
            // Check rounding
            assert_eq!(value, (value * 100.0).round() / 100.0);
        }
    }

    #[test]
    fn test_noise_model_gaussian() {
        let mut rng = StdRng::seed_from_u64(42);
        let noise = NoiseModel::Gaussian { std_dev: 1.0 };

        let value = 100.0;
        let noisy_values: Vec<f64> = (0..100)
            .filter_map(|_| noise.apply(&mut rng, value))
            .collect();

        let mean: f64 = noisy_values.iter().sum::<f64>() / noisy_values.len() as f64;
        assert!((mean - value).abs() < 1.0);
    }

    #[test]
    fn test_noise_model_missing() {
        let mut rng = StdRng::seed_from_u64(42);
        let noise = NoiseModel::MissingAtRandom { probability: 0.3 };

        let value = 100.0;
        let results: Vec<Option<f64>> = (0..1000).map(|_| noise.apply(&mut rng, value)).collect();

        let missing_count = results.iter().filter(|x| x.is_none()).count();
        // Should be approximately 30% missing
        assert!(missing_count > 200 && missing_count < 400);
    }

    #[test]
    fn test_field_config() {
        let mut rng = StdRng::seed_from_u64(42);

        let config = FieldConfig::new(DistributionType::normal(50.0, 20.0))
            .with_min(0.0)
            .with_max(100.0)
            .with_round_to(1);

        for _ in 0..100 {
            if let Some(value) = config.sample(&mut rng) {
                assert!(value >= 0.0 && value <= 100.0);
            }
        }
    }

    #[test]
    fn test_sample_weighted() {
        let mut rng = StdRng::seed_from_u64(42);
        let items = ["apple", "banana", "cherry"];
        let weights = [0.5, 0.3, 0.2];

        let mut counts = [0; 3];
        for _ in 0..1000 {
            if let Some(&item) = sample_weighted(&mut rng, &items, &weights) {
                let idx = items.iter().position(|&x| x == item).unwrap();
                counts[idx] += 1;
            }
        }

        // Apple should be most common
        assert!(counts[0] > counts[1]);
        assert!(counts[1] > counts[2]);
    }
}
