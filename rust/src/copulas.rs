//! Copula implementations for generating data with complex dependency structures.
//!
//! Copulas allow modeling dependencies between variables separately from their
//! marginal distributions. This enables creating realistic multivariate data
//! with any combination of marginal distributions and dependency structures.

use rand::prelude::*;
use rand_distr::{Distribution, Normal, StandardNormal, Uniform};
use std::f64::consts::PI;

/// Error type for copula operations.
#[derive(Debug, Clone)]
pub struct CopulaError(String);

impl std::fmt::Display for CopulaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CopulaError {}

/// Gaussian (Normal) Copula.
///
/// Uses the multivariate normal distribution to model dependencies.
/// The correlation between variables is specified via a correlation matrix.
pub struct GaussianCopula {
    /// Correlation matrix (lower triangular Cholesky decomposition)
    cholesky: Vec<Vec<f64>>,
    /// Dimension
    dim: usize,
}

impl GaussianCopula {
    /// Create a new Gaussian copula with the given correlation matrix.
    ///
    /// The correlation matrix must be positive semi-definite with ones on the diagonal.
    pub fn new(correlation_matrix: Vec<Vec<f64>>) -> Result<Self, CopulaError> {
        let dim = correlation_matrix.len();
        if dim == 0 {
            return Err(CopulaError(
                "Correlation matrix cannot be empty".to_string(),
            ));
        }

        // Validate and compute Cholesky decomposition
        for (i, row) in correlation_matrix.iter().enumerate() {
            if row.len() != dim {
                return Err(CopulaError(format!(
                    "Correlation matrix must be square, row {} has {} elements instead of {}",
                    i,
                    row.len(),
                    dim
                )));
            }
            // Check diagonal is 1
            if (row[i] - 1.0).abs() > 1e-10 {
                return Err(CopulaError(format!(
                    "Diagonal elements must be 1, got {} at position ({}, {})",
                    row[i], i, i
                )));
            }
            // Check symmetry
            for j in 0..i {
                if (row[j] - correlation_matrix[j][i]).abs() > 1e-10 {
                    return Err(CopulaError(format!(
                        "Correlation matrix must be symmetric: ({},{})={} vs ({},{})={}",
                        i, j, row[j], j, i, correlation_matrix[j][i]
                    )));
                }
            }
        }

        // Compute Cholesky decomposition
        let cholesky = Self::cholesky_decompose(&correlation_matrix)?;

        Ok(Self { cholesky, dim })
    }

    /// Compute the Cholesky decomposition of a positive semi-definite matrix.
    fn cholesky_decompose(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, CopulaError> {
        let n = matrix.len();
        let mut l = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i][k] * l[j][k];
                }

                if i == j {
                    let val = matrix[i][i] - sum;
                    if val < 0.0 {
                        return Err(CopulaError(format!(
                            "Matrix is not positive semi-definite at position ({}, {})",
                            i, i
                        )));
                    }
                    l[i][j] = val.sqrt();
                } else {
                    if l[j][j] == 0.0 {
                        l[i][j] = 0.0;
                    } else {
                        l[i][j] = (matrix[i][j] - sum) / l[j][j];
                    }
                }
            }
        }

        Ok(l)
    }

    /// Generate n samples from the copula.
    ///
    /// Returns a vector of n samples, where each sample is a vector of d uniform [0,1] values.
    pub fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> Vec<Vec<f64>> {
        let normal = StandardNormal;
        let mut samples = Vec::with_capacity(n);

        for _ in 0..n {
            // Generate independent standard normal variates
            let z: Vec<f64> = (0..self.dim).map(|_| normal.sample(rng)).collect();

            // Apply Cholesky decomposition to get correlated normals
            let mut correlated = vec![0.0; self.dim];
            for i in 0..self.dim {
                for j in 0..=i {
                    correlated[i] += self.cholesky[i][j] * z[j];
                }
            }

            // Transform to uniform using the normal CDF
            let uniforms: Vec<f64> = correlated.iter().map(|&x| norm_cdf(x)).collect();

            samples.push(uniforms);
        }

        samples
    }

    /// Dimension of the copula.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Clayton Copula.
///
/// An Archimedean copula with lower tail dependence.
/// Good for modeling dependencies where extreme low values tend to occur together.
pub struct ClaytonCopula {
    /// Dependence parameter (theta > 0 for positive dependence)
    theta: f64,
    /// Dimension
    dim: usize,
}

impl ClaytonCopula {
    /// Create a new Clayton copula with the given theta parameter.
    ///
    /// theta > 0: positive dependence
    /// theta -> 0: independence
    /// theta -> infinity: perfect positive dependence
    pub fn new(theta: f64, dim: usize) -> Result<Self, CopulaError> {
        if theta <= 0.0 {
            return Err(CopulaError(format!(
                "Clayton copula theta must be positive, got {}",
                theta
            )));
        }
        if dim < 2 {
            return Err(CopulaError(format!(
                "Dimension must be at least 2, got {}",
                dim
            )));
        }
        Ok(Self { theta, dim })
    }

    /// Generate n samples from the copula.
    pub fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> Vec<Vec<f64>> {
        let uniform = Uniform::new(0.0f64, 1.0f64);
        let mut samples = Vec::with_capacity(n);

        for _ in 0..n {
            // Sample from gamma distribution with shape 1/theta
            let gamma_sample = sample_gamma(rng, 1.0 / self.theta, 1.0);

            // Generate independent exponential(1) variates
            let exp_variates: Vec<f64> = (0..self.dim).map(|_| -uniform.sample(rng).ln()).collect();

            // Apply the Clayton copula transformation
            let uniforms: Vec<f64> = exp_variates
                .iter()
                .map(|&e| ((1.0 + e / gamma_sample) as f64).powf(-1.0 / self.theta))
                .collect();

            samples.push(uniforms);
        }

        samples
    }

    /// Get the theta parameter.
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Dimension of the copula.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Kendall's tau for the copula.
    pub fn kendalls_tau(&self) -> f64 {
        self.theta / (self.theta + 2.0)
    }
}

/// Frank Copula.
///
/// An Archimedean copula with symmetric tail dependence.
/// Good for modeling overall dependence without tail asymmetry.
pub struct FrankCopula {
    /// Dependence parameter (theta != 0)
    theta: f64,
}

impl FrankCopula {
    /// Create a new Frank copula with the given theta parameter.
    ///
    /// theta > 0: positive dependence
    /// theta < 0: negative dependence
    /// theta -> 0: independence
    pub fn new(theta: f64) -> Result<Self, CopulaError> {
        if theta.abs() < 1e-10 {
            return Err(CopulaError("Frank copula theta cannot be zero".to_string()));
        }
        Ok(Self { theta })
    }

    /// Generate n samples from the bivariate Frank copula.
    pub fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> Vec<(f64, f64)> {
        let uniform = Uniform::new(0.0f64, 1.0f64);
        let mut samples = Vec::with_capacity(n);

        for _ in 0..n {
            let u = uniform.sample(rng);
            let v = uniform.sample(rng);

            // Apply the inverse Frank copula transformation
            let w = if self.theta.abs() > 1e-10 {
                let exp_theta = (-self.theta).exp();
                let exp_theta_u = (-self.theta * u).exp();
                let numerator =
                    -((1.0 - exp_theta) / ((v.recip() - 1.0) * exp_theta_u + 1.0) + exp_theta).ln();
                numerator / self.theta
            } else {
                v // Independence case
            };

            samples.push((u, w.clamp(0.0, 1.0)));
        }

        samples
    }

    /// Get the theta parameter.
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Kendall's tau for the copula.
    pub fn kendalls_tau(&self) -> f64 {
        if self.theta.abs() < 1e-10 {
            0.0
        } else {
            1.0 - 4.0 / self.theta * (1.0 - debye_function(self.theta))
        }
    }
}

/// Gumbel Copula.
///
/// An Archimedean copula with upper tail dependence.
/// Good for modeling dependencies where extreme high values tend to occur together.
pub struct GumbelCopula {
    /// Dependence parameter (theta >= 1)
    theta: f64,
}

impl GumbelCopula {
    /// Create a new Gumbel copula with the given theta parameter.
    ///
    /// theta = 1: independence
    /// theta > 1: positive dependence
    /// theta -> infinity: perfect positive dependence
    pub fn new(theta: f64) -> Result<Self, CopulaError> {
        if theta < 1.0 {
            return Err(CopulaError(format!(
                "Gumbel copula theta must be >= 1, got {}",
                theta
            )));
        }
        Ok(Self { theta })
    }

    /// Generate n samples from the bivariate Gumbel copula.
    pub fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> Vec<(f64, f64)> {
        let uniform = Uniform::new(0.0f64, 1.0f64);
        let mut samples = Vec::with_capacity(n);

        for _ in 0..n {
            // Generate stable distribution sample for Gumbel
            let s = sample_stable(rng, 1.0 / self.theta);

            // Generate independent exponential variates
            let e1 = -uniform.sample(rng).ln();
            let e2 = -uniform.sample(rng).ln();

            // Apply transformation
            let u = (-((e1 / s).powf(self.theta))).exp();
            let v = (-((e2 / s).powf(self.theta))).exp();

            samples.push((u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)));
        }

        samples
    }

    /// Get the theta parameter.
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Kendall's tau for the copula.
    pub fn kendalls_tau(&self) -> f64 {
        1.0 - 1.0 / self.theta
    }

    /// Upper tail dependence coefficient.
    pub fn upper_tail_dependence(&self) -> f64 {
        2.0 - 2.0f64.powf(1.0 / self.theta)
    }
}

// Helper functions

/// Standard normal CDF approximation.
fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0f64.sqrt()))
}

/// Error function approximation.
fn erf(x: f64) -> f64 {
    // Horner's method approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Sample from gamma distribution using Marsaglia and Tsang's method.
fn sample_gamma<R: Rng>(rng: &mut R, shape: f64, scale: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();

    if shape < 1.0 {
        // Use Ahrens-Dieter method for shape < 1
        let u = rng.gen::<f64>();
        sample_gamma(rng, 1.0 + shape, scale) * u.powf(1.0 / shape)
    } else {
        // Marsaglia and Tsang's method
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        loop {
            let x = normal.sample(rng);
            let v = 1.0 + c * x;
            if v > 0.0 {
                let v = v * v * v;
                let u = rng.gen::<f64>();

                if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                    return d * v * scale;
                }
                if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                    return d * v * scale;
                }
            }
        }
    }
}

/// Sample from a stable distribution (for Gumbel copula).
fn sample_stable<R: Rng>(rng: &mut R, alpha: f64) -> f64 {
    let uniform = Uniform::new(-PI / 2.0, PI / 2.0);
    let exp_dist = rand_distr::Exp::new(1.0).unwrap();

    let u = uniform.sample(rng);
    let w = exp_dist.sample(rng);

    if (alpha - 1.0).abs() < 1e-10 {
        // Cauchy distribution (alpha = 1)
        u.tan()
    } else {
        // General stable distribution
        let t = alpha * u;
        let s = (PI / 2.0 * alpha).cos().powf(1.0 / alpha);
        let x = (t.sin() / u.cos().powf(1.0 / alpha))
            * (((1.0 - alpha) * u).cos() / w).powf((1.0 - alpha) / alpha);
        s * x
    }
}

/// Debye function D_1(x) for Frank copula Kendall's tau calculation.
fn debye_function(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0
    } else {
        // Numerical integration approximation
        let n = 100;
        let dx = x / n as f64;
        let mut sum = 0.0;
        for i in 0..n {
            let t = (i as f64 + 0.5) * dx;
            sum += t / (t.exp() - 1.0);
        }
        sum * dx / x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_copula_creation() {
        let corr = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let copula = GaussianCopula::new(corr).unwrap();
        assert_eq!(copula.dim(), 2);
    }

    #[test]
    fn test_gaussian_copula_sampling() {
        let corr = vec![vec![1.0, 0.8], vec![0.8, 1.0]];
        let copula = GaussianCopula::new(corr).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let samples = copula.sample_n(&mut rng, 100);

        assert_eq!(samples.len(), 100);
        for sample in &samples {
            assert_eq!(sample.len(), 2);
            for &u in sample {
                assert!(u >= 0.0 && u <= 1.0);
            }
        }
    }

    #[test]
    fn test_clayton_copula() {
        let copula = ClaytonCopula::new(2.0, 2).unwrap();
        assert!((copula.kendalls_tau() - 0.5).abs() < 0.01);

        let mut rng = StdRng::seed_from_u64(42);
        let samples = copula.sample_n(&mut rng, 100);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_frank_copula() {
        let copula = FrankCopula::new(5.0).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let samples = copula.sample_n(&mut rng, 100);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_gumbel_copula() {
        let copula = GumbelCopula::new(2.0).unwrap();
        assert!((copula.kendalls_tau() - 0.5).abs() < 0.01);

        let mut rng = StdRng::seed_from_u64(42);
        let samples = copula.sample_n(&mut rng, 100);
        assert_eq!(samples.len(), 100);
    }
}
