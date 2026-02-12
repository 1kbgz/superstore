//! Correlation matrix support for generating correlated multivariate data.
//!
//! This module provides functions to generate correlated random variables using
//! Cholesky decomposition of covariance matrices.

use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// Error type for correlation operations
#[derive(Debug, Clone)]
pub enum CorrelationError {
    /// Matrix is not positive definite (cannot be Cholesky decomposed)
    NotPositiveDefinite,
    /// Matrix dimensions don't match
    DimensionMismatch { expected: usize, got: usize },
    /// Invalid correlation value (must be between -1 and 1)
    InvalidCorrelation(f64),
    /// Matrix is not square
    NotSquare { rows: usize, cols: usize },
}

impl std::fmt::Display for CorrelationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorrelationError::NotPositiveDefinite => {
                write!(f, "Correlation matrix is not positive definite")
            }
            CorrelationError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            CorrelationError::InvalidCorrelation(v) => {
                write!(
                    f,
                    "Invalid correlation value: {} (must be between -1 and 1)",
                    v
                )
            }
            CorrelationError::NotSquare { rows, cols } => {
                write!(f, "Matrix is not square: {} x {}", rows, cols)
            }
        }
    }
}

impl std::error::Error for CorrelationError {}

/// A correlation matrix for generating correlated multivariate normal data.
///
/// The matrix stores the Cholesky decomposition (lower triangular) which is used
/// to transform independent standard normal samples into correlated samples.
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    /// Cholesky decomposition (lower triangular matrix)
    cholesky: DMatrix<f64>,
    /// Number of variables
    dim: usize,
    /// Variable names (optional)
    names: Option<Vec<String>>,
}

impl CorrelationMatrix {
    /// Create a correlation matrix from a flat row-major array.
    ///
    /// # Arguments
    /// * `data` - Correlation values in row-major order (must be n*n elements)
    /// * `names` - Optional variable names
    ///
    /// # Example
    /// ```
    /// use superstore::correlation::CorrelationMatrix;
    ///
    /// // 2x2 correlation matrix: [[1.0, 0.8], [0.8, 1.0]]
    /// let corr = CorrelationMatrix::new(&[1.0, 0.8, 0.8, 1.0], None).unwrap();
    /// ```
    pub fn new(data: &[f64], names: Option<Vec<String>>) -> Result<Self, CorrelationError> {
        let n = (data.len() as f64).sqrt() as usize;
        if n * n != data.len() {
            return Err(CorrelationError::NotSquare {
                rows: n,
                cols: data.len() / n.max(1),
            });
        }

        // Validate correlation values
        for &v in data {
            if !(-1.0..=1.0).contains(&v) {
                return Err(CorrelationError::InvalidCorrelation(v));
            }
        }

        // Validate names dimension if provided
        if let Some(ref n_vec) = names {
            if n_vec.len() != n {
                return Err(CorrelationError::DimensionMismatch {
                    expected: n,
                    got: n_vec.len(),
                });
            }
        }

        let matrix = DMatrix::from_row_slice(n, n, data);

        // Attempt Cholesky decomposition
        let cholesky = matrix
            .clone()
            .cholesky()
            .ok_or(CorrelationError::NotPositiveDefinite)?
            .l();

        Ok(Self {
            cholesky,
            dim: n,
            names,
        })
    }

    /// Create an identity correlation matrix (uncorrelated variables).
    pub fn identity(dim: usize, names: Option<Vec<String>>) -> Result<Self, CorrelationError> {
        if let Some(ref n_vec) = names {
            if n_vec.len() != dim {
                return Err(CorrelationError::DimensionMismatch {
                    expected: dim,
                    got: n_vec.len(),
                });
            }
        }

        let cholesky = DMatrix::identity(dim, dim);
        Ok(Self {
            cholesky,
            dim,
            names,
        })
    }

    /// Create a correlation matrix where all off-diagonal elements have the same value.
    ///
    /// # Arguments
    /// * `dim` - Number of variables
    /// * `rho` - Correlation coefficient for all pairs
    /// * `names` - Optional variable names
    pub fn uniform(
        dim: usize,
        rho: f64,
        names: Option<Vec<String>>,
    ) -> Result<Self, CorrelationError> {
        if !(-1.0..=1.0).contains(&rho) {
            return Err(CorrelationError::InvalidCorrelation(rho));
        }

        let mut data = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                data[i * dim + j] = if i == j { 1.0 } else { rho };
            }
        }

        Self::new(&data, names)
    }

    /// Create a correlation matrix with exponentially decaying correlations.
    ///
    /// The correlation between variable i and j is rho^|i-j|.
    ///
    /// # Arguments
    /// * `dim` - Number of variables
    /// * `rho` - Base correlation (correlation between adjacent variables)
    /// * `names` - Optional variable names
    pub fn ar1(dim: usize, rho: f64, names: Option<Vec<String>>) -> Result<Self, CorrelationError> {
        if !(-1.0..=1.0).contains(&rho) {
            return Err(CorrelationError::InvalidCorrelation(rho));
        }

        let mut data = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let lag = (i as i32 - j as i32).unsigned_abs() as i32;
                data[i * dim + j] = rho.powi(lag);
            }
        }

        Self::new(&data, names)
    }

    /// Get the dimension (number of variables).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the variable names if set.
    pub fn names(&self) -> Option<&[String]> {
        self.names.as_deref()
    }

    /// Sample correlated multivariate normal data.
    ///
    /// # Arguments
    /// * `n` - Number of samples to generate
    /// * `means` - Mean for each variable (must match dimension)
    /// * `std_devs` - Standard deviation for each variable (must match dimension)
    /// * `seed` - Optional random seed
    ///
    /// # Returns
    /// A Vec of Vec<f64>, where each inner Vec is one sample.
    pub fn sample(
        &self,
        n: usize,
        means: &[f64],
        std_devs: &[f64],
        seed: Option<u64>,
    ) -> Result<Vec<Vec<f64>>, CorrelationError> {
        if means.len() != self.dim {
            return Err(CorrelationError::DimensionMismatch {
                expected: self.dim,
                got: means.len(),
            });
        }
        if std_devs.len() != self.dim {
            return Err(CorrelationError::DimensionMismatch {
                expected: self.dim,
                got: std_devs.len(),
            });
        }

        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut result = Vec::with_capacity(n);

        for _ in 0..n {
            // Generate independent standard normal samples
            let z: Vec<f64> = (0..self.dim)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect();
            let z_vec = DVector::from_vec(z);

            // Transform using Cholesky: y = L * z
            let correlated = &self.cholesky * &z_vec;

            // Scale and shift: x = mean + std_dev * y
            let sample: Vec<f64> = correlated
                .iter()
                .enumerate()
                .map(|(i, &y)| means[i] + std_devs[i] * y)
                .collect();

            result.push(sample);
        }

        Ok(result)
    }

    /// Sample and return as a column-oriented structure (for DataFrame-like access).
    ///
    /// # Returns
    /// A Vec of columns, where each column is a Vec<f64> for one variable.
    pub fn sample_columns(
        &self,
        n: usize,
        means: &[f64],
        std_devs: &[f64],
        seed: Option<u64>,
    ) -> Result<Vec<Vec<f64>>, CorrelationError> {
        let rows = self.sample(n, means, std_devs, seed)?;

        // Transpose: rows -> columns
        let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(n); self.dim];
        for row in rows {
            for (j, value) in row.into_iter().enumerate() {
                columns[j].push(value);
            }
        }

        Ok(columns)
    }
}

/// Helper function to compute sample correlation coefficient between two vectors.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Generate correlated random data using a simple 2-variable correlation.
///
/// This is a convenience function for the common case of generating two correlated variables.
///
/// # Arguments
/// * `n` - Number of samples
/// * `rho` - Correlation coefficient between the two variables
/// * `mean1`, `std1` - Mean and standard deviation for first variable
/// * `mean2`, `std2` - Mean and standard deviation for second variable
/// * `seed` - Optional random seed
pub fn sample_bivariate(
    n: usize,
    rho: f64,
    mean1: f64,
    std1: f64,
    mean2: f64,
    std2: f64,
    seed: Option<u64>,
) -> Result<(Vec<f64>, Vec<f64>), CorrelationError> {
    let corr = CorrelationMatrix::new(&[1.0, rho, rho, 1.0], None)?;
    let columns = corr.sample_columns(n, &[mean1, mean2], &[std1, std2], seed)?;

    Ok((columns[0].clone(), columns[1].clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_correlation() {
        let corr = CorrelationMatrix::identity(3, None).unwrap();
        assert_eq!(corr.dim(), 3);

        let samples = corr
            .sample(1000, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], Some(42))
            .unwrap();
        assert_eq!(samples.len(), 1000);
        assert_eq!(samples[0].len(), 3);
    }

    #[test]
    fn test_uniform_correlation() {
        let corr = CorrelationMatrix::uniform(3, 0.5, None).unwrap();
        assert_eq!(corr.dim(), 3);
    }

    #[test]
    fn test_ar1_correlation() {
        let corr = CorrelationMatrix::ar1(4, 0.9, None).unwrap();
        assert_eq!(corr.dim(), 4);
    }

    #[test]
    fn test_correlation_validation() {
        // Invalid correlation value
        assert!(CorrelationMatrix::new(&[1.0, 1.5, 1.5, 1.0], None).is_err());

        // Non-positive definite
        assert!(CorrelationMatrix::new(
            &[1.0, 0.99, 0.99, 1.0, 0.99, -0.99, 0.99, -0.99, 1.0],
            None
        )
        .is_err());
    }

    #[test]
    fn test_bivariate_correlation() {
        let (x, y) = sample_bivariate(10000, 0.8, 0.0, 1.0, 0.0, 1.0, Some(42)).unwrap();

        assert_eq!(x.len(), 10000);
        assert_eq!(y.len(), 10000);

        // Check correlation is approximately 0.8
        let r = pearson_correlation(&x, &y);
        assert!(
            (r - 0.8).abs() < 0.05,
            "Expected correlation ~0.8, got {}",
            r
        );
    }

    #[test]
    fn test_pearson_correlation() {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-10);

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r_neg = pearson_correlation(&x, &y_neg);
        assert!((r_neg + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_reproducibility() {
        let corr = CorrelationMatrix::uniform(3, 0.7, None).unwrap();

        let s1 = corr
            .sample(100, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], Some(12345))
            .unwrap();
        let s2 = corr
            .sample(100, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], Some(12345))
            .unwrap();

        for (r1, r2) in s1.iter().zip(s2.iter()) {
            for (v1, v2) in r1.iter().zip(r2.iter()) {
                assert_eq!(v1, v2);
            }
        }
    }

    #[test]
    fn test_named_variables() {
        let names = vec![
            "sales".to_string(),
            "profit".to_string(),
            "quantity".to_string(),
        ];
        let corr = CorrelationMatrix::identity(3, Some(names.clone())).unwrap();

        assert_eq!(corr.names(), Some(names.as_slice()));
    }

    #[test]
    fn test_sample_columns() {
        let corr = CorrelationMatrix::identity(2, None).unwrap();
        let cols = corr
            .sample_columns(100, &[0.0, 10.0], &[1.0, 2.0], Some(42))
            .unwrap();

        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].len(), 100);
        assert_eq!(cols[1].len(), 100);

        // Check means are approximately correct
        let mean0: f64 = cols[0].iter().sum::<f64>() / 100.0;
        let mean1: f64 = cols[1].iter().sum::<f64>() / 100.0;

        assert!((mean0 - 0.0).abs() < 0.5);
        assert!((mean1 - 10.0).abs() < 1.0);
    }
}
