//! Python bindings for correlation matrix support.

#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::types::PyList;

use superstore::correlation::{pearson_correlation, sample_bivariate, CorrelationMatrix};

/// A correlation matrix for generating correlated multivariate normal data.
///
/// The matrix stores the Cholesky decomposition (lower triangular) which is used
/// to transform independent standard normal samples into correlated samples.
#[pyclass(name = "CorrelationMatrix")]
pub struct PyCorrelationMatrix {
    inner: CorrelationMatrix,
}

#[pymethods]
impl PyCorrelationMatrix {
    /// Create a correlation matrix from a flat row-major list.
    ///
    /// # Arguments
    /// * `data` - Correlation values in row-major order (must be n*n elements)
    /// * `names` - Optional variable names
    ///
    /// # Example
    /// ```python
    /// # 2x2 correlation matrix: [[1.0, 0.8], [0.8, 1.0]]
    /// corr = CorrelationMatrix([1.0, 0.8, 0.8, 1.0])
    /// ```
    #[new]
    #[pyo3(signature = (data, names=None))]
    fn new(data: Vec<f64>, names: Option<Vec<String>>) -> PyResult<Self> {
        let inner = CorrelationMatrix::new(&data, names)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create an identity correlation matrix (uncorrelated variables).
    #[staticmethod]
    #[pyo3(signature = (dim, names=None))]
    fn identity(dim: usize, names: Option<Vec<String>>) -> PyResult<Self> {
        let inner = CorrelationMatrix::identity(dim, names)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create a correlation matrix where all off-diagonal elements have the same value.
    ///
    /// # Arguments
    /// * `dim` - Number of variables
    /// * `rho` - Correlation coefficient for all pairs
    /// * `names` - Optional variable names
    #[staticmethod]
    #[pyo3(signature = (dim, rho, names=None))]
    fn uniform(dim: usize, rho: f64, names: Option<Vec<String>>) -> PyResult<Self> {
        let inner = CorrelationMatrix::uniform(dim, rho, names)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create a correlation matrix with AR(1) (exponentially decaying) correlations.
    ///
    /// The correlation between variable i and j is rho^|i-j|.
    ///
    /// # Arguments
    /// * `dim` - Number of variables
    /// * `rho` - Base correlation (correlation between adjacent variables)
    /// * `names` - Optional variable names
    #[staticmethod]
    #[pyo3(signature = (dim, rho, names=None))]
    fn ar1(dim: usize, rho: f64, names: Option<Vec<String>>) -> PyResult<Self> {
        let inner = CorrelationMatrix::ar1(dim, rho, names)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get the dimension (number of variables).
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Get the variable names if set.
    #[getter]
    fn names(&self) -> Option<Vec<String>> {
        self.inner.names().map(|n| n.to_vec())
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
    /// A list of samples, where each sample is a list of values.
    #[pyo3(signature = (n, means, std_devs, seed=None))]
    fn sample(
        &self,
        py: Python<'_>,
        n: usize,
        means: Vec<f64>,
        std_devs: Vec<f64>,
        seed: Option<u64>,
    ) -> PyResult<Py<PyList>> {
        let samples = self
            .inner
            .sample(n, &means, &std_devs, seed)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let result = PyList::empty(py);
        for sample in samples {
            let inner_list = PyList::new(py, &sample)?;
            result.append(inner_list)?;
        }
        Ok(result.into())
    }

    /// Sample and return as columns (for DataFrame-like access).
    ///
    /// # Returns
    /// A list of columns, where each column is a list of values for one variable.
    #[pyo3(signature = (n, means, std_devs, seed=None))]
    fn sample_columns(
        &self,
        py: Python<'_>,
        n: usize,
        means: Vec<f64>,
        std_devs: Vec<f64>,
        seed: Option<u64>,
    ) -> PyResult<Py<PyList>> {
        let columns = self
            .inner
            .sample_columns(n, &means, &std_devs, seed)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let result = PyList::empty(py);
        for column in columns {
            let inner_list = PyList::new(py, &column)?;
            result.append(inner_list)?;
        }
        Ok(result.into())
    }
}

/// Generate correlated bivariate normal data.
///
/// This is a convenience function for the common case of generating two correlated variables.
///
/// # Arguments
/// * `n` - Number of samples
/// * `rho` - Correlation coefficient between the two variables
/// * `mean1`, `std1` - Mean and standard deviation for first variable
/// * `mean2`, `std2` - Mean and standard deviation for second variable
/// * `seed` - Optional random seed
///
/// # Returns
/// A tuple of two lists (x, y)
#[pyfunction]
#[pyo3(signature = (n, rho, mean1=0.0, std1=1.0, mean2=0.0, std2=1.0, seed=None))]
fn sampleBivariate(
    py: Python<'_>,
    n: usize,
    rho: f64,
    mean1: f64,
    std1: f64,
    mean2: f64,
    std2: f64,
    seed: Option<u64>,
) -> PyResult<(Py<PyList>, Py<PyList>)> {
    let (x, y) = sample_bivariate(n, rho, mean1, std1, mean2, std2, seed)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let x_list = PyList::new(py, &x)?;
    let y_list = PyList::new(py, &y)?;
    Ok((x_list.into(), y_list.into()))
}

/// Compute the Pearson correlation coefficient between two lists.
///
/// # Arguments
/// * `x` - First variable
/// * `y` - Second variable
///
/// # Returns
/// The correlation coefficient (between -1 and 1)
#[pyfunction]
fn pearsonCorrelation(x: Vec<f64>, y: Vec<f64>) -> f64 {
    pearson_correlation(&x, &y)
}

/// Register correlation functions with the Python module.
pub fn register_correlation(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyCorrelationMatrix>()?;
    m.add_function(wrap_pyfunction!(sampleBivariate, m)?)?;
    m.add_function(wrap_pyfunction!(pearsonCorrelation, m)?)?;
    Ok(())
}
