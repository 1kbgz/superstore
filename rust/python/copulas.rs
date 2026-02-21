//! Python bindings for copula implementations.

use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::rngs::StdRng;
use rand::SeedableRng;
use superstore::{ClaytonCopula, FrankCopula, GaussianCopula, GumbelCopula};

/// Gaussian (Normal) Copula.
///
/// Uses multivariate normal distribution to model dependencies.
/// The correlation between variables is specified via a correlation matrix.
///
/// Example:
///     >>> copula = GaussianCopula([[1.0, 0.8], [0.8, 1.0]])
///     >>> samples = copula.sample(100)
///     >>> # Each sample is a list of uniform [0,1] values with the specified correlation
#[pyclass(name = "GaussianCopula")]
pub struct PyGaussianCopula {
    inner: GaussianCopula,
}

#[pymethods]
impl PyGaussianCopula {
    /// Create a new Gaussian copula with the given correlation matrix.
    ///
    /// Args:
    ///     correlation_matrix: Symmetric positive semi-definite matrix with 1s on diagonal
    #[new]
    fn new(correlation_matrix: Vec<Vec<f64>>) -> PyResult<Self> {
        let inner = GaussianCopula::new(correlation_matrix)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get the dimension of the copula.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Generate n samples from the copula.
    ///
    /// Args:
    ///     n: Number of samples to generate
    ///     seed: Optional random seed
    ///
    /// Returns:
    ///     List of n samples, where each sample is a list of d uniform [0,1] values
    #[pyo3(signature = (n, seed=None))]
    fn sample(&self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<Py<PyList>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let samples = self.inner.sample_n(&mut rng, n);
        // Convert Vec<Vec<f64>> to PyList of PyLists
        let result = PyList::empty(py);
        for sample in samples {
            let inner_list = PyList::new(py, &sample)?;
            result.append(inner_list)?;
        }
        Ok(result.into())
    }
}

/// Clayton Copula.
///
/// An Archimedean copula with lower tail dependence.
/// Good for modeling dependencies where extreme low values tend to occur together.
///
/// Example:
///     >>> copula = ClaytonCopula(2.0, 2)  # theta=2, 2 dimensions
///     >>> samples = copula.sample(100)
#[pyclass(name = "ClaytonCopula")]
pub struct PyClaytonCopula {
    inner: ClaytonCopula,
}

#[pymethods]
impl PyClaytonCopula {
    /// Create a new Clayton copula.
    ///
    /// Args:
    ///     theta: Dependence parameter (must be > 0)
    ///     dim: Number of dimensions (must be >= 2)
    #[new]
    fn new(theta: f64, dim: usize) -> PyResult<Self> {
        let inner = ClaytonCopula::new(theta, dim)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get theta parameter.
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta()
    }

    /// Get the dimension of the copula.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Get Kendall's tau (measure of correlation).
    fn kendalls_tau(&self) -> f64 {
        self.inner.kendalls_tau()
    }

    /// Generate n samples from the copula.
    #[pyo3(signature = (n, seed=None))]
    fn sample(&self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<Py<PyList>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let samples = self.inner.sample_n(&mut rng, n);
        let result = PyList::empty(py);
        for sample in samples {
            let inner_list = PyList::new(py, &sample)?;
            result.append(inner_list)?;
        }
        Ok(result.into())
    }
}

/// Frank Copula.
///
/// An Archimedean copula with symmetric tail dependence.
/// Good for modeling overall dependence without tail asymmetry.
///
/// Example:
///     >>> copula = FrankCopula(5.0)  # positive dependence
///     >>> samples = copula.sample(100)
#[pyclass(name = "FrankCopula")]
pub struct PyFrankCopula {
    inner: FrankCopula,
}

#[pymethods]
impl PyFrankCopula {
    /// Create a new Frank copula.
    ///
    /// Args:
    ///     theta: Dependence parameter (cannot be zero)
    ///         - theta > 0: positive dependence
    ///         - theta < 0: negative dependence
    #[new]
    fn new(theta: f64) -> PyResult<Self> {
        let inner = FrankCopula::new(theta)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get theta parameter.
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta()
    }

    /// Get Kendall's tau (measure of correlation).
    fn kendalls_tau(&self) -> f64 {
        self.inner.kendalls_tau()
    }

    /// Generate n bivariate samples from the copula.
    ///
    /// Returns:
    ///     List of n tuples (u, v), each containing two uniform [0,1] values
    #[pyo3(signature = (n, seed=None))]
    fn sample(&self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<Py<PyList>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let samples = self.inner.sample_n(&mut rng, n);
        let result = PyList::empty(py);
        for (u, v) in samples {
            let pair = PyList::new(py, &[u, v])?;
            result.append(pair)?;
        }
        Ok(result.into())
    }
}

/// Gumbel Copula.
///
/// An Archimedean copula with upper tail dependence.
/// Good for modeling dependencies where extreme high values tend to occur together.
///
/// Example:
///     >>> copula = GumbelCopula(2.0)  # theta=2 means moderate upper tail dependence
///     >>> samples = copula.sample(100)
#[pyclass(name = "GumbelCopula")]
pub struct PyGumbelCopula {
    inner: GumbelCopula,
}

#[pymethods]
impl PyGumbelCopula {
    /// Create a new Gumbel copula.
    ///
    /// Args:
    ///     theta: Dependence parameter (must be >= 1)
    ///         - theta = 1: independence
    ///         - theta > 1: positive dependence
    #[new]
    fn new(theta: f64) -> PyResult<Self> {
        let inner = GumbelCopula::new(theta)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get theta parameter.
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta()
    }

    /// Get Kendall's tau (measure of correlation).
    fn kendalls_tau(&self) -> f64 {
        self.inner.kendalls_tau()
    }

    /// Get upper tail dependence coefficient.
    fn upper_tail_dependence(&self) -> f64 {
        self.inner.upper_tail_dependence()
    }

    /// Generate n bivariate samples from the copula.
    ///
    /// Returns:
    ///     List of n tuples (u, v), each containing two uniform [0,1] values
    #[pyo3(signature = (n, seed=None))]
    fn sample(&self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<Py<PyList>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let samples = self.inner.sample_n(&mut rng, n);
        let result = PyList::empty(py);
        for (u, v) in samples {
            let pair = PyList::new(py, &[u, v])?;
            result.append(pair)?;
        }
        Ok(result.into())
    }
}

/// Register copula classes with Python module.
pub fn register_copulas(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyGaussianCopula>()?;
    m.add_class::<PyClaytonCopula>()?;
    m.add_class::<PyFrankCopula>()?;
    m.add_class::<PyGumbelCopula>()?;
    Ok(())
}
