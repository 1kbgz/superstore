//! Python bindings for temporal dependencies.

use pyo3::prelude::*;
use pyo3::types::PyList;

use rand::rngs::StdRng;
use rand::SeedableRng;

use superstore::temporal::{ARp, ExponentialSmoothing, MarkovChain, RandomWalk, AR1};

/// AR(1) autoregressive model for generating temporally dependent data.
///
/// Generates values according to: x_t = mean + phi * (x_{t-1} - mean) + epsilon_t
/// where epsilon_t ~ N(0, sigma^2)
#[pyclass(name = "AR1")]
pub struct PyAR1 {
    inner: AR1,
}

#[pymethods]
impl PyAR1 {
    /// Create a new AR(1) model.
    ///
    /// Args:
    ///     phi: Autoregressive coefficient (-1 < phi < 1 for stationarity)
    ///     sigma: Standard deviation of innovations
    ///     mean: Mean of the process (default: 0.0)
    ///
    /// Example:
    ///     >>> ar = AR1(phi=0.9, sigma=1.0)  # High persistence
    ///     >>> values = ar.sample(100)
    #[new]
    #[pyo3(signature = (phi, sigma, mean=0.0))]
    fn new(phi: f64, sigma: f64, mean: f64) -> PyResult<Self> {
        let inner = AR1::new(phi, sigma, mean)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Reset the state to the mean.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get the current state.
    #[getter]
    fn state(&self) -> f64 {
        self.inner.state()
    }

    /// Set the current state.
    #[setter]
    fn set_state(&mut self, state: f64) {
        self.inner.set_state(state);
    }

    /// Get the phi coefficient.
    #[getter]
    fn phi(&self) -> f64 {
        self.inner.phi
    }

    /// Get the sigma value.
    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.sigma
    }

    /// Get the mean.
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean
    }

    /// Get the stationary variance of the process.
    fn stationary_variance(&self) -> f64 {
        self.inner.stationary_variance()
    }

    /// Generate n samples.
    ///
    /// Args:
    ///     n: Number of samples to generate
    ///     seed: Optional random seed
    ///
    /// Returns:
    ///     List of n values
    #[pyo3(signature = (n, seed=None))]
    fn sample(&mut self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<Py<PyList>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let samples = self.inner.sample_n(&mut rng, n);
        Ok(PyList::new(py, &samples)?.into())
    }
}

/// AR(p) autoregressive model of order p.
///
/// Generates values according to: x_t = mean + sum_i(phi_i * (x_{t-i} - mean)) + epsilon_t
#[pyclass(name = "ARp")]
pub struct PyARp {
    inner: ARp,
}

#[pymethods]
impl PyARp {
    /// Create a new AR(p) model.
    ///
    /// Args:
    ///     coefficients: List of AR coefficients [phi_1, phi_2, ..., phi_p]
    ///     sigma: Standard deviation of innovations
    ///     mean: Mean of the process (default: 0.0)
    #[new]
    #[pyo3(signature = (coefficients, sigma, mean=0.0))]
    fn new(coefficients: Vec<f64>, sigma: f64, mean: f64) -> PyResult<Self> {
        let inner = ARp::new(coefficients, sigma, mean)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create an AR(2) model.
    #[staticmethod]
    #[pyo3(signature = (phi1, phi2, sigma, mean=0.0))]
    fn ar2(phi1: f64, phi2: f64, sigma: f64, mean: f64) -> PyResult<Self> {
        let inner = ARp::ar2(phi1, phi2, sigma, mean)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get the order of the AR model.
    fn order(&self) -> usize {
        self.inner.order()
    }

    /// Reset the state to the mean.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Generate n samples.
    #[pyo3(signature = (n, seed=None))]
    fn sample(&mut self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<Py<PyList>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let samples = self.inner.sample_n(&mut rng, n);
        Ok(PyList::new(py, &samples)?.into())
    }
}

/// Markov chain for generating temporally dependent categorical data.
#[pyclass(name = "MarkovChain")]
pub struct PyMarkovChain {
    inner: MarkovChain,
}

#[pymethods]
impl PyMarkovChain {
    /// Create a new Markov chain.
    ///
    /// Args:
    ///     transition_matrix: Row-stochastic transition matrix (rows must sum to 1)
    ///     states: List of state labels
    ///
    /// Example:
    ///     >>> # Weather model: P(sunny->rainy)=0.3, P(rainy->sunny)=0.5
    ///     >>> mc = MarkovChain(
    ///     ...     [[0.7, 0.3], [0.5, 0.5]],
    ///     ...     ["sunny", "rainy"]
    ///     ... )
    #[new]
    fn new(transition_matrix: Vec<Vec<f64>>, states: Vec<String>) -> PyResult<Self> {
        let inner = MarkovChain::new(transition_matrix, states)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create a simple two-state Markov chain.
    ///
    /// Args:
    ///     state_a: Name of first state
    ///     state_b: Name of second state
    ///     prob_a_to_b: Probability of transitioning from A to B
    ///     prob_b_to_a: Probability of transitioning from B to A
    #[staticmethod]
    fn two_state(
        state_a: &str,
        state_b: &str,
        prob_a_to_b: f64,
        prob_b_to_a: f64,
    ) -> PyResult<Self> {
        let inner = MarkovChain::two_state(state_a, state_b, prob_a_to_b, prob_b_to_a)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get all states.
    fn states(&self) -> Vec<String> {
        self.inner.states().to_vec()
    }

    /// Get current state.
    #[getter]
    fn current_state(&self) -> String {
        self.inner.current_state().to_string()
    }

    /// Set current state by name.
    fn set_state(&mut self, state: &str) -> PyResult<()> {
        self.inner
            .set_state_by_name(state)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Get stationary distribution.
    fn stationary_distribution(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let dist = self.inner.stationary_distribution();
        Ok(PyList::new(py, &dist)?.into())
    }

    /// Generate n state transitions.
    #[pyo3(signature = (n, seed=None))]
    fn sample(&mut self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<Py<PyList>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let samples = self.inner.sample_n(&mut rng, n);
        Ok(PyList::new(py, &samples)?.into())
    }

    /// Generate n state transitions as indices.
    #[pyo3(signature = (n, seed=None))]
    fn sample_indices(
        &mut self,
        py: Python<'_>,
        n: usize,
        seed: Option<u64>,
    ) -> PyResult<Py<PyList>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let samples = self.inner.sample_n_indices(&mut rng, n);
        Ok(PyList::new(py, &samples)?.into())
    }
}

/// Random walk model.
#[pyclass(name = "RandomWalk")]
pub struct PyRandomWalk {
    inner: RandomWalk,
}

#[pymethods]
impl PyRandomWalk {
    /// Create a new random walk.
    ///
    /// Args:
    ///     sigma: Step standard deviation
    ///     start: Starting position (default: 0.0)
    ///     drift: Mean step size (default: 0.0)
    #[new]
    #[pyo3(signature = (sigma, start=0.0, drift=0.0))]
    fn new(sigma: f64, start: f64, drift: f64) -> PyResult<Self> {
        let inner = RandomWalk::new(sigma, start, drift)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get current position.
    #[getter]
    fn position(&self) -> f64 {
        self.inner.position()
    }

    /// Set current position.
    #[setter]
    fn set_position(&mut self, position: f64) {
        self.inner.set_position(position);
    }

    /// Generate n samples.
    #[pyo3(signature = (n, seed=None))]
    fn sample(&mut self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<Py<PyList>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let samples = self.inner.sample_n(&mut rng, n);
        Ok(PyList::new(py, &samples)?.into())
    }
}

/// Exponential smoothing generator for smooth trend generation.
#[pyclass(name = "ExponentialSmoothing")]
pub struct PyExponentialSmoothing {
    inner: ExponentialSmoothing,
}

#[pymethods]
impl PyExponentialSmoothing {
    /// Create a new exponential smoothing model.
    ///
    /// Args:
    ///     alpha: Smoothing factor (0 < alpha <= 1). Smaller = smoother.
    ///     sigma: Standard deviation of random innovations
    ///     initial: Initial smoothed value (default: 0.0)
    #[new]
    #[pyo3(signature = (alpha, sigma, initial=0.0))]
    fn new(alpha: f64, sigma: f64, initial: f64) -> PyResult<Self> {
        let inner = ExponentialSmoothing::new(alpha, sigma, initial)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get current smoothed value.
    #[getter]
    fn smoothed(&self) -> f64 {
        self.inner.smoothed()
    }

    /// Generate n samples.
    #[pyo3(signature = (n, seed=None))]
    fn sample(&mut self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<Py<PyList>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let samples = self.inner.sample_n(&mut rng, n);
        Ok(PyList::new(py, &samples)?.into())
    }
}

/// Register temporal functions with the Python module.
pub fn register_temporal(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyAR1>()?;
    m.add_class::<PyARp>()?;
    m.add_class::<PyMarkovChain>()?;
    m.add_class::<PyRandomWalk>()?;
    m.add_class::<PyExponentialSmoothing>()?;
    Ok(())
}
