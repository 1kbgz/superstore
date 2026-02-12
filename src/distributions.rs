//! Python bindings for statistical distributions.

use pyo3::prelude::*;
use pyo3::types::PyList;

use rand::rngs::StdRng;
use rand::SeedableRng;

use superstore::distributions::{DistributionType, NoiseModel};

/// Sample from a uniform distribution.
///
/// Args:
///     min: Minimum value
///     max: Maximum value
///     n: Number of samples (default: 1)
///     seed: Optional seed for reproducibility
///
/// Returns:
///     Single value if n=1, list of values otherwise
#[pyfunction]
#[pyo3(name = "sampleUniform", signature = (min, max, n=1, seed=None))]
pub fn py_sample_uniform(
    py: Python<'_>,
    min: f64,
    max: f64,
    n: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let dist = DistributionType::uniform(min, max);

    if n == 1 {
        Ok(dist.sample(&mut rng).into_pyobject(py)?.into_any().unbind())
    } else {
        let samples: Vec<f64> = dist.sample_n(&mut rng, n);
        Ok(samples.into_pyobject(py)?.into_any().unbind())
    }
}

/// Sample from a normal (Gaussian) distribution.
///
/// Args:
///     mean: Mean of the distribution
///     std_dev: Standard deviation
///     n: Number of samples (default: 1)
///     seed: Optional seed for reproducibility
///
/// Returns:
///     Single value if n=1, list of values otherwise
#[pyfunction]
#[pyo3(name = "sampleNormal", signature = (mean, std_dev, n=1, seed=None))]
pub fn py_sample_normal(
    py: Python<'_>,
    mean: f64,
    std_dev: f64,
    n: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let dist = DistributionType::normal(mean, std_dev);

    if n == 1 {
        Ok(dist.sample(&mut rng).into_pyobject(py)?.into_any().unbind())
    } else {
        let samples: Vec<f64> = dist.sample_n(&mut rng, n);
        Ok(samples.into_pyobject(py)?.into_any().unbind())
    }
}

/// Sample from a log-normal distribution.
///
/// Args:
///     mu: Mean of the underlying normal distribution
///     sigma: Standard deviation of the underlying normal distribution
///     n: Number of samples (default: 1)
///     seed: Optional seed for reproducibility
///
/// Returns:
///     Single value if n=1, list of values otherwise
#[pyfunction]
#[pyo3(name = "sampleLogNormal", signature = (mu, sigma, n=1, seed=None))]
pub fn py_sample_log_normal(
    py: Python<'_>,
    mu: f64,
    sigma: f64,
    n: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let dist = DistributionType::log_normal(mu, sigma);

    if n == 1 {
        Ok(dist.sample(&mut rng).into_pyobject(py)?.into_any().unbind())
    } else {
        let samples: Vec<f64> = dist.sample_n(&mut rng, n);
        Ok(samples.into_pyobject(py)?.into_any().unbind())
    }
}

/// Sample from an exponential distribution.
///
/// Args:
///     lambda_: Rate parameter (1/mean)
///     n: Number of samples (default: 1)
///     seed: Optional seed for reproducibility
///
/// Returns:
///     Single value if n=1, list of values otherwise
#[pyfunction]
#[pyo3(name = "sampleExponential", signature = (lambda_, n=1, seed=None))]
pub fn py_sample_exponential(
    py: Python<'_>,
    lambda_: f64,
    n: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let dist = DistributionType::exponential(lambda_);

    if n == 1 {
        Ok(dist.sample(&mut rng).into_pyobject(py)?.into_any().unbind())
    } else {
        let samples: Vec<f64> = dist.sample_n(&mut rng, n);
        Ok(samples.into_pyobject(py)?.into_any().unbind())
    }
}

/// Sample from a Poisson distribution.
///
/// Args:
///     lambda_: Rate parameter (expected count)
///     n: Number of samples (default: 1)
///     seed: Optional seed for reproducibility
///
/// Returns:
///     Single value if n=1, list of values otherwise
#[pyfunction]
#[pyo3(name = "samplePoisson", signature = (lambda_, n=1, seed=None))]
pub fn py_sample_poisson(
    py: Python<'_>,
    lambda_: f64,
    n: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let dist = DistributionType::poisson(lambda_);

    if n == 1 {
        Ok((dist.sample(&mut rng) as i64)
            .into_pyobject(py)?
            .into_any()
            .unbind())
    } else {
        let samples: Vec<i64> = dist
            .sample_n(&mut rng, n)
            .iter()
            .map(|&x| x as i64)
            .collect();
        Ok(samples.into_pyobject(py)?.into_any().unbind())
    }
}

/// Sample from a Pareto (power law) distribution.
///
/// Args:
///     scale: Scale parameter (minimum value)
///     shape: Shape parameter (tail index)
///     n: Number of samples (default: 1)
///     seed: Optional seed for reproducibility
///
/// Returns:
///     Single value if n=1, list of values otherwise
#[pyfunction]
#[pyo3(name = "samplePareto", signature = (scale, shape, n=1, seed=None))]
pub fn py_sample_pareto(
    py: Python<'_>,
    scale: f64,
    shape: f64,
    n: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let dist = DistributionType::pareto(scale, shape);

    if n == 1 {
        Ok(dist.sample(&mut rng).into_pyobject(py)?.into_any().unbind())
    } else {
        let samples: Vec<f64> = dist.sample_n(&mut rng, n);
        Ok(samples.into_pyobject(py)?.into_any().unbind())
    }
}

/// Sample from a Beta distribution.
///
/// Args:
///     alpha: Shape parameter alpha
///     beta: Shape parameter beta
///     n: Number of samples (default: 1)
///     seed: Optional seed for reproducibility
///
/// Returns:
///     Single value if n=1, list of values otherwise (values in [0, 1])
#[pyfunction]
#[pyo3(name = "sampleBeta", signature = (alpha, beta, n=1, seed=None))]
pub fn py_sample_beta(
    py: Python<'_>,
    alpha: f64,
    beta: f64,
    n: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let dist = DistributionType::beta(alpha, beta);

    if n == 1 {
        Ok(dist.sample(&mut rng).into_pyobject(py)?.into_any().unbind())
    } else {
        let samples: Vec<f64> = dist.sample_n(&mut rng, n);
        Ok(samples.into_pyobject(py)?.into_any().unbind())
    }
}

/// Sample from a Gamma distribution.
///
/// Args:
///     shape: Shape parameter
///     scale: Scale parameter
///     n: Number of samples (default: 1)
///     seed: Optional seed for reproducibility
///
/// Returns:
///     Single value if n=1, list of values otherwise
#[pyfunction]
#[pyo3(name = "sampleGamma", signature = (shape, scale, n=1, seed=None))]
pub fn py_sample_gamma(
    py: Python<'_>,
    shape: f64,
    scale: f64,
    n: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let dist = DistributionType::gamma(shape, scale);

    if n == 1 {
        Ok(dist.sample(&mut rng).into_pyobject(py)?.into_any().unbind())
    } else {
        let samples: Vec<f64> = dist.sample_n(&mut rng, n);
        Ok(samples.into_pyobject(py)?.into_any().unbind())
    }
}

/// Sample from a Weibull distribution.
///
/// Args:
///     shape: Shape parameter
///     scale: Scale parameter
///     n: Number of samples (default: 1)
///     seed: Optional seed for reproducibility
///
/// Returns:
///     Single value if n=1, list of values otherwise
#[pyfunction]
#[pyo3(name = "sampleWeibull", signature = (shape, scale, n=1, seed=None))]
pub fn py_sample_weibull(
    py: Python<'_>,
    shape: f64,
    scale: f64,
    n: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let dist = DistributionType::weibull(shape, scale);

    if n == 1 {
        Ok(dist.sample(&mut rng).into_pyobject(py)?.into_any().unbind())
    } else {
        let samples: Vec<f64> = dist.sample_n(&mut rng, n);
        Ok(samples.into_pyobject(py)?.into_any().unbind())
    }
}

/// Sample from a categorical distribution with weights.
///
/// Args:
///     weights: List of weights for each category (will be normalized)
///     n: Number of samples (default: 1)
///     seed: Optional seed for reproducibility
///
/// Returns:
///     Single category index if n=1, list of indices otherwise
#[pyfunction]
#[pyo3(name = "sampleCategorical", signature = (weights, n=1, seed=None))]
pub fn py_sample_categorical(
    py: Python<'_>,
    weights: Vec<f64>,
    n: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let dist = DistributionType::categorical(weights);

    if n == 1 {
        Ok((dist.sample(&mut rng) as i64)
            .into_pyobject(py)?
            .into_any()
            .unbind())
    } else {
        let samples: Vec<i64> = dist
            .sample_n(&mut rng, n)
            .iter()
            .map(|&x| x as i64)
            .collect();
        Ok(samples.into_pyobject(py)?.into_any().unbind())
    }
}

/// Sample from a mixture of normal distributions.
///
/// Args:
///     means: List of means for each component
///     std_devs: List of standard deviations for each component
///     weights: List of weights for each component (will be normalized)
///     n: Number of samples (default: 1)
///     seed: Optional seed for reproducibility
///
/// Returns:
///     Single value if n=1, list of values otherwise
///
/// Example:
///     >>> # Bimodal distribution
///     >>> samples = sampleMixture([30000, 80000], [10000, 20000], [0.6, 0.4], n=1000)
#[pyfunction]
#[pyo3(name = "sampleMixture", signature = (means, std_devs, weights, n=1, seed=None))]
pub fn py_sample_mixture(
    py: Python<'_>,
    means: Vec<f64>,
    std_devs: Vec<f64>,
    weights: Vec<f64>,
    n: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let distributions: Vec<DistributionType> = means
        .iter()
        .zip(std_devs.iter())
        .map(|(&m, &s)| DistributionType::normal(m, s))
        .collect();

    let dist = DistributionType::mixture(distributions, weights);

    if n == 1 {
        Ok(dist.sample(&mut rng).into_pyobject(py)?.into_any().unbind())
    } else {
        let samples: Vec<f64> = dist.sample_n(&mut rng, n);
        Ok(samples.into_pyobject(py)?.into_any().unbind())
    }
}

/// Add Gaussian noise to values.
///
/// Args:
///     values: List of values to add noise to
///     std_dev: Standard deviation of the noise
///     seed: Optional seed for reproducibility
///
/// Returns:
///     List of values with noise added
#[pyfunction]
#[pyo3(name = "addGaussianNoise", signature = (values, std_dev, seed=None))]
pub fn py_add_gaussian_noise(
    py: Python<'_>,
    values: Vec<f64>,
    std_dev: f64,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let noise = NoiseModel::Gaussian { std_dev };

    let noisy: Vec<f64> = values
        .iter()
        .filter_map(|&v| noise.apply(&mut rng, v))
        .collect();

    Ok(noisy.into_pyobject(py)?.into_any().unbind())
}

/// Apply missing at random to values.
///
/// Args:
///     values: List of values
///     probability: Probability of each value being missing (0-1)
///     seed: Optional seed for reproducibility
///
/// Returns:
///     List of values with some replaced by None
#[pyfunction]
#[pyo3(name = "applyMissing", signature = (values, probability, seed=None))]
pub fn py_apply_missing(
    py: Python<'_>,
    values: Vec<f64>,
    probability: f64,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let noise = NoiseModel::MissingAtRandom { probability };

    let list = PyList::empty(py);
    for v in values {
        match noise.apply(&mut rng, v) {
            Some(val) => list.append(val)?,
            None => list.append(py.None())?,
        }
    }

    Ok(list.into_any().unbind())
}

pub fn register_distributions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_sample_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_normal, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_log_normal, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_exponential, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_poisson, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_pareto, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_beta, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_weibull, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_categorical, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_mixture, m)?)?;
    m.add_function(wrap_pyfunction!(py_add_gaussian_noise, m)?)?;
    m.add_function(wrap_pyfunction!(py_apply_missing, m)?)?;
    Ok(())
}
