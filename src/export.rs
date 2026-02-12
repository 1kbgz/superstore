//! Python bindings for direct file export (Parquet/CSV).

#![allow(non_snake_case)]

use pyo3::prelude::*;

use superstore::export::{
    employees_to_csv, employees_to_parquet, superstore_to_csv, superstore_to_parquet,
    ParquetCompression,
};

fn parse_compression(compression: Option<&str>) -> PyResult<ParquetCompression> {
    match compression {
        None | Some("snappy") => Ok(ParquetCompression::Snappy),
        Some("none") => Ok(ParquetCompression::None),
        Some("zstd") => Ok(ParquetCompression::Zstd),
        Some(other) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown compression: {}. Use 'none', 'snappy', or 'zstd'",
            other
        ))),
    }
}

/// Write superstore data directly to a Parquet file.
///
/// # Arguments
/// * `path` - Output file path
/// * `count` - Number of rows to generate
/// * `seed` - Optional random seed for reproducibility
/// * `compression` - Compression type: 'none', 'snappy' (default), or 'zstd'
///
/// # Returns
/// Number of rows written
#[pyfunction]
#[pyo3(signature = (path, count, seed=None, compression=None))]
pub fn superstoreToParquet(
    path: &str,
    count: usize,
    seed: Option<u64>,
    compression: Option<&str>,
) -> PyResult<usize> {
    let comp = parse_compression(compression)?;
    superstore_to_parquet(path, count, seed, comp)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Write employee data directly to a Parquet file.
///
/// # Arguments
/// * `path` - Output file path
/// * `count` - Number of rows to generate
/// * `seed` - Optional random seed for reproducibility
/// * `compression` - Compression type: 'none', 'snappy' (default), or 'zstd'
///
/// # Returns
/// Number of rows written
#[pyfunction]
#[pyo3(signature = (path, count, seed=None, compression=None))]
pub fn employeesToParquet(
    path: &str,
    count: usize,
    seed: Option<u64>,
    compression: Option<&str>,
) -> PyResult<usize> {
    let comp = parse_compression(compression)?;
    employees_to_parquet(path, count, seed, comp)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Write superstore data directly to a CSV file.
///
/// # Arguments
/// * `path` - Output file path
/// * `count` - Number of rows to generate
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// Number of rows written
#[pyfunction]
#[pyo3(signature = (path, count, seed=None))]
pub fn superstoreToCsv(path: &str, count: usize, seed: Option<u64>) -> PyResult<usize> {
    superstore_to_csv(path, count, seed)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Write employee data directly to a CSV file.
///
/// # Arguments
/// * `path` - Output file path
/// * `count` - Number of rows to generate
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// Number of rows written
#[pyfunction]
#[pyo3(signature = (path, count, seed=None))]
pub fn employeesToCsv(path: &str, count: usize, seed: Option<u64>) -> PyResult<usize> {
    employees_to_csv(path, count, seed)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Register export functions with the Python module.
pub fn register_export(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(superstoreToParquet, m)?)?;
    m.add_function(wrap_pyfunction!(employeesToParquet, m)?)?;
    m.add_function(wrap_pyfunction!(superstoreToCsv, m)?)?;
    m.add_function(wrap_pyfunction!(employeesToCsv, m)?)?;
    Ok(())
}
