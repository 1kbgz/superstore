//! Python bindings for parallel data generation.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use superstore::parallel::{employees_parallel, superstore_parallel};

fn superstore_row_to_pydict<'py>(
    py: Python<'py>,
    row: &superstore::SuperstoreRow,
) -> Bound<'py, PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("Row ID", row.row_id).unwrap();
    dict.set_item("Order ID", &row.order_id).unwrap();
    dict.set_item("Order Date", &row.order_date).unwrap();
    dict.set_item("Ship Date", &row.ship_date).unwrap();
    dict.set_item("Ship Mode", &row.ship_mode).unwrap();
    dict.set_item("Customer ID", &row.customer_id).unwrap();
    dict.set_item("Segment", &row.segment).unwrap();
    dict.set_item("Country", &row.country).unwrap();
    dict.set_item("City", &row.city).unwrap();
    dict.set_item("State", &row.state).unwrap();
    dict.set_item("Postal Code", &row.postal_code).unwrap();
    dict.set_item("Region", &row.region).unwrap();
    dict.set_item("Product ID", &row.product_id).unwrap();
    dict.set_item("Category", &row.category).unwrap();
    dict.set_item("Sub-Category", &row.sub_category).unwrap();
    dict.set_item("Sales", row.sales).unwrap();
    dict.set_item("Quantity", row.quantity).unwrap();
    dict.set_item("Discount", row.discount).unwrap();
    dict.set_item("Profit", row.profit).unwrap();
    dict
}

fn employee_row_to_pydict<'py>(
    py: Python<'py>,
    row: &superstore::EmployeeRow,
) -> Bound<'py, PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("Row ID", row.row_id).unwrap();
    dict.set_item("Employee ID", &row.employee_id).unwrap();
    dict.set_item("First Name", &row.first_name).unwrap();
    dict.set_item("Surname", &row.surname).unwrap();
    dict.set_item("Prefix", &row.prefix).unwrap();
    dict.set_item("Suffix", &row.suffix).unwrap();
    dict.set_item("Phone Number", &row.phone_number).unwrap();
    dict.set_item("Email", &row.email).unwrap();
    dict.set_item("SSN", &row.ssn).unwrap();
    dict.set_item("Street", &row.street).unwrap();
    dict.set_item("City", &row.city).unwrap();
    dict.set_item("Postal Code", &row.postal_code).unwrap();
    dict.set_item("Region", &row.region).unwrap();
    dict.set_item("State", &row.state).unwrap();
    dict.set_item("Country", &row.country).unwrap();
    dict.set_item("Start Date", row.start_date.format("%Y-%m-%d").to_string())
        .unwrap();
    dict.set_item(
        "Date of Birth",
        row.date_of_birth.format("%Y-%m-%d").to_string(),
    )
    .unwrap();
    dict
}

/// Generate superstore data in parallel using multiple CPU cores.
///
/// This function uses Rayon to parallelize data generation across all available
/// CPU cores, providing significant speedup for large datasets.
///
/// Args:
///     count: Number of rows to generate
///     output: Output format - "pandas", "polars", or "dict" (default: "pandas")
///     seed: Optional seed for reproducibility
///
/// Returns:
///     DataFrame or list of dicts depending on output format
///
/// Example:
///     >>> df = superstoreParallel(1_000_000)  # Uses all CPU cores
#[pyfunction]
#[pyo3(name = "superstoreParallel", signature = (count=1000, output="pandas", seed=None))]
pub fn py_superstore_parallel(
    py: Python<'_>,
    count: usize,
    output: &str,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let rows = superstore_parallel(count, seed);

    match output {
        "dict" => {
            let list = PyList::empty(py);
            for row in &rows {
                list.append(superstore_row_to_pydict(py, row))?;
            }
            Ok(list.into_any().into())
        }
        "polars" => {
            let polars = py.import("polars")?;
            let list = PyList::empty(py);
            for row in &rows {
                list.append(superstore_row_to_pydict(py, row))?;
            }
            let df = polars.call_method1("DataFrame", (list,))?;
            Ok(df.into())
        }
        _ => {
            // Default to pandas
            let pandas = py.import("pandas")?;
            let list = PyList::empty(py);
            for row in &rows {
                list.append(superstore_row_to_pydict(py, row))?;
            }
            let df = pandas.call_method1("DataFrame", (list,))?;
            Ok(df.into())
        }
    }
}

/// Generate employee data in parallel using multiple CPU cores.
///
/// This function uses Rayon to parallelize data generation across all available
/// CPU cores, providing significant speedup for large datasets.
///
/// Args:
///     count: Number of employees to generate
///     output: Output format - "pandas", "polars", or "dict" (default: "pandas")
///     seed: Optional seed for reproducibility
///
/// Returns:
///     DataFrame or list of dicts depending on output format
///
/// Example:
///     >>> df = employeesParallel(1_000_000)  # Uses all CPU cores
#[pyfunction]
#[pyo3(name = "employeesParallel", signature = (count=1000, output="pandas", seed=None))]
pub fn py_employees_parallel(
    py: Python<'_>,
    count: usize,
    output: &str,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let rows = employees_parallel(count, seed);

    match output {
        "dict" => {
            let list = PyList::empty(py);
            for row in &rows {
                list.append(employee_row_to_pydict(py, row))?;
            }
            Ok(list.into_any().into())
        }
        "polars" => {
            let polars = py.import("polars")?;
            let list = PyList::empty(py);
            for row in &rows {
                list.append(employee_row_to_pydict(py, row))?;
            }
            let df = polars.call_method1("DataFrame", (list,))?;
            Ok(df.into())
        }
        _ => {
            // Default to pandas
            let pandas = py.import("pandas")?;
            let list = PyList::empty(py);
            for row in &rows {
                list.append(employee_row_to_pydict(py, row))?;
            }
            let df = pandas.call_method1("DataFrame", (list,))?;
            Ok(df.into())
        }
    }
}

/// Get the number of CPU threads available for parallel operations.
///
/// Returns:
///     Number of threads Rayon will use
#[pyfunction]
#[pyo3(name = "numThreads")]
pub fn py_num_threads() -> usize {
    superstore::parallel::num_threads()
}

/// Set the number of threads for parallel operations.
///
/// This should be called early in the program before any parallel operations.
/// Once set, it cannot be changed.
///
/// Args:
///     num_threads: Number of threads to use for parallel generation
///
/// Raises:
///     RuntimeError: If the thread pool has already been initialized
#[pyfunction]
#[pyo3(name = "setNumThreads")]
pub fn py_set_num_threads(num_threads: usize) -> PyResult<()> {
    superstore::parallel::set_num_threads(num_threads)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Enable deterministic mode for reproducible results across platforms.
///
/// This sets a fixed number of threads (default: 1) to ensure that
/// parallel operations produce identical results regardless of the
/// number of CPU cores available. Combined with a seed, this guarantees
/// reproducible outputs.
///
/// Args:
///     num_threads: Number of threads to use (default: 1 for maximum determinism)
///
/// Raises:
///     RuntimeError: If the thread pool has already been initialized
///
/// Example:
///     >>> import superstore
///     >>> superstore.setDeterministicMode()  # Use single thread
///     >>> df1 = superstore.superstoreParallel(1000, seed=42)
///     >>> df2 = superstore.superstoreParallel(1000, seed=42)
///     >>> assert df1.equals(df2)  # Always identical
#[pyfunction]
#[pyo3(name = "setDeterministicMode", signature = (num_threads=1))]
pub fn py_set_deterministic_mode(num_threads: usize) -> PyResult<()> {
    superstore::parallel::set_num_threads(num_threads)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

pub fn register_parallel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_superstore_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(py_employees_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(py_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(py_set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(py_set_deterministic_mode, m)?)?;
    Ok(())
}
