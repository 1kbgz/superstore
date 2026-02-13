//! Python bindings for streaming data generation.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use superstore::streaming::{employees_stream, superstore_stream};

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

/// Python iterator for streaming superstore data generation.
#[pyclass]
pub struct SuperstoreStream {
    iter: superstore::streaming::SuperstoreIterator,
}

#[pymethods]
impl SuperstoreStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> Option<Py<PyList>> {
        slf.iter.next().map(|chunk| {
            let list = PyList::empty(py);
            for row in &chunk {
                list.append(superstore_row_to_pydict(py, row)).unwrap();
            }
            list.into()
        })
    }
}

/// Python iterator for streaming employee data generation.
#[pyclass]
pub struct EmployeeStream {
    iter: superstore::streaming::EmployeeIterator,
}

#[pymethods]
impl EmployeeStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> Option<Py<PyList>> {
        slf.iter.next().map(|chunk| {
            let list = PyList::empty(py);
            for row in &chunk {
                list.append(employee_row_to_pydict(py, row)).unwrap();
            }
            list.into()
        })
    }
}

/// Create a streaming superstore data generator.
///
/// This returns an iterator that yields chunks of data, allowing memory-efficient
/// processing of large datasets.
///
/// Args:
///     total_count: Total number of rows to generate
///     chunk_size: Number of rows per chunk (default: 1000)
///     seed: Optional seed for reproducibility
///
/// Returns:
///     An iterator yielding lists of dicts
///
/// Example:
///     >>> for chunk in superstoreStream(1_000_000, chunk_size=10000):
///     ...     process(chunk)  # Each chunk is a list of 10000 dicts
#[pyfunction]
#[pyo3(name = "superstoreStream", signature = (total_count, chunk_size=1000, seed=None))]
pub fn py_superstore_stream(
    total_count: usize,
    chunk_size: usize,
    seed: Option<u64>,
) -> SuperstoreStream {
    SuperstoreStream {
        iter: superstore_stream(total_count, chunk_size, seed),
    }
}

/// Create a streaming employee data generator.
///
/// This returns an iterator that yields chunks of data, allowing memory-efficient
/// processing of large datasets.
///
/// Args:
///     total_count: Total number of employees to generate
///     chunk_size: Number of employees per chunk (default: 1000)
///     seed: Optional seed for reproducibility
///
/// Returns:
///     An iterator yielding lists of dicts
///
/// Example:
///     >>> for chunk in employeesStream(1_000_000, chunk_size=10000):
///     ...     process(chunk)  # Each chunk is a list of 10000 dicts
#[pyfunction]
#[pyo3(name = "employeesStream", signature = (total_count, chunk_size=1000, seed=None))]
pub fn py_employees_stream(
    total_count: usize,
    chunk_size: usize,
    seed: Option<u64>,
) -> EmployeeStream {
    EmployeeStream {
        iter: employees_stream(total_count, chunk_size, seed),
    }
}

pub fn register_streaming(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SuperstoreStream>()?;
    m.add_class::<EmployeeStream>()?;
    m.add_function(wrap_pyfunction!(py_superstore_stream, m)?)?;
    m.add_function(wrap_pyfunction!(py_employees_stream, m)?)?;
    Ok(())
}
