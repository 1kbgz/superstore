//! Python bindings for Arrow memory format.
//!
//! This module exports Arrow data as IPC bytes, which can be read by PyArrow.

#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use superstore::arrow::{employees_arrow, superstore_arrow};

/// Convert an Arrow RecordBatch to IPC bytes.
fn batch_to_ipc_bytes(batch: &arrow::record_batch::RecordBatch) -> Result<Vec<u8>, String> {
    use arrow::ipc::writer::StreamWriter;

    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, batch.schema().as_ref())
            .map_err(|e| e.to_string())?;
        writer.write(batch).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    Ok(buffer)
}

/// Generate superstore data as Arrow IPC bytes.
///
/// The returned bytes can be read by PyArrow:
/// ```python
/// import pyarrow as pa
/// from superstore import superstoreArrowIpc
///
/// ipc_bytes = superstoreArrowIpc(100)
/// reader = pa.ipc.open_stream(ipc_bytes)
/// table = reader.read_all()
/// df = table.to_pandas()
/// ```
///
/// # Arguments
/// * `count` - Number of rows to generate
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// Arrow IPC stream bytes
#[pyfunction]
#[pyo3(signature = (count, seed=None))]
pub fn superstoreArrowIpc(
    py: Python<'_>,
    count: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyBytes>> {
    let batch = superstore_arrow(count, seed)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let ipc_bytes =
        batch_to_ipc_bytes(&batch).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    Ok(PyBytes::new(py, &ipc_bytes).into())
}

/// Generate employee data as Arrow IPC bytes.
///
/// The returned bytes can be read by PyArrow:
/// ```python
/// import pyarrow as pa
/// from superstore import employeesArrowIpc
///
/// ipc_bytes = employeesArrowIpc(100)
/// reader = pa.ipc.open_stream(ipc_bytes)
/// table = reader.read_all()
/// df = table.to_pandas()
/// ```
///
/// # Arguments
/// * `count` - Number of rows to generate
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// Arrow IPC stream bytes
#[pyfunction]
#[pyo3(signature = (count, seed=None))]
pub fn employeesArrowIpc(py: Python<'_>, count: usize, seed: Option<u64>) -> PyResult<Py<PyBytes>> {
    let batch = employees_arrow(count, seed)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let ipc_bytes =
        batch_to_ipc_bytes(&batch).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    Ok(PyBytes::new(py, &ipc_bytes).into())
}

/// Register Arrow functions with the Python module.
pub fn register_arrow(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(superstoreArrowIpc, m)?)?;
    m.add_function(wrap_pyfunction!(employeesArrowIpc, m)?)?;
    Ok(())
}
