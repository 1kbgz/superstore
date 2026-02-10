use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

mod crossfilter;
mod general;
mod timeseries;

#[pymodule]
fn superstore(m: &Bound<PyModule>) -> PyResult<()> {
    // General module functions
    m.add_function(wrap_pyfunction!(general::py_superstore, m)?)?;
    m.add_function(wrap_pyfunction!(general::py_employees, m)?)?;

    // Timeseries module functions
    m.add_function(wrap_pyfunction!(timeseries::py_get_time_series, m)?)?;
    m.add_function(wrap_pyfunction!(timeseries::py_get_time_series_data, m)?)?;

    // Crossfilter module functions
    m.add_function(wrap_pyfunction!(crossfilter::py_machines, m)?)?;
    m.add_function(wrap_pyfunction!(crossfilter::py_usage, m)?)?;
    m.add_function(wrap_pyfunction!(crossfilter::py_status, m)?)?;
    m.add_function(wrap_pyfunction!(crossfilter::py_jobs, m)?)?;

    // Crossfilter helper functions for tests
    m.add_function(wrap_pyfunction!(crossfilter::py_id, m)?)?;
    m.add_function(wrap_pyfunction!(crossfilter::py_clip, m)?)?;
    m.add_function(wrap_pyfunction!(crossfilter::py_randrange, m)?)?;

    // Crossfilter schemas
    crossfilter::add_schemas(m)?;

    Ok(())
}
