use pyo3::prelude::*;

mod arrow;
mod copulas;
mod correlation;
mod crossfilter;
mod distributions;
mod export;
mod general;
mod parallel;
mod streaming;
mod temporal;
mod timeseries;
mod weather;

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

    // Streaming module
    streaming::register_streaming(m)?;

    // Parallel module
    parallel::register_parallel(m)?;

    // Distributions module
    distributions::register_distributions(m)?;

    // Arrow module
    arrow::register_arrow(m)?;

    // Export module (Parquet/CSV)
    export::register_export(m)?;

    // Correlation module
    correlation::register_correlation(m)?;

    // Temporal module
    temporal::register_temporal(m)?;

    // Copulas module
    copulas::register_copulas(m)?;

    // Weather module
    weather::register_weather(m)?;

    Ok(())
}
