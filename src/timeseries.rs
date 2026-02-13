use chrono::NaiveDateTime;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList};
use std::collections::HashMap;

use superstore::timeseries::{get_time_series, get_time_series_data, TimeSeriesData};

/// Create pandas DataFrame from TimeSeriesData struct
fn create_timeseries_pandas(py: Python<'_>, data: &TimeSeriesData) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;

    let columns_dict = PyDict::new(py);
    for col in &data.columns {
        let values = PyList::new(py, &col.values)?;
        columns_dict.set_item(col.name.to_string(), values)?;
    }

    let index_list: Vec<String> = data
        .index
        .iter()
        .map(|dt: &NaiveDateTime| dt.format("%Y-%m-%d %H:%M:%S").to_string())
        .collect();
    let index_py = PyList::new(py, &index_list)?;
    let datetime_index = pandas.call_method1("DatetimeIndex", (index_py,))?;

    let kwargs = [("index", datetime_index)].into_py_dict(py)?;
    let df = pandas.call_method("DataFrame", (columns_dict,), Some(&kwargs))?;

    Ok(df.into())
}

/// Create polars DataFrame from TimeSeriesData struct
fn create_timeseries_polars(py: Python<'_>, data: &TimeSeriesData) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let columns_dict = PyDict::new(py);

    // Add index as a column
    let index_list: Vec<String> = data
        .index
        .iter()
        .map(|dt: &NaiveDateTime| dt.format("%Y-%m-%d %H:%M:%S").to_string())
        .collect();
    columns_dict.set_item("index", PyList::new(py, &index_list)?)?;

    // Add data columns
    for col in &data.columns {
        columns_dict.set_item(col.name.to_string(), PyList::new(py, &col.values)?)?;
    }

    let df = polars.call_method1("DataFrame", (columns_dict,))?;
    Ok(df.into())
}

/// Create dict from TimeSeriesData struct
fn create_timeseries_dict(py: Python<'_>, data: &TimeSeriesData) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);

    let index_list: Vec<String> = data
        .index
        .iter()
        .map(|dt: &NaiveDateTime| dt.format("%Y-%m-%d %H:%M:%S").to_string())
        .collect();
    result.set_item("index", PyList::new(py, &index_list)?)?;

    for col in &data.columns {
        result.set_item(col.name.to_string(), PyList::new(py, &col.values)?)?;
    }

    Ok(result.into())
}

/// Create pandas dict of Series from HashMap data
fn create_hashmap_pandas(
    py: Python<'_>,
    data: &HashMap<char, (Vec<NaiveDateTime>, Vec<f64>)>,
) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let result_dict = PyDict::new(py);

    for (col_name, (dates, values)) in data.iter() {
        let index_list: Vec<String> = dates
            .iter()
            .map(|dt: &NaiveDateTime| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .collect();
        let index_py = PyList::new(py, &index_list)?;
        let datetime_index = pandas.call_method1("DatetimeIndex", (index_py,))?;

        let values_py = PyList::new(py, values)?;
        let kwargs = [("index", datetime_index)].into_py_dict(py)?;
        let series = pandas.call_method("Series", (values_py,), Some(&kwargs))?;

        result_dict.set_item(col_name.to_string(), series)?;
    }

    Ok(result_dict.into())
}

/// Create polars DataFrames dict from HashMap data
fn create_hashmap_polars(
    py: Python<'_>,
    data: &HashMap<char, (Vec<NaiveDateTime>, Vec<f64>)>,
) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let result_dict = PyDict::new(py);

    for (col_name, (dates, values)) in data.iter() {
        let index_list: Vec<String> = dates
            .iter()
            .map(|dt: &NaiveDateTime| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .collect();

        let df_dict = PyDict::new(py);
        df_dict.set_item("index", PyList::new(py, &index_list)?)?;
        df_dict.set_item("value", PyList::new(py, values)?)?;

        let df = polars.call_method1("DataFrame", (df_dict,))?;
        result_dict.set_item(col_name.to_string(), df)?;
    }

    Ok(result_dict.into())
}

/// Create dict from HashMap data
fn create_hashmap_dict(
    py: Python<'_>,
    data: &HashMap<char, (Vec<NaiveDateTime>, Vec<f64>)>,
) -> PyResult<Py<PyAny>> {
    let result_dict = PyDict::new(py);

    for (col_name, (dates, values)) in data.iter() {
        let index_list: Vec<String> = dates
            .iter()
            .map(|dt: &NaiveDateTime| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .collect();

        let col_dict = PyDict::new(py);
        col_dict.set_item("index", PyList::new(py, &index_list)?)?;
        col_dict.set_item("values", PyList::new(py, values)?)?;
        result_dict.set_item(col_name.to_string(), col_dict)?;
    }

    Ok(result_dict.into())
}

/// Parse TimeseriesConfig dict into (nper, freq, ncol, output, seed)
fn parse_timeseries_config(
    dict: &Bound<'_, PyDict>,
) -> PyResult<(usize, String, usize, String, Option<u64>)> {
    let nper: usize = dict
        .get_item("nper")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(30);

    let freq: String = dict
        .get_item("freq")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_else(|| "B".to_string());

    let ncol: usize = dict
        .get_item("ncol")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(4);

    let output: String = dict
        .get_item("output")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_else(|| "pandas".to_string());

    let seed: Option<u64> = dict.get_item("seed")?.and_then(|v| v.extract().ok());

    Ok((nper, freq, ncol, output, seed))
}

/// Generate time series data with structured configuration.
///
/// Args:
///     config: Optional TimeseriesConfig pydantic model, dict, or int (for backward compatibility).
///             If int, treated as nper. If None, uses default configuration.
///     nper: Number of periods (overrides config if provided)
///     freq: Frequency string (overrides config if provided)
///     ncol: Number of columns (overrides config if provided)
///     output: Output format ("pandas", "polars", or "dict")
///     seed: Random seed (overrides config if provided)
///
/// Returns:
///     Time series data in the specified format.
#[pyfunction]
#[pyo3(name = "timeseries", signature = (config=None, nper=None, freq=None, ncol=None, output=None, seed=None))]
pub fn py_get_time_series(
    py: Python<'_>,
    config: Option<&Bound<'_, PyAny>>,
    nper: Option<usize>,
    freq: Option<&str>,
    ncol: Option<usize>,
    output: Option<&str>,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    // Parse config from pydantic model, dict, or int (backward compat)
    let (cfg_nper, cfg_freq, cfg_ncol, cfg_output, cfg_seed) = if let Some(cfg) = config {
        // Check if it's an integer (backward compatibility: timeseries(30))
        if let Ok(int_val) = cfg.extract::<usize>() {
            (int_val, "B".to_string(), 4, "pandas".to_string(), None)
        // Check if it's a pydantic model (has model_dump method)
        } else if cfg.hasattr("model_dump")? {
            // Use mode="json" to ensure enums are serialized as strings
            let kwargs = PyDict::new(py);
            kwargs.set_item("mode", "json")?;
            let dict = cfg.call_method("model_dump", (), Some(&kwargs))?;
            let dict = dict.downcast::<PyDict>()?;
            parse_timeseries_config(dict)?
        } else if let Ok(dict) = cfg.downcast::<PyDict>() {
            parse_timeseries_config(dict)?
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "config must be a TimeseriesConfig, dict, int, or None",
            ));
        }
    } else {
        (30, "B".to_string(), 4, "pandas".to_string(), None)
    };

    // Override with explicit parameters if provided
    let final_nper = nper.unwrap_or(cfg_nper);
    let final_freq = freq.unwrap_or(&cfg_freq);
    let final_ncol = ncol.unwrap_or(cfg_ncol);
    let final_output = output.unwrap_or(&cfg_output);
    let final_seed = seed.or(cfg_seed);

    let data = get_time_series(final_nper, final_freq, final_ncol, final_seed);

    match final_output {
        "pandas" => create_timeseries_pandas(py, &data),
        "polars" => create_timeseries_polars(py, &data),
        "dict" => create_timeseries_dict(py, &data),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid output format '{}'. Must be 'pandas', 'polars', or 'dict'",
            output.unwrap_or("unknown")
        ))),
    }
}

#[pyfunction]
#[pyo3(name = "timeseriesData", signature = (nper=30, freq="B", ncol=4, output="pandas", seed=None))]
pub fn py_get_time_series_data(
    py: Python<'_>,
    nper: usize,
    freq: &str,
    ncol: usize,
    output: &str,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let data = get_time_series_data(nper, freq, ncol, seed);

    match output {
        "pandas" => create_hashmap_pandas(py, &data),
        "polars" => create_hashmap_polars(py, &data),
        "dict" => create_hashmap_dict(py, &data),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid output format '{}'. Must be 'pandas', 'polars', or 'dict'",
            output
        ))),
    }
}
