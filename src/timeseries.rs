use chrono::NaiveDateTime;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList};
use std::collections::HashMap;

use superstore::timeseries::{
    get_time_series_data, get_time_series_with_config, GarchConfig, IntradayConfig, JumpConfig,
    MeanReversionConfig, RegimeConfig, TimeSeriesData, TimeseriesConfig,
};

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

/// Parse full TimeseriesConfig dict into Rust struct
fn parse_full_timeseries_config(dict: &Bound<'_, PyDict>) -> PyResult<(TimeseriesConfig, String)> {
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

    let ar_phi: f64 = dict
        .get_item("ar_phi")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.95);

    let sigma: f64 = dict
        .get_item("sigma")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(1.0);

    let drift: f64 = dict
        .get_item("drift")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.0);

    let cumulative: bool = dict
        .get_item("cumulative")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(true);

    let use_fat_tails: bool = dict
        .get_item("use_fat_tails")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(false);

    let degrees_freedom: f64 = dict
        .get_item("degrees_freedom")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(5.0);

    let cross_correlation: f64 = dict
        .get_item("cross_correlation")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.0);

    // Parse nested RegimeConfig
    let regimes = if let Some(regimes_val) = dict.get_item("regimes")? {
        if let Ok(regimes_dict) = regimes_val.downcast::<PyDict>() {
            let enable: bool = regimes_dict
                .get_item("enable")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(false);
            let n_regimes: usize = regimes_dict
                .get_item("n_regimes")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(2);
            let regime_persistence: f64 = regimes_dict
                .get_item("regime_persistence")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.95);
            let volatility_multipliers: Vec<f64> = regimes_dict
                .get_item("volatility_multipliers")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_else(|| vec![1.0, 2.5]);
            RegimeConfig {
                enable,
                n_regimes,
                regime_persistence,
                volatility_multipliers,
            }
        } else {
            RegimeConfig::default()
        }
    } else {
        RegimeConfig::default()
    };

    // Parse nested JumpConfig
    let jumps = if let Some(jumps_val) = dict.get_item("jumps")? {
        if let Ok(jumps_dict) = jumps_val.downcast::<PyDict>() {
            let enable: bool = jumps_dict
                .get_item("enable")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(false);
            let jump_probability: f64 = jumps_dict
                .get_item("jump_probability")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.01);
            let jump_mean: f64 = jumps_dict
                .get_item("jump_mean")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.0);
            let jump_stddev: f64 = jumps_dict
                .get_item("jump_stddev")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.05);
            JumpConfig {
                enable,
                jump_probability,
                jump_mean,
                jump_stddev,
            }
        } else {
            JumpConfig::default()
        }
    } else {
        JumpConfig::default()
    };

    let config = TimeseriesConfig {
        nper,
        ncol,
        freq,
        seed,
        ar_phi,
        sigma,
        drift,
        cumulative,
        use_fat_tails,
        degrees_freedom,
        cross_correlation,
        regimes,
        jumps,
        // Priority 5 fields - use defaults
        garch: GarchConfig::default(),
        mean_reversion: MeanReversionConfig::default(),
        intraday: IntradayConfig::default(),
        event_windows: superstore::timeseries::EventWindowConfig::default(),
        compute_metrics: false,
    };

    Ok((config, output))
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
    let (mut ts_config, cfg_output) = if let Some(cfg) = config {
        // Check if it's an integer (backward compatibility: timeseries(30))
        if let Ok(int_val) = cfg.extract::<usize>() {
            (
                TimeseriesConfig {
                    nper: int_val,
                    ..Default::default()
                },
                "pandas".to_string(),
            )
        // Check if it's a pydantic model (has model_dump method)
        } else if cfg.hasattr("model_dump")? {
            // Use mode="json" to ensure enums are serialized as strings
            let kwargs = PyDict::new(py);
            kwargs.set_item("mode", "json")?;
            let dict = cfg.call_method("model_dump", (), Some(&kwargs))?;
            let dict = dict.downcast::<PyDict>()?;
            parse_full_timeseries_config(dict)?
        } else if let Ok(dict) = cfg.downcast::<PyDict>() {
            parse_full_timeseries_config(dict)?
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "config must be a TimeseriesConfig, dict, int, or None",
            ));
        }
    } else {
        (TimeseriesConfig::default(), "pandas".to_string())
    };

    // Override with explicit parameters if provided
    if let Some(n) = nper {
        ts_config.nper = n;
    }
    if let Some(f) = freq {
        ts_config.freq = f.to_string();
    }
    if let Some(n) = ncol {
        ts_config.ncol = n;
    }
    if let Some(s) = seed {
        ts_config.seed = Some(s);
    }

    let final_output = output.unwrap_or(&cfg_output);

    // Use enhanced config-based generation
    let data_with_metrics = get_time_series_with_config(&ts_config);
    // Convert to basic TimeSeriesData for output functions
    let data: TimeSeriesData = data_with_metrics.into();

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
