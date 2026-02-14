use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use superstore::logs::{
    generate_app_logs, generate_logs, AppLogEntry, ErrorBurstConfig, LatencyConfig, LogEntry,
    LogFormat, LogsConfig,
};

/// Create pandas DataFrame from LogEntry rows
fn create_logs_pandas(py: Python<'_>, rows: &[LogEntry]) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let data = PyDict::new(py);

    let timestamps: Vec<&str> = rows.iter().map(|r| r.timestamp.as_str()).collect();
    let ip_addresses: Vec<&str> = rows.iter().map(|r| r.ip_address.as_str()).collect();
    let user_ids: Vec<Option<&str>> = rows.iter().map(|r| r.user_id.as_deref()).collect();
    let methods: Vec<&str> = rows.iter().map(|r| r.method.as_str()).collect();
    let paths: Vec<&str> = rows.iter().map(|r| r.path.as_str()).collect();
    let status_codes: Vec<u16> = rows.iter().map(|r| r.status_code).collect();
    let response_bytes: Vec<u32> = rows.iter().map(|r| r.response_bytes).collect();
    let latency_ms: Vec<f64> = rows.iter().map(|r| r.latency_ms).collect();
    let user_agents: Vec<Option<&str>> = rows.iter().map(|r| r.user_agent.as_deref()).collect();
    let referers: Vec<Option<&str>> = rows.iter().map(|r| r.referer.as_deref()).collect();

    data.set_item("timestamp", PyList::new(py, &timestamps)?)?;
    data.set_item("ip_address", PyList::new(py, &ip_addresses)?)?;
    data.set_item("user_id", PyList::new(py, &user_ids)?)?;
    data.set_item("method", PyList::new(py, &methods)?)?;
    data.set_item("path", PyList::new(py, &paths)?)?;
    data.set_item("status_code", PyList::new(py, &status_codes)?)?;
    data.set_item("response_bytes", PyList::new(py, &response_bytes)?)?;
    data.set_item("latency_ms", PyList::new(py, &latency_ms)?)?;
    data.set_item("user_agent", PyList::new(py, &user_agents)?)?;
    data.set_item("referer", PyList::new(py, &referers)?)?;

    let df = pandas.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create polars DataFrame from LogEntry rows
fn create_logs_polars(py: Python<'_>, rows: &[LogEntry]) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let data = PyDict::new(py);

    let timestamps: Vec<&str> = rows.iter().map(|r| r.timestamp.as_str()).collect();
    let ip_addresses: Vec<&str> = rows.iter().map(|r| r.ip_address.as_str()).collect();
    let user_ids: Vec<Option<&str>> = rows.iter().map(|r| r.user_id.as_deref()).collect();
    let methods: Vec<&str> = rows.iter().map(|r| r.method.as_str()).collect();
    let paths: Vec<&str> = rows.iter().map(|r| r.path.as_str()).collect();
    let status_codes: Vec<u16> = rows.iter().map(|r| r.status_code).collect();
    let response_bytes: Vec<u32> = rows.iter().map(|r| r.response_bytes).collect();
    let latency_ms: Vec<f64> = rows.iter().map(|r| r.latency_ms).collect();
    let user_agents: Vec<Option<&str>> = rows.iter().map(|r| r.user_agent.as_deref()).collect();
    let referers: Vec<Option<&str>> = rows.iter().map(|r| r.referer.as_deref()).collect();

    data.set_item("timestamp", PyList::new(py, &timestamps)?)?;
    data.set_item("ip_address", PyList::new(py, &ip_addresses)?)?;
    data.set_item("user_id", PyList::new(py, &user_ids)?)?;
    data.set_item("method", PyList::new(py, &methods)?)?;
    data.set_item("path", PyList::new(py, &paths)?)?;
    data.set_item("status_code", PyList::new(py, &status_codes)?)?;
    data.set_item("response_bytes", PyList::new(py, &response_bytes)?)?;
    data.set_item("latency_ms", PyList::new(py, &latency_ms)?)?;
    data.set_item("user_agent", PyList::new(py, &user_agents)?)?;
    data.set_item("referer", PyList::new(py, &referers)?)?;

    let df = polars.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create dict from LogEntry rows
fn create_logs_dict(py: Python<'_>, rows: &[LogEntry]) -> PyResult<Py<PyAny>> {
    let data = PyDict::new(py);

    let timestamps: Vec<&str> = rows.iter().map(|r| r.timestamp.as_str()).collect();
    let ip_addresses: Vec<&str> = rows.iter().map(|r| r.ip_address.as_str()).collect();
    let user_ids: Vec<Option<&str>> = rows.iter().map(|r| r.user_id.as_deref()).collect();
    let methods: Vec<&str> = rows.iter().map(|r| r.method.as_str()).collect();
    let paths: Vec<&str> = rows.iter().map(|r| r.path.as_str()).collect();
    let status_codes: Vec<u16> = rows.iter().map(|r| r.status_code).collect();
    let response_bytes: Vec<u32> = rows.iter().map(|r| r.response_bytes).collect();
    let latency_ms: Vec<f64> = rows.iter().map(|r| r.latency_ms).collect();
    let user_agents: Vec<Option<&str>> = rows.iter().map(|r| r.user_agent.as_deref()).collect();
    let referers: Vec<Option<&str>> = rows.iter().map(|r| r.referer.as_deref()).collect();

    data.set_item("timestamp", PyList::new(py, &timestamps)?)?;
    data.set_item("ip_address", PyList::new(py, &ip_addresses)?)?;
    data.set_item("user_id", PyList::new(py, &user_ids)?)?;
    data.set_item("method", PyList::new(py, &methods)?)?;
    data.set_item("path", PyList::new(py, &paths)?)?;
    data.set_item("status_code", PyList::new(py, &status_codes)?)?;
    data.set_item("response_bytes", PyList::new(py, &response_bytes)?)?;
    data.set_item("latency_ms", PyList::new(py, &latency_ms)?)?;
    data.set_item("user_agent", PyList::new(py, &user_agents)?)?;
    data.set_item("referer", PyList::new(py, &referers)?)?;

    Ok(data.into())
}

/// Create pandas DataFrame from AppLogEntry rows
fn create_app_logs_pandas(py: Python<'_>, rows: &[AppLogEntry]) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let data = PyDict::new(py);

    let timestamps: Vec<&str> = rows.iter().map(|r| r.timestamp.as_str()).collect();
    let levels: Vec<&str> = rows.iter().map(|r| r.level.as_str()).collect();
    let loggers: Vec<&str> = rows.iter().map(|r| r.logger.as_str()).collect();
    let messages: Vec<&str> = rows.iter().map(|r| r.message.as_str()).collect();
    let thread_ids: Vec<u32> = rows.iter().map(|r| r.thread_id).collect();
    let trace_ids: Vec<Option<&str>> = rows.iter().map(|r| r.trace_id.as_deref()).collect();
    let span_ids: Vec<Option<&str>> = rows.iter().map(|r| r.span_id.as_deref()).collect();
    let exceptions: Vec<Option<&str>> = rows.iter().map(|r| r.exception.as_deref()).collect();

    data.set_item("timestamp", PyList::new(py, &timestamps)?)?;
    data.set_item("level", PyList::new(py, &levels)?)?;
    data.set_item("logger", PyList::new(py, &loggers)?)?;
    data.set_item("message", PyList::new(py, &messages)?)?;
    data.set_item("thread_id", PyList::new(py, &thread_ids)?)?;
    data.set_item("trace_id", PyList::new(py, &trace_ids)?)?;
    data.set_item("span_id", PyList::new(py, &span_ids)?)?;
    data.set_item("exception", PyList::new(py, &exceptions)?)?;

    let df = pandas.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create polars DataFrame from AppLogEntry rows
fn create_app_logs_polars(py: Python<'_>, rows: &[AppLogEntry]) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let data = PyDict::new(py);

    let timestamps: Vec<&str> = rows.iter().map(|r| r.timestamp.as_str()).collect();
    let levels: Vec<&str> = rows.iter().map(|r| r.level.as_str()).collect();
    let loggers: Vec<&str> = rows.iter().map(|r| r.logger.as_str()).collect();
    let messages: Vec<&str> = rows.iter().map(|r| r.message.as_str()).collect();
    let thread_ids: Vec<u32> = rows.iter().map(|r| r.thread_id).collect();
    let trace_ids: Vec<Option<&str>> = rows.iter().map(|r| r.trace_id.as_deref()).collect();
    let span_ids: Vec<Option<&str>> = rows.iter().map(|r| r.span_id.as_deref()).collect();
    let exceptions: Vec<Option<&str>> = rows.iter().map(|r| r.exception.as_deref()).collect();

    data.set_item("timestamp", PyList::new(py, &timestamps)?)?;
    data.set_item("level", PyList::new(py, &levels)?)?;
    data.set_item("logger", PyList::new(py, &loggers)?)?;
    data.set_item("message", PyList::new(py, &messages)?)?;
    data.set_item("thread_id", PyList::new(py, &thread_ids)?)?;
    data.set_item("trace_id", PyList::new(py, &trace_ids)?)?;
    data.set_item("span_id", PyList::new(py, &span_ids)?)?;
    data.set_item("exception", PyList::new(py, &exceptions)?)?;

    let df = polars.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create dict from AppLogEntry rows
fn create_app_logs_dict(py: Python<'_>, rows: &[AppLogEntry]) -> PyResult<Py<PyAny>> {
    let data = PyDict::new(py);

    let timestamps: Vec<&str> = rows.iter().map(|r| r.timestamp.as_str()).collect();
    let levels: Vec<&str> = rows.iter().map(|r| r.level.as_str()).collect();
    let loggers: Vec<&str> = rows.iter().map(|r| r.logger.as_str()).collect();
    let messages: Vec<&str> = rows.iter().map(|r| r.message.as_str()).collect();
    let thread_ids: Vec<u32> = rows.iter().map(|r| r.thread_id).collect();
    let trace_ids: Vec<Option<&str>> = rows.iter().map(|r| r.trace_id.as_deref()).collect();
    let span_ids: Vec<Option<&str>> = rows.iter().map(|r| r.span_id.as_deref()).collect();
    let exceptions: Vec<Option<&str>> = rows.iter().map(|r| r.exception.as_deref()).collect();

    data.set_item("timestamp", PyList::new(py, &timestamps)?)?;
    data.set_item("level", PyList::new(py, &levels)?)?;
    data.set_item("logger", PyList::new(py, &loggers)?)?;
    data.set_item("message", PyList::new(py, &messages)?)?;
    data.set_item("thread_id", PyList::new(py, &thread_ids)?)?;
    data.set_item("trace_id", PyList::new(py, &trace_ids)?)?;
    data.set_item("span_id", PyList::new(py, &span_ids)?)?;
    data.set_item("exception", PyList::new(py, &exceptions)?)?;

    Ok(data.into())
}

/// Parse LogsConfig from Python dict
fn parse_logs_config(dict: &Bound<'_, PyDict>) -> PyResult<(LogsConfig, String)> {
    let count: usize = dict
        .get_item("count")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(1000);

    let output: String = dict
        .get_item("output")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_else(|| "pandas".to_string());

    let seed: Option<u64> = dict.get_item("seed")?.and_then(|v| v.extract().ok());

    let format_str: String = dict
        .get_item("format")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_else(|| "combined".to_string());

    let format = match format_str.to_lowercase().as_str() {
        "common" => LogFormat::Common,
        "json" => LogFormat::Json,
        "application" => LogFormat::Application,
        _ => LogFormat::Combined,
    };

    let start_time: Option<String> = dict.get_item("start_time")?.and_then(|v| v.extract().ok());

    let requests_per_second: f64 = dict
        .get_item("requests_per_second")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(100.0);

    let success_rate: f64 = dict
        .get_item("success_rate")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.95);

    let include_user_agent: bool = dict
        .get_item("include_user_agent")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(true);

    let include_referer: bool = dict
        .get_item("include_referer")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(true);

    let unique_ips: usize = dict
        .get_item("unique_ips")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(1000);

    let unique_users: usize = dict
        .get_item("unique_users")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(500);

    let api_path_ratio: f64 = dict
        .get_item("api_path_ratio")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.7);

    // Parse nested ErrorBurstConfig
    let error_burst = if let Some(eb_val) = dict.get_item("error_burst")? {
        if let Ok(eb_dict) = eb_val.downcast::<PyDict>() {
            let enable: bool = eb_dict
                .get_item("enable")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(true);
            let burst_probability: f64 = eb_dict
                .get_item("burst_probability")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.02);
            let burst_duration_seconds: u32 = eb_dict
                .get_item("burst_duration_seconds")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(30);
            let burst_error_rate: f64 = eb_dict
                .get_item("burst_error_rate")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.5);
            ErrorBurstConfig {
                enable,
                burst_probability,
                burst_duration_seconds,
                burst_error_rate,
            }
        } else {
            ErrorBurstConfig::default()
        }
    } else {
        ErrorBurstConfig::default()
    };

    // Parse nested LatencyConfig
    let latency = if let Some(lat_val) = dict.get_item("latency")? {
        if let Ok(lat_dict) = lat_val.downcast::<PyDict>() {
            let base_latency_ms: f64 = lat_dict
                .get_item("base_latency_ms")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(50.0);
            let latency_stddev: f64 = lat_dict
                .get_item("latency_stddev")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.8);
            let slow_request_probability: f64 = lat_dict
                .get_item("slow_request_probability")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.05);
            let slow_request_multiplier: f64 = lat_dict
                .get_item("slow_request_multiplier")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(10.0);
            LatencyConfig {
                base_latency_ms,
                latency_stddev,
                slow_request_probability,
                slow_request_multiplier,
            }
        } else {
            LatencyConfig::default()
        }
    } else {
        LatencyConfig::default()
    };

    let config = LogsConfig {
        count,
        seed,
        format,
        start_time,
        requests_per_second,
        success_rate,
        error_burst,
        latency,
        include_user_agent,
        include_referer,
        unique_ips,
        unique_users,
        api_path_ratio,
    };

    Ok((config, output))
}

/// Generate web server access logs.
///
/// Returns realistic HTTP access log entries with configurable traffic patterns,
/// status code distributions (via Markov chain), latency (LogNormal), and error bursts.
///
/// # Arguments
/// * `config` - Optional LogsConfig or dict with generation parameters
///
/// # Returns
/// * DataFrame (pandas/polars) or dict of log entries
#[pyfunction]
#[pyo3(signature = (config=None))]
pub fn logs(py: Python<'_>, config: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    let (logs_config, output_format) = if let Some(dict) = config {
        parse_logs_config(dict)?
    } else {
        (LogsConfig::default(), "pandas".to_string())
    };

    let rows = generate_logs(&logs_config);

    match output_format.to_lowercase().as_str() {
        "polars" => create_logs_polars(py, &rows),
        "dict" => create_logs_dict(py, &rows),
        _ => create_logs_pandas(py, &rows),
    }
}

/// Generate application event logs.
///
/// Returns application-level log entries with log levels, loggers, messages,
/// thread IDs, trace/span IDs, and optional exceptions.
///
/// # Arguments
/// * `config` - Optional LogsConfig or dict with generation parameters
///
/// # Returns
/// * DataFrame (pandas/polars) or dict of application log entries
#[pyfunction]
#[pyo3(signature = (config=None))]
pub fn app_logs(py: Python<'_>, config: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    let (logs_config, output_format) = if let Some(dict) = config {
        parse_logs_config(dict)?
    } else {
        (LogsConfig::default(), "pandas".to_string())
    };

    let rows = generate_app_logs(&logs_config);

    match output_format.to_lowercase().as_str() {
        "polars" => create_app_logs_polars(py, &rows),
        "dict" => create_app_logs_dict(py, &rows),
        _ => create_app_logs_pandas(py, &rows),
    }
}
