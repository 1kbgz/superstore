use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use superstore::crossfilter::{
    generate_telemetry, job as rust_job, machines as rust_machines, status as rust_status,
    usage as rust_usage, AnomalyConfig, DegradationConfig, FailureCascadeConfig, Job, Machine,
    MaintenanceConfig, SensorDriftConfig, Status, TelemetryConfig, TelemetryReading,
    TemporalConfig, Usage,
};

fn machine_to_pydict(py: Python<'_>, m: &Machine) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("machine_id", &m.machine_id)?;
    dict.set_item("kind", &m.kind)?;
    dict.set_item("cores", m.cores)?;
    dict.set_item("region", &m.region)?;
    dict.set_item("zone", &m.zone)?;
    Ok(dict.into())
}

fn usage_to_pydict(py: Python<'_>, u: &Usage) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("machine_id", &u.machine_id)?;
    dict.set_item("kind", &u.kind)?;
    dict.set_item("cores", u.cores)?;
    dict.set_item("region", &u.region)?;
    dict.set_item("zone", &u.zone)?;
    dict.set_item("cpu", u.cpu)?;
    dict.set_item("mem", u.mem)?;
    dict.set_item("free", u.free)?;
    dict.set_item("network", u.network)?;
    dict.set_item("disk", u.disk)?;
    Ok(dict.into())
}

fn status_to_pydict(py: Python<'_>, s: &Status, json: bool) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("machine_id", &s.machine_id)?;
    dict.set_item("kind", &s.kind)?;
    dict.set_item("cores", s.cores)?;
    dict.set_item("region", &s.region)?;
    dict.set_item("zone", &s.zone)?;
    dict.set_item("cpu", s.cpu)?;
    dict.set_item("mem", s.mem)?;
    dict.set_item("free", s.free)?;
    dict.set_item("network", s.network)?;
    dict.set_item("disk", s.disk)?;
    dict.set_item("status", &s.status)?;

    if json {
        dict.set_item("last_update", &s.last_update)?;
    } else {
        // Parse the ISO datetime string and convert to Python datetime
        let datetime_module = py.import("datetime")?;
        let datetime_class = datetime_module.getattr("datetime")?;
        let dt = datetime_class.call_method1(
            "fromisoformat",
            (&s.last_update
                .replace("T", " ")
                .split('.')
                .next()
                .unwrap_or(&s.last_update),),
        )?;
        dict.set_item("last_update", dt)?;
    }

    Ok(dict.into())
}

fn job_to_pydict(py: Python<'_>, j: &Job, json: bool) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("machine_id", &j.machine_id)?;
    dict.set_item("job_id", &j.job_id)?;
    dict.set_item("name", &j.name)?;
    dict.set_item("units", j.units)?;

    if json {
        dict.set_item("start_time", &j.start_time)?;
        dict.set_item("end_time", &j.end_time)?;
    } else {
        let datetime_module = py.import("datetime")?;
        let datetime_class = datetime_module.getattr("datetime")?;
        let start = datetime_class.call_method1(
            "fromisoformat",
            (&j.start_time
                .replace("T", " ")
                .split('.')
                .next()
                .unwrap_or(&j.start_time),),
        )?;
        let end = datetime_class.call_method1(
            "fromisoformat",
            (&j.end_time
                .replace("T", " ")
                .split('.')
                .next()
                .unwrap_or(&j.end_time),),
        )?;
        dict.set_item("start_time", start)?;
        dict.set_item("end_time", end)?;
    }

    Ok(dict.into())
}

/// Parse CrossfilterConfig dict into (count, seed)
fn parse_crossfilter_config(dict: &Bound<'_, PyDict>) -> PyResult<(usize, Option<u64>)> {
    // CrossfilterConfig uses n_machines for count
    let count: usize = dict
        .get_item("n_machines")?
        .map(|v| v.extract())
        .transpose()?
        .or_else(|| {
            dict.get_item("count")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
        })
        .unwrap_or(100);

    let seed: Option<u64> = dict.get_item("seed")?.and_then(|v| v.extract().ok());

    Ok((count, seed))
}

/// Generate machine data with structured configuration.
///
/// Args:
///     config: Optional CrossfilterConfig pydantic model, dict, or int (for backward compatibility).
///             If int, treated as count. If None, uses default configuration.
///     count: Number of machines (overrides config if provided)
///     json: Whether to return JSON (deprecated, unused)
///     seed: Random seed (overrides config if provided)
///
/// Returns:
///     List of machine dictionaries.
#[pyfunction]
#[pyo3(name = "machines", signature = (config=None, count=None, json=false, seed=None))]
pub fn py_machines(
    py: Python<'_>,
    config: Option<&Bound<'_, PyAny>>,
    count: Option<usize>,
    json: bool,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let _ = json;

    // Parse config from pydantic model, dict, or int (backward compat)
    let (cfg_count, cfg_seed) = if let Some(cfg) = config {
        // Check if it's an integer (backward compatibility: machines(100))
        if let Ok(int_val) = cfg.extract::<usize>() {
            (int_val, None)
        // Check if it's a pydantic model (has model_dump method)
        } else if cfg.hasattr("model_dump")? {
            // Use mode="json" to ensure enums are serialized as strings
            let kwargs = PyDict::new(py);
            kwargs.set_item("mode", "json")?;
            let dict = cfg.call_method("model_dump", (), Some(&kwargs))?;
            let dict = dict.downcast::<PyDict>()?;
            parse_crossfilter_config(dict)?
        } else if let Ok(dict) = cfg.downcast::<PyDict>() {
            parse_crossfilter_config(dict)?
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "config must be a CrossfilterConfig, dict, int, or None",
            ));
        }
    } else {
        (100, None)
    };

    // Override with explicit parameters if provided
    let final_count = count.unwrap_or(cfg_count);
    let final_seed = seed.or(cfg_seed);

    let machines = rust_machines(final_count, final_seed);
    let list = PyList::empty(py);
    for m in &machines {
        list.append(machine_to_pydict(py, m)?)?;
    }

    Ok(list.into())
}

#[pyfunction]
#[pyo3(name = "usage", signature = (machine, json=false, seed=None))]
pub fn py_usage(
    py: Python<'_>,
    machine: &Bound<'_, PyDict>,
    json: bool,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let _json = json;

    // Extract machine fields
    let machine_id: String = machine
        .get_item("machine_id")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();
    let kind: String = machine
        .get_item("kind")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();
    let cores: i32 = machine
        .get_item("cores")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0);
    let region: String = machine
        .get_item("region")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();
    let zone: String = machine
        .get_item("zone")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();

    // Check if we have previous usage data (cpu field present)
    let has_cpu = machine.get_item("cpu")?.is_some();
    let cpu_val: Option<f64> = machine.get_item("cpu")?.map(|v| v.extract()).transpose()?;

    let m = Machine {
        machine_id: machine_id.clone(),
        kind: kind.clone(),
        cores,
        region: region.clone(),
        zone: zone.clone(),
    };

    if !has_cpu || cpu_val.is_none() {
        // No previous usage, return initial usage
        let u = rust_usage(&m, None, seed);
        return Ok(usage_to_pydict(py, &u)?.into_bound(py).into());
    }

    // Build previous usage from the input dict
    let prev_usage = Usage {
        machine_id,
        kind,
        cores,
        region,
        zone,
        cpu: machine
            .get_item("cpu")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0.0),
        mem: machine
            .get_item("mem")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0.0),
        free: machine
            .get_item("free")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(100.0),
        network: machine
            .get_item("network")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0.0),
        disk: machine
            .get_item("disk")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0.0),
    };

    let u = rust_usage(&m, Some(&prev_usage), seed);
    Ok(usage_to_pydict(py, &u)?.into_bound(py).into())
}

#[pyfunction]
#[pyo3(name = "status", signature = (machine, json=false))]
pub fn py_status(py: Python<'_>, machine: &Bound<'_, PyDict>, json: bool) -> PyResult<Py<PyAny>> {
    let json_flag = json;

    // Extract fields
    let machine_id: String = machine
        .get_item("machine_id")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();
    let kind: String = machine
        .get_item("kind")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();
    let cores: i32 = machine
        .get_item("cores")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0);
    let region: String = machine
        .get_item("region")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();
    let zone: String = machine
        .get_item("zone")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();
    let cpu: f64 = machine
        .get_item("cpu")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.0);
    let mem: f64 = machine
        .get_item("mem")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.0);
    let free: f64 = machine
        .get_item("free")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(100.0);
    let network: f64 = machine
        .get_item("network")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.0);
    let disk: f64 = machine
        .get_item("disk")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.0);

    // Check if cpu is None (for unknown status)
    let has_cpu = machine.get_item("cpu")?.is_some();
    let cpu_val: Option<f64> = machine.get_item("cpu")?.map(|v| v.extract()).transpose()?;

    let u = Usage {
        machine_id,
        kind,
        cores,
        region,
        zone,
        cpu: if has_cpu && cpu_val.is_some() {
            cpu
        } else {
            0.0
        },
        mem,
        free,
        network,
        disk,
    };

    let s = rust_status(&u, json_flag);

    // For unknown status, we need special handling
    if !has_cpu || cpu_val.is_none() {
        let dict = PyDict::new(py);
        // Copy all input fields
        for (key, value) in machine.iter() {
            dict.set_item(key, value)?;
        }
        dict.set_item("status", "unknown")?;

        if json_flag {
            dict.set_item("last_update", &s.last_update)?;
        } else {
            let datetime_module = py.import("datetime")?;
            let datetime_class = datetime_module.getattr("datetime")?;
            let dt = datetime_class.call_method1(
                "fromisoformat",
                (&s.last_update
                    .replace("T", " ")
                    .split('.')
                    .next()
                    .unwrap_or(&s.last_update),),
            )?;
            dict.set_item("last_update", dt)?;
        }
        return Ok(dict.into());
    }

    Ok(status_to_pydict(py, &s, json_flag)?.into_bound(py).into())
}

#[pyfunction]
#[pyo3(name = "jobs", signature = (machine, json=false, seed=None))]
pub fn py_jobs(
    py: Python<'_>,
    machine: &Bound<'_, PyDict>,
    json: bool,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let json_flag = json;

    // Extract machine fields
    let machine_id: String = machine
        .get_item("machine_id")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();
    let kind: String = machine
        .get_item("kind")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();
    let cores: i32 = machine
        .get_item("cores")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0);
    let region: String = machine
        .get_item("region")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();
    let zone: String = machine
        .get_item("zone")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();

    let m = Machine {
        machine_id,
        kind,
        cores,
        region,
        zone,
    };

    match rust_job(&m, json_flag, seed) {
        Some(j) => Ok(job_to_pydict(py, &j, json_flag)?.into_bound(py).into()),
        None => Ok(py.None()),
    }
}

// Helper functions for tests
#[pyfunction]
#[pyo3(name = "_id")]
pub fn py_id() -> String {
    use uuid::Uuid;
    let uuid = Uuid::new_v4().to_string();
    uuid.rsplit('-').next().unwrap().to_string()
}

#[pyfunction]
#[pyo3(name = "_clip")]
pub fn py_clip(value: f64, min: f64, max: f64) -> f64 {
    ((value.max(min).min(max) * 100.0).round()) / 100.0
}

#[pyfunction]
#[pyo3(name = "_randrange")]
pub fn py_randrange(low: f64, high: f64) -> f64 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    rng.gen::<f64>() * (high - low) + low
}

// =============================================================================
// Telemetry Functions
// =============================================================================

fn telemetry_to_pydict(py: Python<'_>, r: &TelemetryReading) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("timestamp", &r.timestamp)?;
    dict.set_item("machine_id", &r.machine_id)?;
    dict.set_item("kind", &r.kind)?;
    dict.set_item("cores", r.cores)?;
    dict.set_item("region", &r.region)?;
    dict.set_item("zone", &r.zone)?;
    dict.set_item("cpu", r.cpu)?;
    dict.set_item("mem", r.mem)?;
    dict.set_item("free", r.free)?;
    dict.set_item("network", r.network)?;
    dict.set_item("disk", r.disk)?;
    dict.set_item("state", &r.state)?;
    dict.set_item("anomaly_type", &r.anomaly_type)?;
    dict.set_item("health_score", r.health_score)?;
    Ok(dict.into())
}

/// Create pandas DataFrame from TelemetryReading rows
fn create_telemetry_pandas(py: Python<'_>, rows: &[TelemetryReading]) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let data = PyDict::new(py);

    let timestamps: Vec<&str> = rows.iter().map(|r| r.timestamp.as_str()).collect();
    let machine_ids: Vec<&str> = rows.iter().map(|r| r.machine_id.as_str()).collect();
    let kinds: Vec<&str> = rows.iter().map(|r| r.kind.as_str()).collect();
    let cores: Vec<i32> = rows.iter().map(|r| r.cores).collect();
    let regions: Vec<&str> = rows.iter().map(|r| r.region.as_str()).collect();
    let zones: Vec<&str> = rows.iter().map(|r| r.zone.as_str()).collect();
    let cpus: Vec<f64> = rows.iter().map(|r| r.cpu).collect();
    let mems: Vec<f64> = rows.iter().map(|r| r.mem).collect();
    let frees: Vec<f64> = rows.iter().map(|r| r.free).collect();
    let networks: Vec<f64> = rows.iter().map(|r| r.network).collect();
    let disks: Vec<f64> = rows.iter().map(|r| r.disk).collect();
    let states: Vec<&str> = rows.iter().map(|r| r.state.as_str()).collect();
    let anomaly_types: Vec<&str> = rows.iter().map(|r| r.anomaly_type.as_str()).collect();
    let health_scores: Vec<f64> = rows.iter().map(|r| r.health_score).collect();

    data.set_item("timestamp", PyList::new(py, &timestamps)?)?;
    data.set_item("machine_id", PyList::new(py, &machine_ids)?)?;
    data.set_item("kind", PyList::new(py, &kinds)?)?;
    data.set_item("cores", PyList::new(py, &cores)?)?;
    data.set_item("region", PyList::new(py, &regions)?)?;
    data.set_item("zone", PyList::new(py, &zones)?)?;
    data.set_item("cpu", PyList::new(py, &cpus)?)?;
    data.set_item("mem", PyList::new(py, &mems)?)?;
    data.set_item("free", PyList::new(py, &frees)?)?;
    data.set_item("network", PyList::new(py, &networks)?)?;
    data.set_item("disk", PyList::new(py, &disks)?)?;
    data.set_item("state", PyList::new(py, &states)?)?;
    data.set_item("anomaly_type", PyList::new(py, &anomaly_types)?)?;
    data.set_item("health_score", PyList::new(py, &health_scores)?)?;

    let df = pandas.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create polars DataFrame from TelemetryReading rows
fn create_telemetry_polars(py: Python<'_>, rows: &[TelemetryReading]) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let data = PyDict::new(py);

    let timestamps: Vec<&str> = rows.iter().map(|r| r.timestamp.as_str()).collect();
    let machine_ids: Vec<&str> = rows.iter().map(|r| r.machine_id.as_str()).collect();
    let kinds: Vec<&str> = rows.iter().map(|r| r.kind.as_str()).collect();
    let cores: Vec<i32> = rows.iter().map(|r| r.cores).collect();
    let regions: Vec<&str> = rows.iter().map(|r| r.region.as_str()).collect();
    let zones: Vec<&str> = rows.iter().map(|r| r.zone.as_str()).collect();
    let cpus: Vec<f64> = rows.iter().map(|r| r.cpu).collect();
    let mems: Vec<f64> = rows.iter().map(|r| r.mem).collect();
    let frees: Vec<f64> = rows.iter().map(|r| r.free).collect();
    let networks: Vec<f64> = rows.iter().map(|r| r.network).collect();
    let disks: Vec<f64> = rows.iter().map(|r| r.disk).collect();
    let states: Vec<&str> = rows.iter().map(|r| r.state.as_str()).collect();
    let anomaly_types: Vec<&str> = rows.iter().map(|r| r.anomaly_type.as_str()).collect();
    let health_scores: Vec<f64> = rows.iter().map(|r| r.health_score).collect();

    data.set_item("timestamp", PyList::new(py, &timestamps)?)?;
    data.set_item("machine_id", PyList::new(py, &machine_ids)?)?;
    data.set_item("kind", PyList::new(py, &kinds)?)?;
    data.set_item("cores", PyList::new(py, &cores)?)?;
    data.set_item("region", PyList::new(py, &regions)?)?;
    data.set_item("zone", PyList::new(py, &zones)?)?;
    data.set_item("cpu", PyList::new(py, &cpus)?)?;
    data.set_item("mem", PyList::new(py, &mems)?)?;
    data.set_item("free", PyList::new(py, &frees)?)?;
    data.set_item("network", PyList::new(py, &networks)?)?;
    data.set_item("disk", PyList::new(py, &disks)?)?;
    data.set_item("state", PyList::new(py, &states)?)?;
    data.set_item("anomaly_type", PyList::new(py, &anomaly_types)?)?;
    data.set_item("health_score", PyList::new(py, &health_scores)?)?;

    let df = polars.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Parse TelemetryConfig from Python dict
fn parse_telemetry_config(dict: &Bound<'_, PyDict>) -> PyResult<(TelemetryConfig, String)> {
    let machine_count: usize = dict
        .get_item("machine_count")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(10);

    let readings_per_machine: usize = dict
        .get_item("readings_per_machine")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(100);

    let seed: Option<u64> = dict.get_item("seed")?.and_then(|v| v.extract().ok());

    let start_time: Option<String> = dict.get_item("start_time")?.and_then(|v| v.extract().ok());

    let frequency_seconds: u32 = dict
        .get_item("frequency_seconds")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(60);

    let output: String = dict
        .get_item("output")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_else(|| "pandas".to_string());

    // Parse nested AnomalyConfig
    let anomalies = if let Some(anom_val) = dict.get_item("anomalies")? {
        if let Ok(anom_dict) = anom_val.downcast::<PyDict>() {
            AnomalyConfig {
                enable: anom_dict
                    .get_item("enable")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(false),
                cpu_spike_probability: anom_dict
                    .get_item("cpu_spike_probability")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.02),
                cpu_spike_magnitude: anom_dict
                    .get_item("cpu_spike_magnitude")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(40.0),
                memory_leak_probability: anom_dict
                    .get_item("memory_leak_probability")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.01),
                memory_leak_rate: anom_dict
                    .get_item("memory_leak_rate")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.5),
                network_saturation_probability: anom_dict
                    .get_item("network_saturation_probability")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.01),
                disk_fill_probability: anom_dict
                    .get_item("disk_fill_probability")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.005),
                multi_resource_probability: anom_dict
                    .get_item("multi_resource_probability")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.005),
                anomaly_duration_min: anom_dict
                    .get_item("anomaly_duration_min")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(5),
                anomaly_duration_max: anom_dict
                    .get_item("anomaly_duration_max")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(30),
            }
        } else {
            AnomalyConfig::default()
        }
    } else {
        AnomalyConfig::default()
    };

    // Parse nested SensorDriftConfig
    let sensor_drift = if let Some(sd_val) = dict.get_item("sensor_drift")? {
        if let Ok(sd_dict) = sd_val.downcast::<PyDict>() {
            SensorDriftConfig {
                enable: sd_dict
                    .get_item("enable")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(false),
                drift_rate: sd_dict
                    .get_item("drift_rate")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.001),
                drift_bias: sd_dict
                    .get_item("drift_bias")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.7),
                recalibration_probability: sd_dict
                    .get_item("recalibration_probability")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.01),
            }
        } else {
            SensorDriftConfig::default()
        }
    } else {
        SensorDriftConfig::default()
    };

    // Parse nested TemporalConfig
    let temporal = if let Some(temp_val) = dict.get_item("temporal")? {
        if let Ok(temp_dict) = temp_val.downcast::<PyDict>() {
            TemporalConfig {
                enable: temp_dict
                    .get_item("enable")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(false),
                diurnal_amplitude: temp_dict
                    .get_item("diurnal_amplitude")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.3),
                peak_hour: temp_dict
                    .get_item("peak_hour")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(14),
                weekend_reduction: temp_dict
                    .get_item("weekend_reduction")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.4),
            }
        } else {
            TemporalConfig::default()
        }
    } else {
        TemporalConfig::default()
    };

    // Parse nested FailureCascadeConfig
    let failure_cascade = if let Some(fc_val) = dict.get_item("failure_cascade")? {
        if let Ok(fc_dict) = fc_val.downcast::<PyDict>() {
            FailureCascadeConfig {
                enable: fc_dict
                    .get_item("enable")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(false),
                cascade_probability: fc_dict
                    .get_item("cascade_probability")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.3),
                cascade_delay_readings: fc_dict
                    .get_item("cascade_delay_readings")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(5),
                zone_correlation: fc_dict
                    .get_item("zone_correlation")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.7),
            }
        } else {
            FailureCascadeConfig::default()
        }
    } else {
        FailureCascadeConfig::default()
    };

    // Parse nested MaintenanceConfig
    let maintenance = if let Some(mt_val) = dict.get_item("maintenance")? {
        if let Ok(mt_dict) = mt_val.downcast::<PyDict>() {
            MaintenanceConfig {
                enable: mt_dict
                    .get_item("enable")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(false),
                window_probability: mt_dict
                    .get_item("window_probability")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.02),
                window_duration_min: mt_dict
                    .get_item("window_duration_min")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(10),
                window_duration_max: mt_dict
                    .get_item("window_duration_max")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(60),
                scheduled_hours: mt_dict
                    .get_item("scheduled_hours")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_else(|| vec![2, 3, 4]),
            }
        } else {
            MaintenanceConfig::default()
        }
    } else {
        MaintenanceConfig::default()
    };

    // Parse nested DegradationConfig
    let degradation = if let Some(dg_val) = dict.get_item("degradation")? {
        if let Ok(dg_dict) = dg_val.downcast::<PyDict>() {
            DegradationConfig {
                enable: dg_dict
                    .get_item("enable")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(false),
                degradation_rate: dg_dict
                    .get_item("degradation_rate")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.002),
                failure_threshold: dg_dict
                    .get_item("failure_threshold")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.95),
                recovery_rate: dg_dict
                    .get_item("recovery_rate")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.1),
            }
        } else {
            DegradationConfig::default()
        }
    } else {
        DegradationConfig::default()
    };

    let config = TelemetryConfig {
        machine_count,
        readings_per_machine,
        seed,
        start_time,
        frequency_seconds,
        anomalies,
        sensor_drift,
        temporal,
        failure_cascade,
        maintenance,
        degradation,
    };

    Ok((config, output))
}

/// Generate IoT telemetry data with configurable behaviors and preset scenarios.
///
/// # Arguments
/// * `config` - Optional TelemetryConfig dict or None for defaults
/// * `scenario` - Optional preset scenario name: "normal", "cpu_spikes", "memory_leak",
///   "network_congestion", "disk_pressure", "cascade_failure", "maintenance_window",
///   "sensor_drift", "degradation_cycle", "production", "chaos"
///
/// # Returns
/// * DataFrame (pandas/polars) with telemetry readings
#[pyfunction]
#[pyo3(signature = (config=None, scenario=None))]
pub fn telemetry(
    py: Python<'_>,
    config: Option<&Bound<'_, PyDict>>,
    scenario: Option<&str>,
) -> PyResult<Py<PyAny>> {
    // Start with preset scenario if specified
    let mut telemetry_config = match scenario {
        Some("normal") => TelemetryConfig::normal(),
        Some("cpu_spikes") => TelemetryConfig::cpu_spikes(),
        Some("memory_leak") => TelemetryConfig::memory_leak(),
        Some("network_congestion") => TelemetryConfig::network_congestion(),
        Some("disk_pressure") => TelemetryConfig::disk_pressure(),
        Some("cascade_failure") => TelemetryConfig::cascade_failure(),
        Some("maintenance_window") => TelemetryConfig::maintenance_window(),
        Some("sensor_drift") => TelemetryConfig::sensor_drift_scenario(),
        Some("degradation_cycle") => TelemetryConfig::degradation_cycle(),
        Some("production") => TelemetryConfig::production(),
        Some("chaos") => TelemetryConfig::chaos(),
        _ => TelemetryConfig::default(),
    };

    let mut output_format = "pandas".to_string();

    // Override with config if provided
    if let Some(dict) = config {
        let (parsed_config, out) = parse_telemetry_config(dict)?;
        // Merge: config overrides scenario defaults only for explicitly set values
        if dict.get_item("machine_count")?.is_some() {
            telemetry_config.machine_count = parsed_config.machine_count;
        }
        if dict.get_item("readings_per_machine")?.is_some() {
            telemetry_config.readings_per_machine = parsed_config.readings_per_machine;
        }
        if dict.get_item("seed")?.is_some() {
            telemetry_config.seed = parsed_config.seed;
        }
        if dict.get_item("start_time")?.is_some() {
            telemetry_config.start_time = parsed_config.start_time;
        }
        if dict.get_item("frequency_seconds")?.is_some() {
            telemetry_config.frequency_seconds = parsed_config.frequency_seconds;
        }
        if dict.get_item("anomalies")?.is_some() {
            telemetry_config.anomalies = parsed_config.anomalies;
        }
        if dict.get_item("sensor_drift")?.is_some() {
            telemetry_config.sensor_drift = parsed_config.sensor_drift;
        }
        if dict.get_item("temporal")?.is_some() {
            telemetry_config.temporal = parsed_config.temporal;
        }
        if dict.get_item("failure_cascade")?.is_some() {
            telemetry_config.failure_cascade = parsed_config.failure_cascade;
        }
        if dict.get_item("maintenance")?.is_some() {
            telemetry_config.maintenance = parsed_config.maintenance;
        }
        if dict.get_item("degradation")?.is_some() {
            telemetry_config.degradation = parsed_config.degradation;
        }
        output_format = out;
    }

    let readings = generate_telemetry(&telemetry_config);

    match output_format.to_lowercase().as_str() {
        "polars" => create_telemetry_polars(py, &readings),
        "dict" => {
            let list = PyList::empty(py);
            for r in &readings {
                list.append(telemetry_to_pydict(py, r)?)?;
            }
            Ok(list.into())
        }
        _ => create_telemetry_pandas(py, &readings),
    }
}

// Add schemas as module attributes
pub fn add_schemas(m: &Bound<PyModule>) -> PyResult<()> {
    let py = m.py();

    let machine_schema = PyDict::new(py);
    machine_schema.set_item("machine_id", "string")?;
    machine_schema.set_item("kind", "string")?;
    machine_schema.set_item("cores", "integer")?;
    machine_schema.set_item("region", "string")?;
    machine_schema.set_item("zone", "string")?;
    m.add("MACHINE_SCHEMA", machine_schema)?;

    let usage_schema = PyDict::new(py);
    usage_schema.set_item("machine_id", "string")?;
    usage_schema.set_item("kind", "string")?;
    usage_schema.set_item("cores", "integer")?;
    usage_schema.set_item("region", "string")?;
    usage_schema.set_item("zone", "string")?;
    usage_schema.set_item("cpu", "float")?;
    usage_schema.set_item("mem", "float")?;
    usage_schema.set_item("free", "float")?;
    usage_schema.set_item("network", "float")?;
    usage_schema.set_item("disk", "float")?;
    m.add("USAGE_SCHEMA", usage_schema)?;

    let status_schema = PyDict::new(py);
    status_schema.set_item("machine_id", "string")?;
    status_schema.set_item("kind", "string")?;
    status_schema.set_item("cores", "integer")?;
    status_schema.set_item("region", "string")?;
    status_schema.set_item("zone", "string")?;
    status_schema.set_item("cpu", "float")?;
    status_schema.set_item("mem", "float")?;
    status_schema.set_item("free", "float")?;
    status_schema.set_item("network", "float")?;
    status_schema.set_item("disk", "float")?;
    status_schema.set_item("status", "string")?;
    status_schema.set_item("last_update", "datetime")?;
    m.add("STATUS_SCHEMA", status_schema)?;

    let jobs_schema = PyDict::new(py);
    jobs_schema.set_item("machine_id", "string")?;
    jobs_schema.set_item("job_id", "string")?;
    jobs_schema.set_item("name", "string")?;
    jobs_schema.set_item("units", "integer")?;
    jobs_schema.set_item("start_time", "datetime")?;
    jobs_schema.set_item("end_time", "datetime")?;
    m.add("JOBS_SCHEMA", jobs_schema)?;

    let telemetry_schema = PyDict::new(py);
    telemetry_schema.set_item("timestamp", "datetime")?;
    telemetry_schema.set_item("machine_id", "string")?;
    telemetry_schema.set_item("kind", "string")?;
    telemetry_schema.set_item("cores", "integer")?;
    telemetry_schema.set_item("region", "string")?;
    telemetry_schema.set_item("zone", "string")?;
    telemetry_schema.set_item("cpu", "float")?;
    telemetry_schema.set_item("mem", "float")?;
    telemetry_schema.set_item("free", "float")?;
    telemetry_schema.set_item("network", "float")?;
    telemetry_schema.set_item("disk", "float")?;
    telemetry_schema.set_item("state", "string")?;
    telemetry_schema.set_item("anomaly_type", "string")?;
    telemetry_schema.set_item("health_score", "float")?;
    m.add("TELEMETRY_SCHEMA", telemetry_schema)?;

    // Telemetry scenarios list
    let scenarios = PyList::new(
        py,
        vec![
            "normal",
            "cpu_spikes",
            "memory_leak",
            "network_congestion",
            "disk_pressure",
            "cascade_failure",
            "maintenance_window",
            "sensor_drift",
            "degradation_cycle",
            "production",
            "chaos",
        ],
    )?;
    m.add("TELEMETRY_SCENARIOS", scenarios)?;

    Ok(())
}
