use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use superstore::crossfilter::{
    job as rust_job, machines as rust_machines, status as rust_status, usage as rust_usage, Job,
    Machine, Status, Usage,
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

#[pyfunction]
#[pyo3(name = "machines")]
pub fn py_machines(py: Python<'_>, count: Option<usize>, json: Option<bool>) -> PyResult<PyObject> {
    let count = count.unwrap_or(100);
    let _json = json.unwrap_or(false);

    let machines = rust_machines(count);
    let list = PyList::empty(py);
    for m in &machines {
        list.append(machine_to_pydict(py, m)?)?;
    }

    Ok(list.into())
}

#[pyfunction]
#[pyo3(name = "usage")]
pub fn py_usage(
    py: Python<'_>,
    machine: &Bound<'_, PyDict>,
    json: Option<bool>,
) -> PyResult<PyObject> {
    let _json = json.unwrap_or(false);

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
        let u = rust_usage(&m, None);
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

    let u = rust_usage(&m, Some(&prev_usage));
    Ok(usage_to_pydict(py, &u)?.into_bound(py).into())
}

#[pyfunction]
#[pyo3(name = "status")]
pub fn py_status(
    py: Python<'_>,
    machine: &Bound<'_, PyDict>,
    json: Option<bool>,
) -> PyResult<PyObject> {
    let json_flag = json.unwrap_or(false);

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
#[pyo3(name = "jobs")]
pub fn py_jobs(
    py: Python<'_>,
    machine: &Bound<'_, PyDict>,
    json: Option<bool>,
) -> PyResult<PyObject> {
    let json_flag = json.unwrap_or(false);

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

    match rust_job(&m, json_flag) {
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

    Ok(())
}
