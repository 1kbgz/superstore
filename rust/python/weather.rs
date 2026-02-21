use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use superstore::weather::{generate_weather, ClimateZone, WeatherConfig, WeatherReading};

/// Convert Python dict to WeatherConfig
fn dict_to_weather_config(
    py: Python<'_>,
    config_dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<WeatherConfig> {
    let mut config = WeatherConfig::default();

    if let Some(dict) = config_dict {
        if let Some(val) = dict.get_item("count")? {
            config.count = val.extract()?;
        }
        if let Some(val) = dict.get_item("seed")? {
            config.seed = val.extract()?;
        }
        if let Some(val) = dict.get_item("start_date")? {
            let s: Option<String> = val.extract()?;
            config.start_date = s;
        }
        if let Some(val) = dict.get_item("frequency_minutes")? {
            config.frequency_minutes = val.extract()?;
        }
        if let Some(val) = dict.get_item("climate_zone")? {
            let zone_str: String = val.extract()?;
            config.climate_zone = match zone_str.as_str() {
                "tropical" => ClimateZone::Tropical,
                "subtropical" => ClimateZone::Subtropical,
                "temperate" => ClimateZone::Temperate,
                "continental" => ClimateZone::Continental,
                "polar" => ClimateZone::Polar,
                "arid" => ClimateZone::Arid,
                "mediterranean" => ClimateZone::Mediterranean,
                _ => ClimateZone::Temperate,
            };
        }
        if let Some(val) = dict.get_item("latitude")? {
            config.latitude = val.extract()?;
        }
        if let Some(val) = dict.get_item("base_temp_celsius")? {
            config.base_temp_celsius = val.extract()?;
        }
        if let Some(val) = dict.get_item("temp_daily_amplitude")? {
            config.temp_daily_amplitude = val.extract()?;
        }
        if let Some(val) = dict.get_item("temp_seasonal_amplitude")? {
            config.temp_seasonal_amplitude = val.extract()?;
        }
        if let Some(val) = dict.get_item("temp_noise_stddev")? {
            config.temp_noise_stddev = val.extract()?;
        }
        if let Some(val) = dict.get_item("base_humidity_percent")? {
            config.base_humidity_percent = val.extract()?;
        }
        if let Some(val) = dict.get_item("humidity_temp_correlation")? {
            config.humidity_temp_correlation = val.extract()?;
        }
        if let Some(val) = dict.get_item("precipitation_probability")? {
            config.precipitation_probability = val.extract()?;
        }
        if let Some(val) = dict.get_item("enable_weather_events")? {
            config.enable_weather_events = val.extract()?;
        }
        if let Some(val) = dict.get_item("event_probability")? {
            config.event_probability = val.extract()?;
        }
        if let Some(val) = dict.get_item("outlier_probability")? {
            config.outlier_probability = val.extract()?;
        }
        if let Some(val) = dict.get_item("sensor_drift")? {
            config.sensor_drift = val.extract()?;
        }
        if let Some(val) = dict.get_item("sensor_drift_rate")? {
            config.sensor_drift_rate = val.extract()?;
        }
    }

    Ok(config)
}

/// Convert WeatherReading to Python dict
fn weather_reading_to_pydict(py: Python<'_>, reading: &WeatherReading) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("timestamp", &reading.timestamp)?;
    dict.set_item("temperature_celsius", reading.temperature_celsius)?;
    dict.set_item("humidity_percent", reading.humidity_percent)?;
    dict.set_item("precipitation_mm", reading.precipitation_mm)?;
    dict.set_item("weather_event", &reading.weather_event)?;
    dict.set_item("is_outlier", reading.is_outlier)?;
    Ok(dict.into())
}

/// Create pandas DataFrame from weather readings
fn create_weather_pandas(py: Python<'_>, readings: &[WeatherReading]) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let data = PyDict::new(py);

    let timestamps: Vec<&str> = readings.iter().map(|r| r.timestamp.as_str()).collect();
    let temperatures: Vec<f64> = readings.iter().map(|r| r.temperature_celsius).collect();
    let humidities: Vec<f64> = readings.iter().map(|r| r.humidity_percent).collect();
    let precipitations: Vec<f64> = readings.iter().map(|r| r.precipitation_mm).collect();
    let events: Vec<&str> = readings.iter().map(|r| r.weather_event.as_str()).collect();
    let outliers: Vec<bool> = readings.iter().map(|r| r.is_outlier).collect();

    data.set_item("timestamp", PyList::new(py, &timestamps)?)?;
    data.set_item("temperature_celsius", PyList::new(py, &temperatures)?)?;
    data.set_item("humidity_percent", PyList::new(py, &humidities)?)?;
    data.set_item("precipitation_mm", PyList::new(py, &precipitations)?)?;
    data.set_item("weather_event", PyList::new(py, &events)?)?;
    data.set_item("is_outlier", PyList::new(py, &outliers)?)?;

    let df = pandas.call_method1("DataFrame", (data,))?;

    // Convert timestamp column to datetime
    let df = df.call_method1("assign", ())?;
    let kwargs = PyDict::new(py);
    let timestamp_col = df.getattr("timestamp")?;
    let datetime_col = pandas.call_method1("to_datetime", (timestamp_col,))?;
    kwargs.set_item("timestamp", datetime_col)?;
    let df = df.call_method("assign", (), Some(&kwargs))?;

    Ok(df.into())
}

/// Create polars DataFrame from weather readings
fn create_weather_polars(py: Python<'_>, readings: &[WeatherReading]) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let data = PyDict::new(py);

    let timestamps: Vec<&str> = readings.iter().map(|r| r.timestamp.as_str()).collect();
    let temperatures: Vec<f64> = readings.iter().map(|r| r.temperature_celsius).collect();
    let humidities: Vec<f64> = readings.iter().map(|r| r.humidity_percent).collect();
    let precipitations: Vec<f64> = readings.iter().map(|r| r.precipitation_mm).collect();
    let events: Vec<&str> = readings.iter().map(|r| r.weather_event.as_str()).collect();
    let outliers: Vec<bool> = readings.iter().map(|r| r.is_outlier).collect();

    data.set_item("timestamp", PyList::new(py, &timestamps)?)?;
    data.set_item("temperature_celsius", PyList::new(py, &temperatures)?)?;
    data.set_item("humidity_percent", PyList::new(py, &humidities)?)?;
    data.set_item("precipitation_mm", PyList::new(py, &precipitations)?)?;
    data.set_item("weather_event", PyList::new(py, &events)?)?;
    data.set_item("is_outlier", PyList::new(py, &outliers)?)?;

    let df = polars.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create list of dicts from weather readings
fn create_weather_dict(py: Python<'_>, readings: &[WeatherReading]) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for reading in readings {
        list.append(weather_reading_to_pydict(py, reading)?)?;
    }
    Ok(list.into())
}

/// Generate weather data with structured configuration.
///
/// Args:
///     config: Optional WeatherConfig pydantic model or dict with configuration.
///             If None, uses default configuration.
///     count: Number of readings (overrides config if provided)
///     output: Output format ("pandas", "polars", or "dict")
///     seed: Random seed (overrides config if provided)
///
/// Returns:
///     Weather sensor data in the specified format.
#[pyfunction]
#[pyo3(name = "weather", signature = (config=None, count=None, output="pandas", seed=None))]
pub fn py_weather(
    py: Python<'_>,
    config: Option<&Bound<'_, PyAny>>,
    count: Option<usize>,
    output: &str,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    // Parse config from pydantic model or dict
    let mut weather_config = if let Some(cfg) = config {
        // Check if it's a pydantic model (has model_dump method)
        if cfg.hasattr("model_dump")? {
            // Use mode="json" to ensure enums are serialized as strings
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("mode", "json")?;
            let dict = cfg.call_method("model_dump", (), Some(&kwargs))?;
            let dict = dict.downcast::<PyDict>()?;
            dict_to_weather_config(py, Some(dict))?
        } else if let Ok(dict) = cfg.downcast::<PyDict>() {
            dict_to_weather_config(py, Some(dict))?
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "config must be a WeatherConfig, dict, or None",
            ));
        }
    } else {
        WeatherConfig::default()
    };

    // Override with explicit parameters if provided
    if let Some(c) = count {
        weather_config.count = c;
    }
    if let Some(s) = seed {
        weather_config.seed = Some(s);
    }

    let readings = generate_weather(&weather_config);

    match output {
        "pandas" => create_weather_pandas(py, &readings),
        "polars" => create_weather_polars(py, &readings),
        "dict" => create_weather_dict(py, &readings),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid output format '{}'. Must be 'pandas', 'polars', or 'dict'",
            output
        ))),
    }
}

/// Register weather functions with the Python module
pub fn register_weather(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_weather, m)?)?;
    Ok(())
}
