use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList};

use superstore::timeseries::{get_time_series, get_time_series_data};

#[pyfunction]
#[pyo3(name = "getTimeSeries")]
pub fn py_get_time_series(
    py: Python<'_>,
    nper: Option<usize>,
    freq: Option<&str>,
    ncol: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let nper = nper.unwrap_or(30);
    let freq = freq.unwrap_or("B");
    let ncol = ncol.unwrap_or(4);

    let data = get_time_series(nper, freq, ncol);

    // Import pandas
    let pandas = py.import("pandas")?;

    // Create dict of column name -> values
    let columns_dict = PyDict::new(py);
    for col in &data.columns {
        let values = PyList::new(py, &col.values)?;
        columns_dict.set_item(col.name.to_string(), values)?;
    }

    // Create DatetimeIndex from the index timestamps
    let index_list: Vec<String> = data
        .index
        .iter()
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
        .collect();
    let index_py = PyList::new(py, &index_list)?;
    let datetime_index = pandas.call_method1("DatetimeIndex", (index_py,))?;

    // Create DataFrame with data and index
    let kwargs = [("index", datetime_index)].into_py_dict(py)?;
    let df = pandas.call_method("DataFrame", (columns_dict,), Some(&kwargs))?;

    Ok(df.into())
}

#[pyfunction]
#[pyo3(name = "getTimeSeriesData")]
pub fn py_get_time_series_data(
    py: Python<'_>,
    nper: Option<usize>,
    freq: Option<&str>,
    ncol: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let nper = nper.unwrap_or(30);
    let freq = freq.unwrap_or("B");
    let ncol = ncol.unwrap_or(4);

    let data = get_time_series_data(nper, freq, ncol);

    // Import pandas
    let pandas = py.import("pandas")?;

    // Create dict of column name -> Series
    let result_dict = PyDict::new(py);
    for (col_name, (dates, values)) in &data {
        // Create DatetimeIndex
        let index_list: Vec<String> = dates
            .iter()
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .collect();
        let index_py = PyList::new(py, &index_list)?;
        let datetime_index = pandas.call_method1("DatetimeIndex", (index_py,))?;

        // Create Series
        let values_py = PyList::new(py, values)?;
        let kwargs = [("index", datetime_index)].into_py_dict(py)?;
        let series = pandas.call_method("Series", (values_py,), Some(&kwargs))?;

        result_dict.set_item(col_name.to_string(), series)?;
    }

    Ok(result_dict.into())
}
