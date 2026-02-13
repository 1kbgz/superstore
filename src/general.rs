use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use superstore::general::{employees, superstore, EmployeeRow, SuperstoreRow};

fn superstore_row_to_pydict(py: Python<'_>, row: &SuperstoreRow) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("Row ID", row.row_id)?;
    dict.set_item("Order ID", &row.order_id)?;
    dict.set_item("Order Date", &row.order_date)?;
    dict.set_item("Ship Date", &row.ship_date)?;
    dict.set_item("Ship Mode", &row.ship_mode)?;
    dict.set_item("Customer ID", &row.customer_id)?;
    dict.set_item("Segment", &row.segment)?;
    dict.set_item("Country", &row.country)?;
    dict.set_item("City", &row.city)?;
    dict.set_item("State", &row.state)?;
    dict.set_item("Postal Code", &row.postal_code)?;
    dict.set_item("Region", &row.region)?;
    dict.set_item("Product ID", &row.product_id)?;
    dict.set_item("Category", &row.category)?;
    dict.set_item("Sub-Category", &row.sub_category)?;
    dict.set_item("Sales", row.sales)?;
    dict.set_item("Quantity", row.quantity)?;
    dict.set_item("Discount", row.discount)?;
    dict.set_item("Profit", row.profit)?;
    Ok(dict.into())
}

fn employee_row_to_pydict(py: Python<'_>, row: &EmployeeRow) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("Row ID", row.row_id)?;
    dict.set_item("Employee ID", &row.employee_id)?;
    dict.set_item("First Name", &row.first_name)?;
    dict.set_item("Surname", &row.surname)?;
    dict.set_item("Prefix", &row.prefix)?;
    dict.set_item("Suffix", &row.suffix)?;
    dict.set_item("Phone Number", &row.phone_number)?;
    dict.set_item("Email", &row.email)?;
    dict.set_item("SSN", &row.ssn)?;
    dict.set_item("Street", &row.street)?;
    dict.set_item("City", &row.city)?;
    dict.set_item("Postal Code", &row.postal_code)?;
    dict.set_item("Region", &row.region)?;
    dict.set_item("State", &row.state)?;
    dict.set_item("Country", &row.country)?;
    // Convert dates to Python date objects
    dict.set_item("Start Date", row.start_date.to_string())?;
    dict.set_item("Date of Birth", row.date_of_birth.to_string())?;
    Ok(dict.into())
}

/// Create a pandas DataFrame from superstore rows
fn create_superstore_pandas(py: Python<'_>, rows: &[SuperstoreRow]) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let list = PyList::empty(py);
    for row in rows {
        list.append(superstore_row_to_pydict(py, row)?)?;
    }
    let df = pandas.call_method1("DataFrame", (list,))?;
    Ok(df.into())
}

/// Create a polars DataFrame from superstore rows
fn create_superstore_polars(py: Python<'_>, rows: &[SuperstoreRow]) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let data = PyDict::new(py);

    // Build column arrays
    let row_ids: Vec<i32> = rows.iter().map(|r| r.row_id).collect();
    let order_ids: Vec<&str> = rows.iter().map(|r| r.order_id.as_str()).collect();
    let order_dates: Vec<&str> = rows.iter().map(|r| r.order_date.as_str()).collect();
    let ship_dates: Vec<&str> = rows.iter().map(|r| r.ship_date.as_str()).collect();
    let ship_modes: Vec<&str> = rows.iter().map(|r| r.ship_mode.as_str()).collect();
    let customer_ids: Vec<&str> = rows.iter().map(|r| r.customer_id.as_str()).collect();
    let segments: Vec<&str> = rows.iter().map(|r| r.segment.as_str()).collect();
    let countries: Vec<&str> = rows.iter().map(|r| r.country.as_str()).collect();
    let cities: Vec<&str> = rows.iter().map(|r| r.city.as_str()).collect();
    let states: Vec<&str> = rows.iter().map(|r| r.state.as_str()).collect();
    let postal_codes: Vec<&str> = rows.iter().map(|r| r.postal_code.as_str()).collect();
    let regions: Vec<&str> = rows.iter().map(|r| r.region.as_str()).collect();
    let product_ids: Vec<&str> = rows.iter().map(|r| r.product_id.as_str()).collect();
    let categories: Vec<&str> = rows.iter().map(|r| r.category.as_str()).collect();
    let sub_categories: Vec<&str> = rows.iter().map(|r| r.sub_category.as_str()).collect();
    let sales: Vec<i32> = rows.iter().map(|r| r.sales).collect();
    let quantities: Vec<i32> = rows.iter().map(|r| r.quantity).collect();
    let discounts: Vec<f64> = rows.iter().map(|r| r.discount).collect();
    let profits: Vec<f64> = rows.iter().map(|r| r.profit).collect();

    data.set_item("Row ID", PyList::new(py, &row_ids)?)?;
    data.set_item("Order ID", PyList::new(py, &order_ids)?)?;
    data.set_item("Order Date", PyList::new(py, &order_dates)?)?;
    data.set_item("Ship Date", PyList::new(py, &ship_dates)?)?;
    data.set_item("Ship Mode", PyList::new(py, &ship_modes)?)?;
    data.set_item("Customer ID", PyList::new(py, &customer_ids)?)?;
    data.set_item("Segment", PyList::new(py, &segments)?)?;
    data.set_item("Country", PyList::new(py, &countries)?)?;
    data.set_item("City", PyList::new(py, &cities)?)?;
    data.set_item("State", PyList::new(py, &states)?)?;
    data.set_item("Postal Code", PyList::new(py, &postal_codes)?)?;
    data.set_item("Region", PyList::new(py, &regions)?)?;
    data.set_item("Product ID", PyList::new(py, &product_ids)?)?;
    data.set_item("Category", PyList::new(py, &categories)?)?;
    data.set_item("Sub-Category", PyList::new(py, &sub_categories)?)?;
    data.set_item("Sales", PyList::new(py, &sales)?)?;
    data.set_item("Quantity", PyList::new(py, &quantities)?)?;
    data.set_item("Discount", PyList::new(py, &discounts)?)?;
    data.set_item("Profit", PyList::new(py, &profits)?)?;

    let df = polars.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create a list of dicts from superstore rows
fn create_superstore_dict(py: Python<'_>, rows: &[SuperstoreRow]) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for row in rows {
        list.append(superstore_row_to_pydict(py, row)?)?;
    }
    Ok(list.into())
}

/// Create a pandas DataFrame from employee rows
fn create_employees_pandas(py: Python<'_>, rows: &[EmployeeRow]) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let list = PyList::empty(py);
    for row in rows {
        list.append(employee_row_to_pydict(py, row)?)?;
    }
    let df = pandas.call_method1("DataFrame", (list,))?;
    Ok(df.into())
}

/// Create a polars DataFrame from employee rows
fn create_employees_polars(py: Python<'_>, rows: &[EmployeeRow]) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let data = PyDict::new(py);

    let row_ids: Vec<i32> = rows.iter().map(|r| r.row_id).collect();
    let employee_ids: Vec<&str> = rows.iter().map(|r| r.employee_id.as_str()).collect();
    let first_names: Vec<&str> = rows.iter().map(|r| r.first_name.as_str()).collect();
    let surnames: Vec<&str> = rows.iter().map(|r| r.surname.as_str()).collect();
    let prefixes: Vec<&str> = rows.iter().map(|r| r.prefix.as_str()).collect();
    let suffixes: Vec<&str> = rows.iter().map(|r| r.suffix.as_str()).collect();
    let phone_numbers: Vec<&str> = rows.iter().map(|r| r.phone_number.as_str()).collect();
    let emails: Vec<&str> = rows.iter().map(|r| r.email.as_str()).collect();
    let ssns: Vec<&str> = rows.iter().map(|r| r.ssn.as_str()).collect();
    let streets: Vec<&str> = rows.iter().map(|r| r.street.as_str()).collect();
    let cities: Vec<&str> = rows.iter().map(|r| r.city.as_str()).collect();
    let postal_codes: Vec<&str> = rows.iter().map(|r| r.postal_code.as_str()).collect();
    let regions: Vec<&str> = rows.iter().map(|r| r.region.as_str()).collect();
    let states: Vec<&str> = rows.iter().map(|r| r.state.as_str()).collect();
    let countries: Vec<&str> = rows.iter().map(|r| r.country.as_str()).collect();
    let start_dates: Vec<String> = rows.iter().map(|r| r.start_date.to_string()).collect();
    let dobs: Vec<String> = rows.iter().map(|r| r.date_of_birth.to_string()).collect();

    data.set_item("Row ID", PyList::new(py, &row_ids)?)?;
    data.set_item("Employee ID", PyList::new(py, &employee_ids)?)?;
    data.set_item("First Name", PyList::new(py, &first_names)?)?;
    data.set_item("Surname", PyList::new(py, &surnames)?)?;
    data.set_item("Prefix", PyList::new(py, &prefixes)?)?;
    data.set_item("Suffix", PyList::new(py, &suffixes)?)?;
    data.set_item("Phone Number", PyList::new(py, &phone_numbers)?)?;
    data.set_item("Email", PyList::new(py, &emails)?)?;
    data.set_item("SSN", PyList::new(py, &ssns)?)?;
    data.set_item("Street", PyList::new(py, &streets)?)?;
    data.set_item("City", PyList::new(py, &cities)?)?;
    data.set_item("Postal Code", PyList::new(py, &postal_codes)?)?;
    data.set_item("Region", PyList::new(py, &regions)?)?;
    data.set_item("State", PyList::new(py, &states)?)?;
    data.set_item("Country", PyList::new(py, &countries)?)?;
    data.set_item("Start Date", PyList::new(py, &start_dates)?)?;
    data.set_item("Date of Birth", PyList::new(py, &dobs)?)?;

    let df = polars.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create a list of dicts from employee rows
fn create_employees_dict(py: Python<'_>, rows: &[EmployeeRow]) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for row in rows {
        list.append(employee_row_to_pydict(py, row)?)?;
    }
    Ok(list.into())
}

#[pyfunction]
#[pyo3(name = "superstore", signature = (count=1000, output="pandas", seed=None))]
pub fn py_superstore(
    py: Python<'_>,
    count: usize,
    output: &str,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let rows = superstore(count, seed);

    match output {
        "pandas" => create_superstore_pandas(py, &rows),
        "polars" => create_superstore_polars(py, &rows),
        "dict" => create_superstore_dict(py, &rows),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid output format '{}'. Must be 'pandas', 'polars', or 'dict'",
            output
        ))),
    }
}

#[pyfunction]
#[pyo3(name = "employees", signature = (count=1000, output="pandas", seed=None))]
pub fn py_employees(
    py: Python<'_>,
    count: usize,
    output: &str,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let rows = employees(count, seed);

    match output {
        "pandas" => create_employees_pandas(py, &rows),
        "polars" => create_employees_polars(py, &rows),
        "dict" => create_employees_dict(py, &rows),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid output format '{}'. Must be 'pandas', 'polars', or 'dict'",
            output
        ))),
    }
}
