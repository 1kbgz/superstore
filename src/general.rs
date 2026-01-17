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

#[pyfunction]
#[pyo3(name = "superstore")]
pub fn py_superstore(py: Python<'_>, count: Option<usize>) -> PyResult<PyObject> {
    let count = count.unwrap_or(1000);
    let rows = superstore(count);

    // Import pandas
    let pandas = py.import("pandas")?;

    // Create list of dicts
    let list = PyList::empty(py);
    for row in &rows {
        list.append(superstore_row_to_pydict(py, row)?)?;
    }

    // Create DataFrame
    let df = pandas.call_method1("DataFrame", (list,))?;
    Ok(df.into())
}

#[pyfunction]
#[pyo3(name = "employees")]
pub fn py_employees(py: Python<'_>, count: Option<usize>) -> PyResult<PyObject> {
    let count = count.unwrap_or(1000);
    let rows = employees(count);

    // Import pandas
    let pandas = py.import("pandas")?;

    // Create list of dicts
    let list = PyList::empty(py);
    for row in &rows {
        list.append(employee_row_to_pydict(py, row)?)?;
    }

    // Create DataFrame
    let df = pandas.call_method1("DataFrame", (list,))?;
    Ok(df.into())
}
