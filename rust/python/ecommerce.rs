use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use superstore::ecommerce::{
    ecommerce, generate_cart_events, generate_catalog, generate_customers, generate_funnel_events,
    generate_orders, generate_sessions, CartConfig, CartEvent, CatalogConfig, Customer,
    EcommerceConfig, EcommerceData, FunnelConfig, FunnelEvent, Order, Product, RfmConfig, Session,
    SessionConfig,
};

// =============================================================================
// Helper Functions for creating DataFrames
// =============================================================================

/// Create pandas DataFrame from Session rows
fn create_sessions_pandas(py: Python<'_>, rows: &[Session]) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let data = PyDict::new(py);

    let session_ids: Vec<&str> = rows.iter().map(|r| r.session_id.as_str()).collect();
    let user_ids: Vec<&str> = rows.iter().map(|r| r.user_id.as_str()).collect();
    let start_times: Vec<&str> = rows.iter().map(|r| r.start_time.as_str()).collect();
    let end_times: Vec<&str> = rows.iter().map(|r| r.end_time.as_str()).collect();
    let durations: Vec<u32> = rows.iter().map(|r| r.duration_seconds).collect();
    let devices: Vec<&str> = rows.iter().map(|r| r.device_type.as_str()).collect();
    let browsers: Vec<&str> = rows.iter().map(|r| r.browser.as_str()).collect();
    let sources: Vec<&str> = rows.iter().map(|r| r.traffic_source.as_str()).collect();
    let landings: Vec<&str> = rows.iter().map(|r| r.landing_page.as_str()).collect();
    let pages: Vec<u32> = rows.iter().map(|r| r.pages_viewed).collect();
    let bounced: Vec<bool> = rows.iter().map(|r| r.bounced).collect();
    let converted: Vec<bool> = rows.iter().map(|r| r.converted).collect();
    let values: Vec<f64> = rows.iter().map(|r| r.total_value).collect();

    data.set_item("session_id", PyList::new(py, &session_ids)?)?;
    data.set_item("user_id", PyList::new(py, &user_ids)?)?;
    data.set_item("start_time", PyList::new(py, &start_times)?)?;
    data.set_item("end_time", PyList::new(py, &end_times)?)?;
    data.set_item("duration_seconds", PyList::new(py, &durations)?)?;
    data.set_item("device_type", PyList::new(py, &devices)?)?;
    data.set_item("browser", PyList::new(py, &browsers)?)?;
    data.set_item("traffic_source", PyList::new(py, &sources)?)?;
    data.set_item("landing_page", PyList::new(py, &landings)?)?;
    data.set_item("pages_viewed", PyList::new(py, &pages)?)?;
    data.set_item("bounced", PyList::new(py, &bounced)?)?;
    data.set_item("converted", PyList::new(py, &converted)?)?;
    data.set_item("total_value", PyList::new(py, &values)?)?;

    let df = pandas.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create polars DataFrame from Session rows
fn create_sessions_polars(py: Python<'_>, rows: &[Session]) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let data = PyDict::new(py);

    let session_ids: Vec<&str> = rows.iter().map(|r| r.session_id.as_str()).collect();
    let user_ids: Vec<&str> = rows.iter().map(|r| r.user_id.as_str()).collect();
    let start_times: Vec<&str> = rows.iter().map(|r| r.start_time.as_str()).collect();
    let end_times: Vec<&str> = rows.iter().map(|r| r.end_time.as_str()).collect();
    let durations: Vec<u32> = rows.iter().map(|r| r.duration_seconds).collect();
    let devices: Vec<&str> = rows.iter().map(|r| r.device_type.as_str()).collect();
    let browsers: Vec<&str> = rows.iter().map(|r| r.browser.as_str()).collect();
    let sources: Vec<&str> = rows.iter().map(|r| r.traffic_source.as_str()).collect();
    let landings: Vec<&str> = rows.iter().map(|r| r.landing_page.as_str()).collect();
    let pages: Vec<u32> = rows.iter().map(|r| r.pages_viewed).collect();
    let bounced: Vec<bool> = rows.iter().map(|r| r.bounced).collect();
    let converted: Vec<bool> = rows.iter().map(|r| r.converted).collect();
    let values: Vec<f64> = rows.iter().map(|r| r.total_value).collect();

    data.set_item("session_id", PyList::new(py, &session_ids)?)?;
    data.set_item("user_id", PyList::new(py, &user_ids)?)?;
    data.set_item("start_time", PyList::new(py, &start_times)?)?;
    data.set_item("end_time", PyList::new(py, &end_times)?)?;
    data.set_item("duration_seconds", PyList::new(py, &durations)?)?;
    data.set_item("device_type", PyList::new(py, &devices)?)?;
    data.set_item("browser", PyList::new(py, &browsers)?)?;
    data.set_item("traffic_source", PyList::new(py, &sources)?)?;
    data.set_item("landing_page", PyList::new(py, &landings)?)?;
    data.set_item("pages_viewed", PyList::new(py, &pages)?)?;
    data.set_item("bounced", PyList::new(py, &bounced)?)?;
    data.set_item("converted", PyList::new(py, &converted)?)?;
    data.set_item("total_value", PyList::new(py, &values)?)?;

    let df = polars.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create dict from Session rows
fn create_sessions_dict(py: Python<'_>, rows: &[Session]) -> PyResult<Py<PyAny>> {
    let data = PyDict::new(py);

    let session_ids: Vec<&str> = rows.iter().map(|r| r.session_id.as_str()).collect();
    let user_ids: Vec<&str> = rows.iter().map(|r| r.user_id.as_str()).collect();
    let start_times: Vec<&str> = rows.iter().map(|r| r.start_time.as_str()).collect();
    let end_times: Vec<&str> = rows.iter().map(|r| r.end_time.as_str()).collect();
    let durations: Vec<u32> = rows.iter().map(|r| r.duration_seconds).collect();
    let devices: Vec<&str> = rows.iter().map(|r| r.device_type.as_str()).collect();
    let browsers: Vec<&str> = rows.iter().map(|r| r.browser.as_str()).collect();
    let sources: Vec<&str> = rows.iter().map(|r| r.traffic_source.as_str()).collect();
    let landings: Vec<&str> = rows.iter().map(|r| r.landing_page.as_str()).collect();
    let pages: Vec<u32> = rows.iter().map(|r| r.pages_viewed).collect();
    let bounced: Vec<bool> = rows.iter().map(|r| r.bounced).collect();
    let converted: Vec<bool> = rows.iter().map(|r| r.converted).collect();
    let values: Vec<f64> = rows.iter().map(|r| r.total_value).collect();

    data.set_item("session_id", PyList::new(py, &session_ids)?)?;
    data.set_item("user_id", PyList::new(py, &user_ids)?)?;
    data.set_item("start_time", PyList::new(py, &start_times)?)?;
    data.set_item("end_time", PyList::new(py, &end_times)?)?;
    data.set_item("duration_seconds", PyList::new(py, &durations)?)?;
    data.set_item("device_type", PyList::new(py, &devices)?)?;
    data.set_item("browser", PyList::new(py, &browsers)?)?;
    data.set_item("traffic_source", PyList::new(py, &sources)?)?;
    data.set_item("landing_page", PyList::new(py, &landings)?)?;
    data.set_item("pages_viewed", PyList::new(py, &pages)?)?;
    data.set_item("bounced", PyList::new(py, &bounced)?)?;
    data.set_item("converted", PyList::new(py, &converted)?)?;
    data.set_item("total_value", PyList::new(py, &values)?)?;

    Ok(data.into())
}

/// Create pandas DataFrame from Product rows
fn create_products_pandas(py: Python<'_>, rows: &[Product]) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let data = PyDict::new(py);

    let ids: Vec<&str> = rows.iter().map(|r| r.product_id.as_str()).collect();
    let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
    let categories: Vec<&str> = rows.iter().map(|r| r.category.as_str()).collect();
    let subcategories: Vec<&str> = rows.iter().map(|r| r.subcategory.as_str()).collect();
    let prices: Vec<f64> = rows.iter().map(|r| r.price).collect();
    let ratings: Vec<f64> = rows.iter().map(|r| r.rating).collect();
    let reviews: Vec<u32> = rows.iter().map(|r| r.review_count).collect();
    let in_stock: Vec<bool> = rows.iter().map(|r| r.in_stock).collect();

    data.set_item("product_id", PyList::new(py, &ids)?)?;
    data.set_item("name", PyList::new(py, &names)?)?;
    data.set_item("category", PyList::new(py, &categories)?)?;
    data.set_item("subcategory", PyList::new(py, &subcategories)?)?;
    data.set_item("price", PyList::new(py, &prices)?)?;
    data.set_item("rating", PyList::new(py, &ratings)?)?;
    data.set_item("review_count", PyList::new(py, &reviews)?)?;
    data.set_item("in_stock", PyList::new(py, &in_stock)?)?;

    let df = pandas.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create polars DataFrame from Product rows
fn create_products_polars(py: Python<'_>, rows: &[Product]) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let data = PyDict::new(py);

    let ids: Vec<&str> = rows.iter().map(|r| r.product_id.as_str()).collect();
    let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
    let categories: Vec<&str> = rows.iter().map(|r| r.category.as_str()).collect();
    let subcategories: Vec<&str> = rows.iter().map(|r| r.subcategory.as_str()).collect();
    let prices: Vec<f64> = rows.iter().map(|r| r.price).collect();
    let ratings: Vec<f64> = rows.iter().map(|r| r.rating).collect();
    let reviews: Vec<u32> = rows.iter().map(|r| r.review_count).collect();
    let in_stock: Vec<bool> = rows.iter().map(|r| r.in_stock).collect();

    data.set_item("product_id", PyList::new(py, &ids)?)?;
    data.set_item("name", PyList::new(py, &names)?)?;
    data.set_item("category", PyList::new(py, &categories)?)?;
    data.set_item("subcategory", PyList::new(py, &subcategories)?)?;
    data.set_item("price", PyList::new(py, &prices)?)?;
    data.set_item("rating", PyList::new(py, &ratings)?)?;
    data.set_item("review_count", PyList::new(py, &reviews)?)?;
    data.set_item("in_stock", PyList::new(py, &in_stock)?)?;

    let df = polars.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create dict from Product rows
fn create_products_dict(py: Python<'_>, rows: &[Product]) -> PyResult<Py<PyAny>> {
    let data = PyDict::new(py);

    let ids: Vec<&str> = rows.iter().map(|r| r.product_id.as_str()).collect();
    let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
    let categories: Vec<&str> = rows.iter().map(|r| r.category.as_str()).collect();
    let subcategories: Vec<&str> = rows.iter().map(|r| r.subcategory.as_str()).collect();
    let prices: Vec<f64> = rows.iter().map(|r| r.price).collect();
    let ratings: Vec<f64> = rows.iter().map(|r| r.rating).collect();
    let reviews: Vec<u32> = rows.iter().map(|r| r.review_count).collect();
    let in_stock: Vec<bool> = rows.iter().map(|r| r.in_stock).collect();

    data.set_item("product_id", PyList::new(py, &ids)?)?;
    data.set_item("name", PyList::new(py, &names)?)?;
    data.set_item("category", PyList::new(py, &categories)?)?;
    data.set_item("subcategory", PyList::new(py, &subcategories)?)?;
    data.set_item("price", PyList::new(py, &prices)?)?;
    data.set_item("rating", PyList::new(py, &ratings)?)?;
    data.set_item("review_count", PyList::new(py, &reviews)?)?;
    data.set_item("in_stock", PyList::new(py, &in_stock)?)?;

    Ok(data.into())
}

/// Create pandas DataFrame from Order rows
fn create_orders_pandas(py: Python<'_>, rows: &[Order]) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let data = PyDict::new(py);

    let ids: Vec<&str> = rows.iter().map(|r| r.order_id.as_str()).collect();
    let user_ids: Vec<&str> = rows.iter().map(|r| r.user_id.as_str()).collect();
    let session_ids: Vec<&str> = rows.iter().map(|r| r.session_id.as_str()).collect();
    let times: Vec<&str> = rows.iter().map(|r| r.order_time.as_str()).collect();
    let items: Vec<u32> = rows.iter().map(|r| r.total_items).collect();
    let subtotals: Vec<f64> = rows.iter().map(|r| r.subtotal).collect();
    let discounts: Vec<f64> = rows.iter().map(|r| r.discount).collect();
    let taxes: Vec<f64> = rows.iter().map(|r| r.tax).collect();
    let shippings: Vec<f64> = rows.iter().map(|r| r.shipping).collect();
    let totals: Vec<f64> = rows.iter().map(|r| r.total).collect();
    let payments: Vec<&str> = rows.iter().map(|r| r.payment_method.as_str()).collect();
    let statuses: Vec<&str> = rows.iter().map(|r| r.status.as_str()).collect();

    data.set_item("order_id", PyList::new(py, &ids)?)?;
    data.set_item("user_id", PyList::new(py, &user_ids)?)?;
    data.set_item("session_id", PyList::new(py, &session_ids)?)?;
    data.set_item("order_time", PyList::new(py, &times)?)?;
    data.set_item("total_items", PyList::new(py, &items)?)?;
    data.set_item("subtotal", PyList::new(py, &subtotals)?)?;
    data.set_item("discount", PyList::new(py, &discounts)?)?;
    data.set_item("tax", PyList::new(py, &taxes)?)?;
    data.set_item("shipping", PyList::new(py, &shippings)?)?;
    data.set_item("total", PyList::new(py, &totals)?)?;
    data.set_item("payment_method", PyList::new(py, &payments)?)?;
    data.set_item("status", PyList::new(py, &statuses)?)?;

    let df = pandas.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create polars DataFrame from Order rows
fn create_orders_polars(py: Python<'_>, rows: &[Order]) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let data = PyDict::new(py);

    let ids: Vec<&str> = rows.iter().map(|r| r.order_id.as_str()).collect();
    let user_ids: Vec<&str> = rows.iter().map(|r| r.user_id.as_str()).collect();
    let session_ids: Vec<&str> = rows.iter().map(|r| r.session_id.as_str()).collect();
    let times: Vec<&str> = rows.iter().map(|r| r.order_time.as_str()).collect();
    let items: Vec<u32> = rows.iter().map(|r| r.total_items).collect();
    let subtotals: Vec<f64> = rows.iter().map(|r| r.subtotal).collect();
    let discounts: Vec<f64> = rows.iter().map(|r| r.discount).collect();
    let taxes: Vec<f64> = rows.iter().map(|r| r.tax).collect();
    let shippings: Vec<f64> = rows.iter().map(|r| r.shipping).collect();
    let totals: Vec<f64> = rows.iter().map(|r| r.total).collect();
    let payments: Vec<&str> = rows.iter().map(|r| r.payment_method.as_str()).collect();
    let statuses: Vec<&str> = rows.iter().map(|r| r.status.as_str()).collect();

    data.set_item("order_id", PyList::new(py, &ids)?)?;
    data.set_item("user_id", PyList::new(py, &user_ids)?)?;
    data.set_item("session_id", PyList::new(py, &session_ids)?)?;
    data.set_item("order_time", PyList::new(py, &times)?)?;
    data.set_item("total_items", PyList::new(py, &items)?)?;
    data.set_item("subtotal", PyList::new(py, &subtotals)?)?;
    data.set_item("discount", PyList::new(py, &discounts)?)?;
    data.set_item("tax", PyList::new(py, &taxes)?)?;
    data.set_item("shipping", PyList::new(py, &shippings)?)?;
    data.set_item("total", PyList::new(py, &totals)?)?;
    data.set_item("payment_method", PyList::new(py, &payments)?)?;
    data.set_item("status", PyList::new(py, &statuses)?)?;

    let df = polars.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create dict from Order rows
fn create_orders_dict(py: Python<'_>, rows: &[Order]) -> PyResult<Py<PyAny>> {
    let data = PyDict::new(py);

    let ids: Vec<&str> = rows.iter().map(|r| r.order_id.as_str()).collect();
    let user_ids: Vec<&str> = rows.iter().map(|r| r.user_id.as_str()).collect();
    let session_ids: Vec<&str> = rows.iter().map(|r| r.session_id.as_str()).collect();
    let times: Vec<&str> = rows.iter().map(|r| r.order_time.as_str()).collect();
    let items: Vec<u32> = rows.iter().map(|r| r.total_items).collect();
    let subtotals: Vec<f64> = rows.iter().map(|r| r.subtotal).collect();
    let discounts: Vec<f64> = rows.iter().map(|r| r.discount).collect();
    let taxes: Vec<f64> = rows.iter().map(|r| r.tax).collect();
    let shippings: Vec<f64> = rows.iter().map(|r| r.shipping).collect();
    let totals: Vec<f64> = rows.iter().map(|r| r.total).collect();
    let payments: Vec<&str> = rows.iter().map(|r| r.payment_method.as_str()).collect();
    let statuses: Vec<&str> = rows.iter().map(|r| r.status.as_str()).collect();

    data.set_item("order_id", PyList::new(py, &ids)?)?;
    data.set_item("user_id", PyList::new(py, &user_ids)?)?;
    data.set_item("session_id", PyList::new(py, &session_ids)?)?;
    data.set_item("order_time", PyList::new(py, &times)?)?;
    data.set_item("total_items", PyList::new(py, &items)?)?;
    data.set_item("subtotal", PyList::new(py, &subtotals)?)?;
    data.set_item("discount", PyList::new(py, &discounts)?)?;
    data.set_item("tax", PyList::new(py, &taxes)?)?;
    data.set_item("shipping", PyList::new(py, &shippings)?)?;
    data.set_item("total", PyList::new(py, &totals)?)?;
    data.set_item("payment_method", PyList::new(py, &payments)?)?;
    data.set_item("status", PyList::new(py, &statuses)?)?;

    Ok(data.into())
}

/// Create pandas DataFrame from Customer rows
fn create_customers_pandas(py: Python<'_>, rows: &[Customer]) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let data = PyDict::new(py);

    let ids: Vec<&str> = rows.iter().map(|r| r.customer_id.as_str()).collect();
    let emails: Vec<&str> = rows.iter().map(|r| r.email.as_str()).collect();
    let first_orders: Vec<Option<&str>> =
        rows.iter().map(|r| r.first_order_date.as_deref()).collect();
    let last_orders: Vec<Option<&str>> =
        rows.iter().map(|r| r.last_order_date.as_deref()).collect();
    let total_orders: Vec<u32> = rows.iter().map(|r| r.total_orders).collect();
    let total_spent: Vec<f64> = rows.iter().map(|r| r.total_spent).collect();
    let avg_values: Vec<f64> = rows.iter().map(|r| r.avg_order_value).collect();
    let recency: Vec<u32> = rows.iter().map(|r| r.rfm_recency).collect();
    let frequency: Vec<u32> = rows.iter().map(|r| r.rfm_frequency).collect();
    let monetary: Vec<f64> = rows.iter().map(|r| r.rfm_monetary).collect();
    let scores: Vec<&str> = rows.iter().map(|r| r.rfm_score.as_str()).collect();
    let segments: Vec<&str> = rows.iter().map(|r| r.rfm_segment.as_str()).collect();

    data.set_item("customer_id", PyList::new(py, &ids)?)?;
    data.set_item("email", PyList::new(py, &emails)?)?;
    data.set_item("first_order_date", PyList::new(py, &first_orders)?)?;
    data.set_item("last_order_date", PyList::new(py, &last_orders)?)?;
    data.set_item("total_orders", PyList::new(py, &total_orders)?)?;
    data.set_item("total_spent", PyList::new(py, &total_spent)?)?;
    data.set_item("avg_order_value", PyList::new(py, &avg_values)?)?;
    data.set_item("rfm_recency", PyList::new(py, &recency)?)?;
    data.set_item("rfm_frequency", PyList::new(py, &frequency)?)?;
    data.set_item("rfm_monetary", PyList::new(py, &monetary)?)?;
    data.set_item("rfm_score", PyList::new(py, &scores)?)?;
    data.set_item("rfm_segment", PyList::new(py, &segments)?)?;

    let df = pandas.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create polars DataFrame from Customer rows
fn create_customers_polars(py: Python<'_>, rows: &[Customer]) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let data = PyDict::new(py);

    let ids: Vec<&str> = rows.iter().map(|r| r.customer_id.as_str()).collect();
    let emails: Vec<&str> = rows.iter().map(|r| r.email.as_str()).collect();
    let first_orders: Vec<Option<&str>> =
        rows.iter().map(|r| r.first_order_date.as_deref()).collect();
    let last_orders: Vec<Option<&str>> =
        rows.iter().map(|r| r.last_order_date.as_deref()).collect();
    let total_orders: Vec<u32> = rows.iter().map(|r| r.total_orders).collect();
    let total_spent: Vec<f64> = rows.iter().map(|r| r.total_spent).collect();
    let avg_values: Vec<f64> = rows.iter().map(|r| r.avg_order_value).collect();
    let recency: Vec<u32> = rows.iter().map(|r| r.rfm_recency).collect();
    let frequency: Vec<u32> = rows.iter().map(|r| r.rfm_frequency).collect();
    let monetary: Vec<f64> = rows.iter().map(|r| r.rfm_monetary).collect();
    let scores: Vec<&str> = rows.iter().map(|r| r.rfm_score.as_str()).collect();
    let segments: Vec<&str> = rows.iter().map(|r| r.rfm_segment.as_str()).collect();

    data.set_item("customer_id", PyList::new(py, &ids)?)?;
    data.set_item("email", PyList::new(py, &emails)?)?;
    data.set_item("first_order_date", PyList::new(py, &first_orders)?)?;
    data.set_item("last_order_date", PyList::new(py, &last_orders)?)?;
    data.set_item("total_orders", PyList::new(py, &total_orders)?)?;
    data.set_item("total_spent", PyList::new(py, &total_spent)?)?;
    data.set_item("avg_order_value", PyList::new(py, &avg_values)?)?;
    data.set_item("rfm_recency", PyList::new(py, &recency)?)?;
    data.set_item("rfm_frequency", PyList::new(py, &frequency)?)?;
    data.set_item("rfm_monetary", PyList::new(py, &monetary)?)?;
    data.set_item("rfm_score", PyList::new(py, &scores)?)?;
    data.set_item("rfm_segment", PyList::new(py, &segments)?)?;

    let df = polars.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create dict from Customer rows
fn create_customers_dict(py: Python<'_>, rows: &[Customer]) -> PyResult<Py<PyAny>> {
    let data = PyDict::new(py);

    let ids: Vec<&str> = rows.iter().map(|r| r.customer_id.as_str()).collect();
    let emails: Vec<&str> = rows.iter().map(|r| r.email.as_str()).collect();
    let first_orders: Vec<Option<&str>> =
        rows.iter().map(|r| r.first_order_date.as_deref()).collect();
    let last_orders: Vec<Option<&str>> =
        rows.iter().map(|r| r.last_order_date.as_deref()).collect();
    let total_orders: Vec<u32> = rows.iter().map(|r| r.total_orders).collect();
    let total_spent: Vec<f64> = rows.iter().map(|r| r.total_spent).collect();
    let avg_values: Vec<f64> = rows.iter().map(|r| r.avg_order_value).collect();
    let recency: Vec<u32> = rows.iter().map(|r| r.rfm_recency).collect();
    let frequency: Vec<u32> = rows.iter().map(|r| r.rfm_frequency).collect();
    let monetary: Vec<f64> = rows.iter().map(|r| r.rfm_monetary).collect();
    let scores: Vec<&str> = rows.iter().map(|r| r.rfm_score.as_str()).collect();
    let segments: Vec<&str> = rows.iter().map(|r| r.rfm_segment.as_str()).collect();

    data.set_item("customer_id", PyList::new(py, &ids)?)?;
    data.set_item("email", PyList::new(py, &emails)?)?;
    data.set_item("first_order_date", PyList::new(py, &first_orders)?)?;
    data.set_item("last_order_date", PyList::new(py, &last_orders)?)?;
    data.set_item("total_orders", PyList::new(py, &total_orders)?)?;
    data.set_item("total_spent", PyList::new(py, &total_spent)?)?;
    data.set_item("avg_order_value", PyList::new(py, &avg_values)?)?;
    data.set_item("rfm_recency", PyList::new(py, &recency)?)?;
    data.set_item("rfm_frequency", PyList::new(py, &frequency)?)?;
    data.set_item("rfm_monetary", PyList::new(py, &monetary)?)?;
    data.set_item("rfm_score", PyList::new(py, &scores)?)?;
    data.set_item("rfm_segment", PyList::new(py, &segments)?)?;

    Ok(data.into())
}

/// Create pandas DataFrame from CartEvent rows
fn create_cart_events_pandas(py: Python<'_>, rows: &[CartEvent]) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let data = PyDict::new(py);

    let ids: Vec<&str> = rows.iter().map(|r| r.event_id.as_str()).collect();
    let session_ids: Vec<&str> = rows.iter().map(|r| r.session_id.as_str()).collect();
    let user_ids: Vec<&str> = rows.iter().map(|r| r.user_id.as_str()).collect();
    let timestamps: Vec<&str> = rows.iter().map(|r| r.timestamp.as_str()).collect();
    let types: Vec<&str> = rows.iter().map(|r| r.event_type.as_str()).collect();
    let product_ids: Vec<&str> = rows.iter().map(|r| r.product_id.as_str()).collect();
    let quantities: Vec<u32> = rows.iter().map(|r| r.quantity).collect();
    let unit_prices: Vec<f64> = rows.iter().map(|r| r.unit_price).collect();
    let total_prices: Vec<f64> = rows.iter().map(|r| r.total_price).collect();

    data.set_item("event_id", PyList::new(py, &ids)?)?;
    data.set_item("session_id", PyList::new(py, &session_ids)?)?;
    data.set_item("user_id", PyList::new(py, &user_ids)?)?;
    data.set_item("timestamp", PyList::new(py, &timestamps)?)?;
    data.set_item("event_type", PyList::new(py, &types)?)?;
    data.set_item("product_id", PyList::new(py, &product_ids)?)?;
    data.set_item("quantity", PyList::new(py, &quantities)?)?;
    data.set_item("unit_price", PyList::new(py, &unit_prices)?)?;
    data.set_item("total_price", PyList::new(py, &total_prices)?)?;

    let df = pandas.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create polars DataFrame from CartEvent rows
fn create_cart_events_polars(py: Python<'_>, rows: &[CartEvent]) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let data = PyDict::new(py);

    let ids: Vec<&str> = rows.iter().map(|r| r.event_id.as_str()).collect();
    let session_ids: Vec<&str> = rows.iter().map(|r| r.session_id.as_str()).collect();
    let user_ids: Vec<&str> = rows.iter().map(|r| r.user_id.as_str()).collect();
    let timestamps: Vec<&str> = rows.iter().map(|r| r.timestamp.as_str()).collect();
    let types: Vec<&str> = rows.iter().map(|r| r.event_type.as_str()).collect();
    let product_ids: Vec<&str> = rows.iter().map(|r| r.product_id.as_str()).collect();
    let quantities: Vec<u32> = rows.iter().map(|r| r.quantity).collect();
    let unit_prices: Vec<f64> = rows.iter().map(|r| r.unit_price).collect();
    let total_prices: Vec<f64> = rows.iter().map(|r| r.total_price).collect();

    data.set_item("event_id", PyList::new(py, &ids)?)?;
    data.set_item("session_id", PyList::new(py, &session_ids)?)?;
    data.set_item("user_id", PyList::new(py, &user_ids)?)?;
    data.set_item("timestamp", PyList::new(py, &timestamps)?)?;
    data.set_item("event_type", PyList::new(py, &types)?)?;
    data.set_item("product_id", PyList::new(py, &product_ids)?)?;
    data.set_item("quantity", PyList::new(py, &quantities)?)?;
    data.set_item("unit_price", PyList::new(py, &unit_prices)?)?;
    data.set_item("total_price", PyList::new(py, &total_prices)?)?;

    let df = polars.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create dict from CartEvent rows
fn create_cart_events_dict(py: Python<'_>, rows: &[CartEvent]) -> PyResult<Py<PyAny>> {
    let data = PyDict::new(py);

    let ids: Vec<&str> = rows.iter().map(|r| r.event_id.as_str()).collect();
    let session_ids: Vec<&str> = rows.iter().map(|r| r.session_id.as_str()).collect();
    let user_ids: Vec<&str> = rows.iter().map(|r| r.user_id.as_str()).collect();
    let timestamps: Vec<&str> = rows.iter().map(|r| r.timestamp.as_str()).collect();
    let types: Vec<&str> = rows.iter().map(|r| r.event_type.as_str()).collect();
    let product_ids: Vec<&str> = rows.iter().map(|r| r.product_id.as_str()).collect();
    let quantities: Vec<u32> = rows.iter().map(|r| r.quantity).collect();
    let unit_prices: Vec<f64> = rows.iter().map(|r| r.unit_price).collect();
    let total_prices: Vec<f64> = rows.iter().map(|r| r.total_price).collect();

    data.set_item("event_id", PyList::new(py, &ids)?)?;
    data.set_item("session_id", PyList::new(py, &session_ids)?)?;
    data.set_item("user_id", PyList::new(py, &user_ids)?)?;
    data.set_item("timestamp", PyList::new(py, &timestamps)?)?;
    data.set_item("event_type", PyList::new(py, &types)?)?;
    data.set_item("product_id", PyList::new(py, &product_ids)?)?;
    data.set_item("quantity", PyList::new(py, &quantities)?)?;
    data.set_item("unit_price", PyList::new(py, &unit_prices)?)?;
    data.set_item("total_price", PyList::new(py, &total_prices)?)?;

    Ok(data.into())
}

// =============================================================================
// Config Parsing
// =============================================================================

fn parse_session_config(dict: &Bound<'_, PyDict>) -> SessionConfig {
    let mut config = SessionConfig::default();
    if let Some(v) = dict.get_item("avg_pages_per_session").ok().flatten() {
        config.avg_pages_per_session = v.extract().unwrap_or(config.avg_pages_per_session);
    }
    if let Some(v) = dict.get_item("cart_add_probability").ok().flatten() {
        config.cart_add_probability = v.extract().unwrap_or(config.cart_add_probability);
    }
    if let Some(v) = dict.get_item("checkout_start_probability").ok().flatten() {
        config.checkout_start_probability =
            v.extract().unwrap_or(config.checkout_start_probability);
    }
    if let Some(v) = dict
        .get_item("purchase_completion_probability")
        .ok()
        .flatten()
    {
        config.purchase_completion_probability = v
            .extract()
            .unwrap_or(config.purchase_completion_probability);
    }
    if let Some(v) = dict.get_item("avg_session_duration_seconds").ok().flatten() {
        config.avg_session_duration_seconds =
            v.extract().unwrap_or(config.avg_session_duration_seconds);
    }
    if let Some(v) = dict.get_item("enable_bounces").ok().flatten() {
        config.enable_bounces = v.extract().unwrap_or(config.enable_bounces);
    }
    if let Some(v) = dict.get_item("bounce_rate").ok().flatten() {
        config.bounce_rate = v.extract().unwrap_or(config.bounce_rate);
    }
    config
}

fn parse_cart_config(dict: &Bound<'_, PyDict>) -> CartConfig {
    let mut config = CartConfig::default();
    if let Some(v) = dict.get_item("avg_items_per_cart").ok().flatten() {
        config.avg_items_per_cart = v.extract().unwrap_or(config.avg_items_per_cart);
    }
    if let Some(v) = dict.get_item("remove_probability").ok().flatten() {
        config.remove_probability = v.extract().unwrap_or(config.remove_probability);
    }
    if let Some(v) = dict.get_item("quantity_update_probability").ok().flatten() {
        config.quantity_update_probability =
            v.extract().unwrap_or(config.quantity_update_probability);
    }
    if let Some(v) = dict.get_item("max_items").ok().flatten() {
        config.max_items = v.extract().unwrap_or(config.max_items);
    }
    if let Some(v) = dict.get_item("enable_abandonment").ok().flatten() {
        config.enable_abandonment = v.extract().unwrap_or(config.enable_abandonment);
    }
    if let Some(v) = dict.get_item("abandonment_rate").ok().flatten() {
        config.abandonment_rate = v.extract().unwrap_or(config.abandonment_rate);
    }
    config
}

fn parse_catalog_config(dict: &Bound<'_, PyDict>) -> CatalogConfig {
    let mut config = CatalogConfig::default();
    if let Some(v) = dict.get_item("num_products").ok().flatten() {
        config.num_products = v.extract().unwrap_or(config.num_products);
    }
    if let Some(v) = dict.get_item("min_price").ok().flatten() {
        config.min_price = v.extract().unwrap_or(config.min_price);
    }
    if let Some(v) = dict.get_item("max_price").ok().flatten() {
        config.max_price = v.extract().unwrap_or(config.max_price);
    }
    if let Some(v) = dict.get_item("lognormal_prices").ok().flatten() {
        config.lognormal_prices = v.extract().unwrap_or(config.lognormal_prices);
    }
    if let Some(v) = dict.get_item("categories").ok().flatten() {
        config.categories = v.extract().unwrap_or(config.categories);
    }
    config
}

fn parse_rfm_config(dict: &Bound<'_, PyDict>) -> RfmConfig {
    let mut config = RfmConfig::default();
    if let Some(v) = dict.get_item("enable").ok().flatten() {
        config.enable = v.extract().unwrap_or(config.enable);
    }
    if let Some(v) = dict.get_item("recency_window_days").ok().flatten() {
        config.recency_window_days = v.extract().unwrap_or(config.recency_window_days);
    }
    if let Some(v) = dict.get_item("num_buckets").ok().flatten() {
        config.num_buckets = v.extract().unwrap_or(config.num_buckets);
    }
    if let Some(v) = dict.get_item("pareto_shape").ok().flatten() {
        config.pareto_shape = v.extract().unwrap_or(config.pareto_shape);
    }
    config
}

fn parse_funnel_config(dict: &Bound<'_, PyDict>) -> FunnelConfig {
    let mut config = FunnelConfig::default();
    if let Some(v) = dict.get_item("enable").ok().flatten() {
        config.enable = v.extract().unwrap_or(config.enable);
    }
    if let Some(v) = dict.get_item("stages").ok().flatten() {
        config.stages = v.extract().unwrap_or(config.stages);
    }
    if let Some(v) = dict.get_item("time_of_day_effects").ok().flatten() {
        config.time_of_day_effects = v.extract().unwrap_or(config.time_of_day_effects);
    }
    if let Some(v) = dict.get_item("day_of_week_effects").ok().flatten() {
        config.day_of_week_effects = v.extract().unwrap_or(config.day_of_week_effects);
    }
    config
}

fn parse_ecommerce_config(dict: &Bound<'_, PyDict>) -> EcommerceConfig {
    let mut config = EcommerceConfig::default();

    if let Some(v) = dict.get_item("sessions").ok().flatten() {
        config.sessions = v.extract().unwrap_or(config.sessions);
    }
    if let Some(v) = dict.get_item("customers").ok().flatten() {
        config.customers = v.extract().unwrap_or(config.customers);
    }
    if let Some(v) = dict.get_item("seed").ok().flatten() {
        config.seed = v.extract().ok();
    }
    if let Some(v) = dict.get_item("start_date").ok().flatten() {
        config.start_date = v.extract().ok();
    }
    if let Some(v) = dict.get_item("days").ok().flatten() {
        config.days = v.extract().unwrap_or(config.days);
    }
    if let Some(v) = dict.get_item("session").ok().flatten() {
        if let Ok(d) = v.downcast::<PyDict>() {
            config.session = parse_session_config(d);
        }
    }
    if let Some(v) = dict.get_item("cart").ok().flatten() {
        if let Ok(d) = v.downcast::<PyDict>() {
            config.cart = parse_cart_config(d);
        }
    }
    if let Some(v) = dict.get_item("catalog").ok().flatten() {
        if let Ok(d) = v.downcast::<PyDict>() {
            config.catalog = parse_catalog_config(d);
        }
    }
    if let Some(v) = dict.get_item("rfm").ok().flatten() {
        if let Ok(d) = v.downcast::<PyDict>() {
            config.rfm = parse_rfm_config(d);
        }
    }
    if let Some(v) = dict.get_item("funnel").ok().flatten() {
        if let Ok(d) = v.downcast::<PyDict>() {
            config.funnel = parse_funnel_config(d);
        }
    }

    config
}

// =============================================================================
// PyO3 Functions
// =============================================================================

/// Generate e-commerce sessions
///
/// Args:
///     count: Number of sessions to generate
///     seed: Optional random seed for reproducibility
///     output: Output format ("pandas", "polars", or "dict")
///
/// Returns:
///     DataFrame or dict with session data
#[pyfunction]
#[pyo3(signature = (count, seed = None, output = "pandas"))]
pub fn ecommerce_sessions(
    py: Python<'_>,
    count: usize,
    seed: Option<u64>,
    output: &str,
) -> PyResult<Py<PyAny>> {
    let config = EcommerceConfig {
        sessions: count,
        seed,
        ..Default::default()
    };
    let sessions = generate_sessions(&config);

    match output {
        "polars" => create_sessions_polars(py, &sessions),
        "dict" => create_sessions_dict(py, &sessions),
        _ => create_sessions_pandas(py, &sessions),
    }
}

/// Generate e-commerce product catalog
///
/// Args:
///     count: Number of products to generate
///     seed: Optional random seed for reproducibility
///     output: Output format ("pandas", "polars", or "dict")
///
/// Returns:
///     DataFrame or dict with product data
#[pyfunction]
#[pyo3(signature = (count, seed = None, output = "pandas"))]
pub fn ecommerce_products(
    py: Python<'_>,
    count: usize,
    seed: Option<u64>,
    output: &str,
) -> PyResult<Py<PyAny>> {
    let config = EcommerceConfig {
        seed,
        catalog: CatalogConfig {
            num_products: count,
            ..Default::default()
        },
        ..Default::default()
    };
    let products = generate_catalog(&config);

    match output {
        "polars" => create_products_polars(py, &products),
        "dict" => create_products_dict(py, &products),
        _ => create_products_pandas(py, &products),
    }
}

/// Generate complete e-commerce dataset
///
/// Args:
///     config: EcommerceConfig dict with generation parameters
///     output: Output format ("pandas", "polars", or "dict")
///
/// Returns:
///     Dict with DataFrames for products, sessions, cart_events, orders, customers
#[pyfunction]
#[pyo3(signature = (config = None, output = "pandas"))]
pub fn ecommerce_data(
    py: Python<'_>,
    config: Option<&Bound<'_, PyDict>>,
    output: &str,
) -> PyResult<Py<PyAny>> {
    let cfg = match config {
        Some(d) => parse_ecommerce_config(d),
        None => EcommerceConfig::default(),
    };

    let data = ecommerce(&cfg);
    let result = PyDict::new(py);

    match output {
        "polars" => {
            result.set_item("products", create_products_polars(py, &data.products)?)?;
            result.set_item("sessions", create_sessions_polars(py, &data.sessions)?)?;
            result.set_item(
                "cart_events",
                create_cart_events_polars(py, &data.cart_events)?,
            )?;
            result.set_item("orders", create_orders_polars(py, &data.orders)?)?;
            result.set_item("customers", create_customers_polars(py, &data.customers)?)?;
        }
        "dict" => {
            result.set_item("products", create_products_dict(py, &data.products)?)?;
            result.set_item("sessions", create_sessions_dict(py, &data.sessions)?)?;
            result.set_item(
                "cart_events",
                create_cart_events_dict(py, &data.cart_events)?,
            )?;
            result.set_item("orders", create_orders_dict(py, &data.orders)?)?;
            result.set_item("customers", create_customers_dict(py, &data.customers)?)?;
        }
        _ => {
            result.set_item("products", create_products_pandas(py, &data.products)?)?;
            result.set_item("sessions", create_sessions_pandas(py, &data.sessions)?)?;
            result.set_item(
                "cart_events",
                create_cart_events_pandas(py, &data.cart_events)?,
            )?;
            result.set_item("orders", create_orders_pandas(py, &data.orders)?)?;
            result.set_item("customers", create_customers_pandas(py, &data.customers)?)?;
        }
    }

    Ok(result.into())
}

/// Register ecommerce module functions
pub fn register_ecommerce(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ecommerce_sessions, m)?)?;
    m.add_function(wrap_pyfunction!(ecommerce_products, m)?)?;
    m.add_function(wrap_pyfunction!(ecommerce_data, m)?)?;
    Ok(())
}
