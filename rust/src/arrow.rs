//! Arrow memory format support for zero-copy interop with pandas/polars.
//!
//! This module provides functions to convert generated data directly to Apache Arrow
//! RecordBatches, enabling efficient memory sharing with Python DataFrames.

use arrow::array::{ArrayRef, Float64Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use crate::general::{employees, superstore, EmployeeRow, SuperstoreRow};

/// Create the Arrow schema for superstore data
pub fn superstore_schema() -> Schema {
    Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("order_id", DataType::Utf8, false),
        Field::new("order_date", DataType::Utf8, false),
        Field::new("ship_date", DataType::Utf8, false),
        Field::new("ship_mode", DataType::Utf8, false),
        Field::new("customer_id", DataType::Utf8, false),
        Field::new("segment", DataType::Utf8, false),
        Field::new("country", DataType::Utf8, false),
        Field::new("city", DataType::Utf8, false),
        Field::new("state", DataType::Utf8, false),
        Field::new("postal_code", DataType::Utf8, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("product_id", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("sub_category", DataType::Utf8, false),
        Field::new("sales", DataType::Int32, false),
        Field::new("quantity", DataType::Int32, false),
        Field::new("discount", DataType::Float64, false),
        Field::new("profit", DataType::Float64, false),
    ])
}

/// Create the Arrow schema for employee data
pub fn employee_schema() -> Schema {
    Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("employee_id", DataType::Utf8, false),
        Field::new("first_name", DataType::Utf8, false),
        Field::new("surname", DataType::Utf8, false),
        Field::new("prefix", DataType::Utf8, false),
        Field::new("suffix", DataType::Utf8, false),
        Field::new("phone_number", DataType::Utf8, false),
        Field::new("email", DataType::Utf8, false),
        Field::new("ssn", DataType::Utf8, false),
        Field::new("street", DataType::Utf8, false),
        Field::new("city", DataType::Utf8, false),
        Field::new("postal_code", DataType::Utf8, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("state", DataType::Utf8, false),
        Field::new("country", DataType::Utf8, false),
        Field::new("start_date", DataType::Utf8, false),
        Field::new("date_of_birth", DataType::Utf8, false),
    ])
}

/// Convert superstore rows to an Arrow RecordBatch
pub fn superstore_to_arrow(
    rows: &[SuperstoreRow],
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let schema = Arc::new(superstore_schema());

    let row_id: ArrayRef = Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.row_id)));
    let order_id: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.order_id.as_str()),
    ));
    let order_date: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.order_date.as_str()),
    ));
    let ship_date: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.ship_date.as_str()),
    ));
    let ship_mode: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.ship_mode.as_str()),
    ));
    let customer_id: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.customer_id.as_str()),
    ));
    let segment: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.segment.as_str()),
    ));
    let country: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.country.as_str()),
    ));
    let city: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.city.as_str()),
    ));
    let state: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.state.as_str()),
    ));
    let postal_code: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.postal_code.as_str()),
    ));
    let region: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.region.as_str()),
    ));
    let product_id: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.product_id.as_str()),
    ));
    let category: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.category.as_str()),
    ));
    let sub_category: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.sub_category.as_str()),
    ));
    let sales: ArrayRef = Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.sales)));
    let quantity: ArrayRef = Arc::new(Int32Array::from_iter_values(
        rows.iter().map(|r| r.quantity),
    ));
    let discount: ArrayRef = Arc::new(Float64Array::from_iter_values(
        rows.iter().map(|r| r.discount),
    ));
    let profit: ArrayRef = Arc::new(Float64Array::from_iter_values(
        rows.iter().map(|r| r.profit),
    ));

    RecordBatch::try_new(
        schema,
        vec![
            row_id,
            order_id,
            order_date,
            ship_date,
            ship_mode,
            customer_id,
            segment,
            country,
            city,
            state,
            postal_code,
            region,
            product_id,
            category,
            sub_category,
            sales,
            quantity,
            discount,
            profit,
        ],
    )
}

/// Convert employee rows to an Arrow RecordBatch
pub fn employees_to_arrow(rows: &[EmployeeRow]) -> Result<RecordBatch, arrow::error::ArrowError> {
    let schema = Arc::new(employee_schema());

    let row_id: ArrayRef = Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.row_id)));
    let employee_id: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.employee_id.as_str()),
    ));
    let first_name: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.first_name.as_str()),
    ));
    let surname: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.surname.as_str()),
    ));
    let prefix: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.prefix.as_str()),
    ));
    let suffix: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.suffix.as_str()),
    ));
    let phone_number: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.phone_number.as_str()),
    ));
    let email: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.email.as_str()),
    ));
    let ssn: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.ssn.as_str()),
    ));
    let street: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.street.as_str()),
    ));
    let city: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.city.as_str()),
    ));
    let postal_code: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.postal_code.as_str()),
    ));
    let region: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.region.as_str()),
    ));
    let state: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.state.as_str()),
    ));
    let country: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.country.as_str()),
    ));
    let start_date: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.start_date.to_string()),
    ));
    let date_of_birth: ArrayRef = Arc::new(StringArray::from_iter_values(
        rows.iter().map(|r| r.date_of_birth.to_string()),
    ));

    RecordBatch::try_new(
        schema,
        vec![
            row_id,
            employee_id,
            first_name,
            surname,
            prefix,
            suffix,
            phone_number,
            email,
            ssn,
            street,
            city,
            postal_code,
            region,
            state,
            country,
            start_date,
            date_of_birth,
        ],
    )
}

/// Generate superstore data directly as an Arrow RecordBatch
pub fn superstore_arrow(
    count: usize,
    seed: Option<u64>,
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let rows = superstore(count, seed, None);
    superstore_to_arrow(&rows)
}

/// Generate employee data directly as an Arrow RecordBatch
pub fn employees_arrow(
    count: usize,
    seed: Option<u64>,
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let rows = employees(count, seed, None);
    employees_to_arrow(&rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_superstore_arrow() {
        let batch = superstore_arrow(100, Some(42)).unwrap();
        assert_eq!(batch.num_rows(), 100);
        assert_eq!(batch.num_columns(), 19);
    }

    #[test]
    fn test_employees_arrow() {
        let batch = employees_arrow(100, Some(42)).unwrap();
        assert_eq!(batch.num_rows(), 100);
        assert_eq!(batch.num_columns(), 17);
    }

    #[test]
    fn test_superstore_schema() {
        let schema = superstore_schema();
        assert_eq!(schema.fields().len(), 19);
        assert_eq!(schema.field(0).name(), "row_id");
        assert_eq!(schema.field(0).data_type(), &DataType::Int32);
    }

    #[test]
    fn test_employee_schema() {
        let schema = employee_schema();
        assert_eq!(schema.fields().len(), 17);
        assert_eq!(schema.field(0).name(), "row_id");
    }

    #[test]
    fn test_arrow_reproducibility() {
        let batch1 = superstore_arrow(10, Some(12345)).unwrap();
        let batch2 = superstore_arrow(10, Some(12345)).unwrap();

        // Compare row_id columns
        let col1 = batch1
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let col2 = batch2
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..10 {
            assert_eq!(col1.value(i), col2.value(i));
        }
    }
}
