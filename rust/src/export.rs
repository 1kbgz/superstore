//! Direct file export support for Parquet and CSV formats.
//!
//! This module provides functions to write generated data directly to files
//! without going through a DataFrame intermediary.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::arrow::{employees_to_arrow, superstore_to_arrow};
use crate::general::{employees, superstore, EmployeeRow, SuperstoreRow};

/// Error type for export operations
#[derive(Debug)]
pub enum ExportError {
    Io(std::io::Error),
    Arrow(arrow::error::ArrowError),
    Parquet(parquet::errors::ParquetError),
}

impl From<std::io::Error> for ExportError {
    fn from(e: std::io::Error) -> Self {
        ExportError::Io(e)
    }
}

impl From<arrow::error::ArrowError> for ExportError {
    fn from(e: arrow::error::ArrowError) -> Self {
        ExportError::Arrow(e)
    }
}

impl From<parquet::errors::ParquetError> for ExportError {
    fn from(e: parquet::errors::ParquetError) -> Self {
        ExportError::Parquet(e)
    }
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportError::Io(e) => write!(f, "IO error: {}", e),
            ExportError::Arrow(e) => write!(f, "Arrow error: {}", e),
            ExportError::Parquet(e) => write!(f, "Parquet error: {}", e),
        }
    }
}

impl std::error::Error for ExportError {}

/// Compression type for Parquet files
#[derive(Clone, Copy, Debug, Default)]
pub enum ParquetCompression {
    None,
    #[default]
    Snappy,
    Zstd,
}

impl From<ParquetCompression> for Compression {
    fn from(c: ParquetCompression) -> Self {
        match c {
            ParquetCompression::None => Compression::UNCOMPRESSED,
            ParquetCompression::Snappy => Compression::SNAPPY,
            ParquetCompression::Zstd => Compression::ZSTD(Default::default()),
        }
    }
}

/// Write superstore data to a Parquet file
pub fn superstore_to_parquet<P: AsRef<Path>>(
    path: P,
    count: usize,
    seed: Option<u64>,
    compression: ParquetCompression,
) -> Result<usize, ExportError> {
    let rows = superstore(count, seed);
    let batch = superstore_to_arrow(&rows)?;

    let file = File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(compression.into())
        .build();

    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(rows.len())
}

/// Write employee data to a Parquet file
pub fn employees_to_parquet<P: AsRef<Path>>(
    path: P,
    count: usize,
    seed: Option<u64>,
    compression: ParquetCompression,
) -> Result<usize, ExportError> {
    let rows = employees(count, seed);
    let batch = employees_to_arrow(&rows)?;

    let file = File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(compression.into())
        .build();

    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(rows.len())
}

fn escape_csv_field(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

fn write_superstore_row<W: Write>(writer: &mut W, row: &SuperstoreRow) -> std::io::Result<()> {
    writeln!(
        writer,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        row.row_id,
        escape_csv_field(&row.order_id),
        escape_csv_field(&row.order_date),
        escape_csv_field(&row.ship_date),
        escape_csv_field(&row.ship_mode),
        escape_csv_field(&row.customer_id),
        escape_csv_field(&row.segment),
        escape_csv_field(&row.country),
        escape_csv_field(&row.city),
        escape_csv_field(&row.state),
        escape_csv_field(&row.postal_code),
        escape_csv_field(&row.region),
        escape_csv_field(&row.product_id),
        escape_csv_field(&row.category),
        escape_csv_field(&row.sub_category),
        row.sales,
        row.quantity,
        row.discount,
        row.profit
    )
}

fn write_employee_row<W: Write>(writer: &mut W, row: &EmployeeRow) -> std::io::Result<()> {
    writeln!(
        writer,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        row.row_id,
        escape_csv_field(&row.employee_id),
        escape_csv_field(&row.first_name),
        escape_csv_field(&row.surname),
        escape_csv_field(&row.prefix),
        escape_csv_field(&row.suffix),
        escape_csv_field(&row.phone_number),
        escape_csv_field(&row.email),
        escape_csv_field(&row.ssn),
        escape_csv_field(&row.street),
        escape_csv_field(&row.city),
        escape_csv_field(&row.postal_code),
        escape_csv_field(&row.region),
        escape_csv_field(&row.state),
        escape_csv_field(&row.country),
        row.start_date,
        row.date_of_birth
    )
}

/// Write superstore data to a CSV file
pub fn superstore_to_csv<P: AsRef<Path>>(
    path: P,
    count: usize,
    seed: Option<u64>,
) -> Result<usize, ExportError> {
    let rows = superstore(count, seed);

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(
        writer,
        "row_id,order_id,order_date,ship_date,ship_mode,customer_id,segment,country,city,state,postal_code,region,product_id,category,sub_category,sales,quantity,discount,profit"
    )?;

    // Write rows
    for row in &rows {
        write_superstore_row(&mut writer, row)?;
    }

    writer.flush()?;
    Ok(rows.len())
}

/// Write employee data to a CSV file
pub fn employees_to_csv<P: AsRef<Path>>(
    path: P,
    count: usize,
    seed: Option<u64>,
) -> Result<usize, ExportError> {
    let rows = employees(count, seed);

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(
        writer,
        "row_id,employee_id,first_name,surname,prefix,suffix,phone_number,email,ssn,street,city,postal_code,region,state,country,start_date,date_of_birth"
    )?;

    // Write rows
    for row in &rows {
        write_employee_row(&mut writer, row)?;
    }

    writer.flush()?;
    Ok(rows.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_superstore_to_csv() {
        let path = std::env::temp_dir().join("test_superstore.csv");
        let path_str = path.to_str().unwrap();
        let count = superstore_to_csv(path_str, 100, Some(42)).unwrap();
        assert_eq!(count, 100);

        let content = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 101); // header + 100 rows
        assert!(lines[0].starts_with("row_id,order_id"));

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_employees_to_csv() {
        let path = std::env::temp_dir().join("test_employees.csv");
        let path_str = path.to_str().unwrap();
        let count = employees_to_csv(path_str, 50, Some(42)).unwrap();
        assert_eq!(count, 50);

        let content = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 51); // header + 50 rows
        assert!(lines[0].starts_with("row_id,employee_id"));

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_superstore_to_parquet() {
        let path = std::env::temp_dir().join("test_superstore.parquet");
        let path_str = path.to_str().unwrap();
        let count =
            superstore_to_parquet(path_str, 100, Some(42), ParquetCompression::Snappy).unwrap();
        assert_eq!(count, 100);

        // Verify file exists and has content
        let metadata = fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_employees_to_parquet() {
        let path = std::env::temp_dir().join("test_employees.parquet");
        let path_str = path.to_str().unwrap();
        let count = employees_to_parquet(path_str, 50, Some(42), ParquetCompression::Zstd).unwrap();
        assert_eq!(count, 50);

        // Verify file exists and has content
        let metadata = fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_csv_escape() {
        assert_eq!(escape_csv_field("hello"), "hello");
        assert_eq!(escape_csv_field("hello,world"), "\"hello,world\"");
        assert_eq!(escape_csv_field("hello\"world"), "\"hello\"\"world\"");
        assert_eq!(escape_csv_field("hello\nworld"), "\"hello\nworld\"");
    }

    #[test]
    fn test_parquet_compression_variants() {
        let path_none = std::env::temp_dir().join("test_none.parquet");
        let path_snappy = std::env::temp_dir().join("test_snappy.parquet");
        let path_zstd = std::env::temp_dir().join("test_zstd.parquet");

        superstore_to_parquet(
            path_none.to_str().unwrap(),
            100,
            Some(42),
            ParquetCompression::None,
        )
        .unwrap();
        superstore_to_parquet(
            path_snappy.to_str().unwrap(),
            100,
            Some(42),
            ParquetCompression::Snappy,
        )
        .unwrap();
        superstore_to_parquet(
            path_zstd.to_str().unwrap(),
            100,
            Some(42),
            ParquetCompression::Zstd,
        )
        .unwrap();

        let size_none = fs::metadata(&path_none).unwrap().len();
        let size_snappy = fs::metadata(&path_snappy).unwrap().len();
        let size_zstd = fs::metadata(&path_zstd).unwrap().len();

        // Compressed files should generally be smaller than uncompressed
        assert!(size_snappy <= size_none || size_zstd <= size_none);

        fs::remove_file(&path_none).unwrap();
        fs::remove_file(&path_snappy).unwrap();
        fs::remove_file(&path_zstd).unwrap();
    }
}
