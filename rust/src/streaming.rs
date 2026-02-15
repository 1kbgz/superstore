//! Streaming data generation for memory-efficient processing of large datasets.
//!
//! This module provides iterator-based generators that yield data in chunks,
//! allowing processing of arbitrarily large datasets without loading everything
//! into memory at once.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::general::{EmployeeRow, SuperstoreRow};
use crate::utils::{US_SECTORS, US_SECTORS_MAP};

use chrono::{Datelike, NaiveDate, Utc};
use fake::faker::address::en::{CityName, StateName, ZipCode};
use fake::faker::internet::en::SafeEmail;
use fake::faker::name::en::{FirstName, LastName};
use fake::faker::phone_number::en::PhoneNumber;
use fake::Fake;

const SHIP_MODES: [&str; 3] = ["First Class", "Standard Class", "Second Class"];
const SEGMENTS: [&str; 4] = ["A", "B", "C", "D"];
const PREFIXES: [&str; 6] = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Rev."];
const SUFFIXES: [&str; 4] = ["Jr.", "Sr.", "III", "IV"];

fn generate_ein<R: Rng>(rng: &mut R) -> String {
    format!(
        "{:02}-{:07}",
        rng.gen_range(10..99),
        rng.gen_range(1000000..9999999)
    )
}

fn generate_license_plate<R: Rng>(rng: &mut R) -> String {
    let letters: String = (0..3)
        .map(|_| (b'A' + rng.gen_range(0..26)) as char)
        .collect();
    let numbers: u16 = rng.gen_range(100..1000);
    format!("{}{}", letters, numbers)
}

fn generate_bban<R: Rng>(rng: &mut R) -> String {
    (0..18)
        .map(|_| (b'0' + rng.gen_range(0..10)) as char)
        .collect()
}

fn generate_ssn<R: Rng>(rng: &mut R) -> String {
    format!(
        "{:03}-{:02}-{:04}",
        rng.gen_range(100..999),
        rng.gen_range(10..99),
        rng.gen_range(1000..9999)
    )
}

fn generate_street_address<R: Rng>(rng: &mut R) -> String {
    let number: u32 = rng.gen_range(1..9999);
    let street_names = [
        "Main St",
        "Oak Ave",
        "Elm St",
        "Park Rd",
        "Cedar Ln",
        "Maple Dr",
        "Pine St",
        "Washington Ave",
        "Lake View Dr",
        "Hill St",
    ];
    format!("{} {}", number, street_names.choose(rng).unwrap())
}

fn random_date_this_year<R: Rng>(rng: &mut R) -> NaiveDate {
    let year = Utc::now().naive_utc().date().year();
    let day_of_year = rng.gen_range(1..=365);
    NaiveDate::from_yo_opt(year, day_of_year)
        .unwrap_or_else(|| NaiveDate::from_ymd_opt(year, 1, 1).unwrap())
}

fn random_date_between<R: Rng>(rng: &mut R, start: NaiveDate) -> NaiveDate {
    let today = Utc::now().naive_utc().date();
    if start >= today {
        return today;
    }
    let days_between = (today - start).num_days() as u32;
    if days_between == 0 {
        return start;
    }
    let random_days = rng.gen_range(0..=days_between);
    start + chrono::Duration::days(random_days as i64)
}

fn random_date_30_years<R: Rng>(rng: &mut R) -> NaiveDate {
    let today = Utc::now().naive_utc().date();
    let thirty_years_ago = today - chrono::Duration::days(30 * 365);
    let days_range = (today - thirty_years_ago).num_days() as u32;
    let random_days = rng.gen_range(0..=days_range);
    thirty_years_ago + chrono::Duration::days(random_days as i64)
}

fn random_date_of_birth<R: Rng>(rng: &mut R) -> NaiveDate {
    let today = Utc::now().naive_utc().date();
    let min_age = 18;
    let max_age = 70;
    let min_date = today - chrono::Duration::days(max_age * 365);
    let max_date = today - chrono::Duration::days(min_age * 365);
    let days_range = (max_date - min_date).num_days() as u32;
    let random_days = rng.gen_range(0..=days_range);
    min_date + chrono::Duration::days(random_days as i64)
}

/// Iterator that generates superstore rows in chunks.
///
/// This is memory-efficient for large datasets as it only holds one chunk
/// in memory at a time.
pub struct SuperstoreIterator {
    rng: StdRng,
    total_count: usize,
    generated: usize,
    chunk_size: usize,
    sectors: Vec<&'static str>,
}

impl SuperstoreIterator {
    /// Create a new streaming superstore generator.
    ///
    /// # Arguments
    /// * `total_count` - Total number of rows to generate
    /// * `chunk_size` - Number of rows per chunk (default: 1000)
    /// * `seed` - Optional seed for reproducibility
    pub fn new(total_count: usize, chunk_size: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self {
            rng,
            total_count,
            generated: 0,
            chunk_size,
            sectors: US_SECTORS.clone(),
        }
    }
}

impl Iterator for SuperstoreIterator {
    type Item = Vec<SuperstoreRow>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.generated >= self.total_count {
            return None;
        }

        let remaining = self.total_count - self.generated;
        let chunk_len = remaining.min(self.chunk_size);
        let mut chunk = Vec::with_capacity(chunk_len);

        for i in 0..chunk_len {
            let row_id = (self.generated + i) as i32;
            let order_date = random_date_this_year(&mut self.rng);
            let ship_date = random_date_between(&mut self.rng, order_date);

            let sector = *self.sectors.choose(&mut self.rng).unwrap();
            let industries = US_SECTORS_MAP.get(sector).unwrap();
            let industry = *industries.choose(&mut self.rng).unwrap();

            let row = SuperstoreRow {
                row_id,
                order_id: generate_ein(&mut self.rng),
                order_date: order_date.format("%Y-%m-%d").to_string(),
                ship_date: ship_date.format("%Y-%m-%d").to_string(),
                ship_mode: SHIP_MODES.choose(&mut self.rng).unwrap().to_string(),
                customer_id: generate_license_plate(&mut self.rng),
                segment: SEGMENTS.choose(&mut self.rng).unwrap().to_string(),
                country: "US".to_string(),
                city: CityName().fake_with_rng(&mut self.rng),
                state: StateName().fake_with_rng(&mut self.rng),
                postal_code: ZipCode().fake_with_rng(&mut self.rng),
                region: format!("Region {}", self.rng.gen_range(0..5)),
                product_id: generate_bban(&mut self.rng),
                category: sector.to_string(),
                sub_category: industry.to_string(),
                item_status: "Regular".to_string(),
                item_price: (self.rng.gen_range(1..=100) as f64) * 10.0 + 0.99,
                sales: self.rng.gen_range(1..=100) * 100,
                quantity: self.rng.gen_range(1..=100) * 10,
                discount: (self.rng.gen::<f64>() * 100.0 * 100.0).round() / 100.0,
                profit: (self.rng.gen::<f64>() * 1000.0 * 100.0).round() / 100.0,
                // Priority 4 fields (not enabled in streaming simple mode)
                bundle_id: None,
                payment_method: None,
                is_fraud: None,
                processing_fee: None,
                backorder_days: None,
                stock_status: None,
            };
            chunk.push(row);
        }

        self.generated += chunk_len;
        Some(chunk)
    }
}

/// Iterator that generates employee rows in chunks.
pub struct EmployeeIterator {
    rng: StdRng,
    total_count: usize,
    generated: usize,
    chunk_size: usize,
}

impl EmployeeIterator {
    /// Create a new streaming employee generator.
    ///
    /// # Arguments
    /// * `total_count` - Total number of rows to generate
    /// * `chunk_size` - Number of rows per chunk (default: 1000)
    /// * `seed` - Optional seed for reproducibility
    pub fn new(total_count: usize, chunk_size: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self {
            rng,
            total_count,
            generated: 0,
            chunk_size,
        }
    }
}

impl Iterator for EmployeeIterator {
    type Item = Vec<EmployeeRow>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.generated >= self.total_count {
            return None;
        }

        let remaining = self.total_count - self.generated;
        let chunk_len = remaining.min(self.chunk_size);
        let mut chunk = Vec::with_capacity(chunk_len);

        for i in 0..chunk_len {
            let row_id = (self.generated + i) as i32;
            let row = EmployeeRow {
                row_id,
                employee_id: generate_ein(&mut self.rng),
                first_name: FirstName().fake_with_rng(&mut self.rng),
                surname: LastName().fake_with_rng(&mut self.rng),
                prefix: PREFIXES.choose(&mut self.rng).unwrap().to_string(),
                suffix: SUFFIXES.choose(&mut self.rng).unwrap().to_string(),
                phone_number: PhoneNumber().fake_with_rng(&mut self.rng),
                email: SafeEmail().fake_with_rng(&mut self.rng),
                ssn: generate_ssn(&mut self.rng),
                street: generate_street_address(&mut self.rng),
                city: CityName().fake_with_rng(&mut self.rng),
                postal_code: ZipCode().fake_with_rng(&mut self.rng),
                region: format!("Region {}", self.rng.gen_range(0..5)),
                state: StateName().fake_with_rng(&mut self.rng),
                country: "US".to_string(),
                start_date: random_date_30_years(&mut self.rng),
                date_of_birth: random_date_of_birth(&mut self.rng),
            };
            chunk.push(row);
        }

        self.generated += chunk_len;
        Some(chunk)
    }
}

/// Create a streaming superstore generator.
///
/// # Example
/// ```
/// use superstore::streaming::superstore_stream;
///
/// // Generate 1 million rows in chunks of 10,000
/// for chunk in superstore_stream(1_000_000, 10_000, Some(42)) {
///     // Process each chunk
///     println!("Processing {} rows", chunk.len());
/// }
/// ```
pub fn superstore_stream(
    total_count: usize,
    chunk_size: usize,
    seed: Option<u64>,
) -> SuperstoreIterator {
    SuperstoreIterator::new(total_count, chunk_size, seed)
}

/// Create a streaming employee generator.
///
/// # Example
/// ```
/// use superstore::streaming::employees_stream;
///
/// // Generate 1 million employees in chunks of 10,000
/// for chunk in employees_stream(1_000_000, 10_000, Some(42)) {
///     // Process each chunk
///     println!("Processing {} employees", chunk.len());
/// }
/// ```
pub fn employees_stream(
    total_count: usize,
    chunk_size: usize,
    seed: Option<u64>,
) -> EmployeeIterator {
    EmployeeIterator::new(total_count, chunk_size, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_superstore_stream_basic() {
        let chunks: Vec<_> = superstore_stream(100, 30, Some(42)).collect();
        assert_eq!(chunks.len(), 4); // 30 + 30 + 30 + 10
        assert_eq!(chunks[0].len(), 30);
        assert_eq!(chunks[1].len(), 30);
        assert_eq!(chunks[2].len(), 30);
        assert_eq!(chunks[3].len(), 10);
    }

    #[test]
    fn test_superstore_stream_reproducible() {
        let chunks1: Vec<_> = superstore_stream(100, 50, Some(42)).collect();
        let chunks2: Vec<_> = superstore_stream(100, 50, Some(42)).collect();

        assert_eq!(chunks1.len(), chunks2.len());
        for (c1, c2) in chunks1.iter().zip(chunks2.iter()) {
            for (r1, r2) in c1.iter().zip(c2.iter()) {
                assert_eq!(r1.order_id, r2.order_id);
                assert_eq!(r1.city, r2.city);
            }
        }
    }

    #[test]
    fn test_employees_stream_basic() {
        let chunks: Vec<_> = employees_stream(100, 30, Some(42)).collect();
        assert_eq!(chunks.len(), 4);
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_employees_stream_reproducible() {
        let chunks1: Vec<_> = employees_stream(50, 20, Some(123)).collect();
        let chunks2: Vec<_> = employees_stream(50, 20, Some(123)).collect();

        for (c1, c2) in chunks1.iter().zip(chunks2.iter()) {
            for (r1, r2) in c1.iter().zip(c2.iter()) {
                assert_eq!(r1.employee_id, r2.employee_id);
                assert_eq!(r1.first_name, r2.first_name);
            }
        }
    }

    #[test]
    fn test_stream_total_count() {
        // Verify total row count matches expected
        let total: usize = superstore_stream(1000, 100, None).map(|c| c.len()).sum();
        assert_eq!(total, 1000);
    }
}
