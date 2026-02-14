//! Parallel data generation using Rayon for multi-threaded processing.
//!
//! This module provides parallel versions of data generators that utilize
//! multiple CPU cores for faster generation of large datasets.

use rayon::prelude::*;

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

/// Generate superstore data in parallel using multiple threads.
///
/// This function divides the work across available CPU cores for faster
/// generation of large datasets. With a seed, results are reproducible
/// but row order may differ from the sequential version.
///
/// # Arguments
/// * `count` - Total number of rows to generate
/// * `seed` - Optional seed for reproducibility (per-thread seeds derived from this)
///
/// # Example
/// ```
/// use superstore::parallel::superstore_parallel;
///
/// // Generate 1 million rows using all CPU cores
/// let data = superstore_parallel(1_000_000, Some(42));
/// ```
pub fn superstore_parallel(count: usize, seed: Option<u64>) -> Vec<SuperstoreRow> {
    let num_threads = rayon::current_num_threads();
    let chunk_size = (count + num_threads - 1) / num_threads;

    let chunks: Vec<Vec<SuperstoreRow>> = (0..num_threads)
        .into_par_iter()
        .map(|thread_idx| {
            let start_idx = thread_idx * chunk_size;
            let end_idx = (start_idx + chunk_size).min(count);
            if start_idx >= count {
                return Vec::new();
            }

            // Create per-thread RNG with deterministic seed based on thread index
            let mut rng = match seed {
                Some(s) => StdRng::seed_from_u64(s.wrapping_add(thread_idx as u64)),
                None => StdRng::from_entropy(),
            };

            let sectors: Vec<&str> = US_SECTORS.clone();
            let mut chunk = Vec::with_capacity(end_idx - start_idx);

            for row_id in start_idx..end_idx {
                let order_date = random_date_this_year(&mut rng);
                let ship_date = random_date_between(&mut rng, order_date);

                let sector = *sectors.choose(&mut rng).unwrap();
                let industries = US_SECTORS_MAP.get(sector).unwrap();
                let industry = *industries.choose(&mut rng).unwrap();

                let row = SuperstoreRow {
                    row_id: row_id as i32,
                    order_id: generate_ein(&mut rng),
                    order_date: order_date.format("%Y-%m-%d").to_string(),
                    ship_date: ship_date.format("%Y-%m-%d").to_string(),
                    ship_mode: SHIP_MODES.choose(&mut rng).unwrap().to_string(),
                    customer_id: generate_license_plate(&mut rng),
                    segment: SEGMENTS.choose(&mut rng).unwrap().to_string(),
                    country: "US".to_string(),
                    city: CityName().fake_with_rng(&mut rng),
                    state: StateName().fake_with_rng(&mut rng),
                    postal_code: ZipCode().fake_with_rng(&mut rng),
                    region: format!("Region {}", rng.gen_range(0..5)),
                    product_id: generate_bban(&mut rng),
                    category: sector.to_string(),
                    sub_category: industry.to_string(),
                    item_status: "Regular".to_string(),
                    item_price: (rng.gen_range(1..=100) as f64) * 10.0 + 0.99,
                    sales: rng.gen_range(1..=100) * 100,
                    quantity: rng.gen_range(1..=100) * 10,
                    discount: (rng.gen::<f64>() * 100.0 * 100.0).round() / 100.0,
                    profit: (rng.gen::<f64>() * 1000.0 * 100.0).round() / 100.0,
                };
                chunk.push(row);
            }
            chunk
        })
        .collect();

    // Flatten chunks into single vector
    chunks.into_iter().flatten().collect()
}

/// Generate employee data in parallel using multiple threads.
///
/// This function divides the work across available CPU cores for faster
/// generation of large datasets.
///
/// # Arguments
/// * `count` - Total number of employees to generate
/// * `seed` - Optional seed for reproducibility
///
/// # Example
/// ```
/// use superstore::parallel::employees_parallel;
///
/// // Generate 1 million employees using all CPU cores
/// let data = employees_parallel(1_000_000, Some(42));
/// ```
pub fn employees_parallel(count: usize, seed: Option<u64>) -> Vec<EmployeeRow> {
    let num_threads = rayon::current_num_threads();
    let chunk_size = (count + num_threads - 1) / num_threads;

    let chunks: Vec<Vec<EmployeeRow>> = (0..num_threads)
        .into_par_iter()
        .map(|thread_idx| {
            let start_idx = thread_idx * chunk_size;
            let end_idx = (start_idx + chunk_size).min(count);
            if start_idx >= count {
                return Vec::new();
            }

            // Create per-thread RNG with deterministic seed based on thread index
            let mut rng = match seed {
                Some(s) => StdRng::seed_from_u64(s.wrapping_add(thread_idx as u64)),
                None => StdRng::from_entropy(),
            };

            let mut chunk = Vec::with_capacity(end_idx - start_idx);

            for row_id in start_idx..end_idx {
                let row = EmployeeRow {
                    row_id: row_id as i32,
                    employee_id: generate_ein(&mut rng),
                    first_name: FirstName().fake_with_rng(&mut rng),
                    surname: LastName().fake_with_rng(&mut rng),
                    prefix: PREFIXES.choose(&mut rng).unwrap().to_string(),
                    suffix: SUFFIXES.choose(&mut rng).unwrap().to_string(),
                    phone_number: PhoneNumber().fake_with_rng(&mut rng),
                    email: SafeEmail().fake_with_rng(&mut rng),
                    ssn: generate_ssn(&mut rng),
                    street: generate_street_address(&mut rng),
                    city: CityName().fake_with_rng(&mut rng),
                    postal_code: ZipCode().fake_with_rng(&mut rng),
                    region: format!("Region {}", rng.gen_range(0..5)),
                    state: StateName().fake_with_rng(&mut rng),
                    country: "US".to_string(),
                    start_date: random_date_30_years(&mut rng),
                    date_of_birth: random_date_of_birth(&mut rng),
                };
                chunk.push(row);
            }
            chunk
        })
        .collect();

    chunks.into_iter().flatten().collect()
}

/// Get the number of threads Rayon will use for parallel operations.
pub fn num_threads() -> usize {
    rayon::current_num_threads()
}

/// Set the number of threads for parallel operations.
///
/// This should be called early in the program before any parallel operations.
/// Returns an error if the thread pool has already been initialized.
pub fn set_num_threads(num_threads: usize) -> Result<(), rayon::ThreadPoolBuildError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_superstore_parallel_count() {
        let data = superstore_parallel(1000, Some(42));
        assert_eq!(data.len(), 1000);
    }

    #[test]
    fn test_superstore_parallel_reproducible() {
        let data1 = superstore_parallel(100, Some(42));
        let data2 = superstore_parallel(100, Some(42));

        // With same seed and same thread count, should be identical
        assert_eq!(data1.len(), data2.len());
        for (r1, r2) in data1.iter().zip(data2.iter()) {
            assert_eq!(r1.row_id, r2.row_id);
            assert_eq!(r1.order_id, r2.order_id);
        }
    }

    #[test]
    fn test_employees_parallel_count() {
        let data = employees_parallel(1000, Some(42));
        assert_eq!(data.len(), 1000);
    }

    #[test]
    fn test_employees_parallel_reproducible() {
        let data1 = employees_parallel(100, Some(42));
        let data2 = employees_parallel(100, Some(42));

        assert_eq!(data1.len(), data2.len());
        for (r1, r2) in data1.iter().zip(data2.iter()) {
            assert_eq!(r1.row_id, r2.row_id);
            assert_eq!(r1.employee_id, r2.employee_id);
        }
    }

    #[test]
    fn test_parallel_different_from_no_seed() {
        let data1 = superstore_parallel(100, None);
        let data2 = superstore_parallel(100, None);

        // Should be different without seed
        let same_count = data1
            .iter()
            .zip(data2.iter())
            .filter(|(r1, r2)| r1.order_id == r2.order_id)
            .count();
        assert!(same_count < 50); // Very unlikely to be more than half the same
    }

    #[test]
    fn test_num_threads() {
        let threads = num_threads();
        assert!(threads >= 1);
    }
}
