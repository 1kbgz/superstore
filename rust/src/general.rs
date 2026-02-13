use chrono::{Datelike, NaiveDate, Utc};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::copulas::GaussianCopula;
use crate::utils::{US_SECTORS, US_SECTORS_MAP};

use fake::faker::address::en::{CityName, StateName, ZipCode};
use fake::faker::internet::en::SafeEmail;
use fake::faker::name::en::{FirstName, LastName};
use fake::faker::phone_number::en::PhoneNumber;
use fake::Fake;

const SHIP_MODES: [&str; 3] = ["First Class", "Standard Class", "Second Class"];
const SEGMENTS: [&str; 4] = ["A", "B", "C", "D"];
const PREFIXES: [&str; 6] = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Rev."];
const SUFFIXES: [&str; 4] = ["Jr.", "Sr.", "III", "IV"];

// Default pool size for pre-generated data
const DEFAULT_POOL_SIZE: usize = 1000;

struct LocationPool {
    cities: Vec<String>,
    states: Vec<String>,
    zip_codes: Vec<String>,
}

impl LocationPool {
    fn generate<R: Rng>(rng: &mut R, pool_size: usize) -> Self {
        let cities: Vec<String> = (0..pool_size)
            .map(|_| CityName().fake_with_rng(rng))
            .collect();
        let states: Vec<String> = (0..pool_size)
            .map(|_| StateName().fake_with_rng(rng))
            .collect();
        let zip_codes: Vec<String> = (0..pool_size)
            .map(|_| ZipCode().fake_with_rng(rng))
            .collect();
        Self {
            cities,
            states,
            zip_codes,
        }
    }

    fn random_city<R: Rng>(&self, rng: &mut R) -> &str {
        self.cities.choose(rng).unwrap()
    }

    fn random_state<R: Rng>(&self, rng: &mut R) -> &str {
        self.states.choose(rng).unwrap()
    }

    fn random_zip<R: Rng>(&self, rng: &mut R) -> &str {
        self.zip_codes.choose(rng).unwrap()
    }
}

struct NamePool {
    first_names: Vec<String>,
    last_names: Vec<String>,
    emails: Vec<String>,
    phone_numbers: Vec<String>,
}

impl NamePool {
    fn generate<R: Rng>(rng: &mut R, pool_size: usize) -> Self {
        let first_names: Vec<String> = (0..pool_size)
            .map(|_| FirstName().fake_with_rng(rng))
            .collect();
        let last_names: Vec<String> = (0..pool_size)
            .map(|_| LastName().fake_with_rng(rng))
            .collect();
        let emails: Vec<String> = (0..pool_size)
            .map(|_| SafeEmail().fake_with_rng(rng))
            .collect();
        let phone_numbers: Vec<String> = (0..pool_size)
            .map(|_| PhoneNumber().fake_with_rng(rng))
            .collect();
        Self {
            first_names,
            last_names,
            emails,
            phone_numbers,
        }
    }
}

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

/// Create an RNG from an optional seed
fn create_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperstoreRow {
    pub row_id: i32,
    pub order_id: String,
    pub order_date: String,
    pub ship_date: String,
    pub ship_mode: String,
    pub customer_id: String,
    pub segment: String,
    pub country: String,
    pub city: String,
    pub state: String,
    pub postal_code: String,
    pub region: String,
    pub product_id: String,
    pub category: String,
    pub sub_category: String,
    pub sales: i32,
    pub quantity: i32,
    pub discount: f64,
    pub profit: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmployeeRow {
    pub row_id: i32,
    pub employee_id: String,
    pub first_name: String,
    pub surname: String,
    pub prefix: String,
    pub suffix: String,
    pub phone_number: String,
    pub email: String,
    pub ssn: String,
    pub street: String,
    pub city: String,
    pub postal_code: String,
    pub region: String,
    pub state: String,
    pub country: String,
    pub start_date: NaiveDate,
    pub date_of_birth: NaiveDate,
}

pub fn superstore(count: usize, seed: Option<u64>, pool_size: Option<usize>) -> Vec<SuperstoreRow> {
    let mut rng = create_rng(seed);
    let mut data = Vec::with_capacity(count);

    let sectors: Vec<&str> = US_SECTORS.clone();

    // Pre-generate location pool for performance (avoids calling fake crate 3M times)
    let actual_pool_size = pool_size.unwrap_or(DEFAULT_POOL_SIZE);
    let location_pool = LocationPool::generate(&mut rng, actual_pool_size);

    // Pre-generate region strings to avoid format! in hot loop
    let regions: [String; 5] = [
        "Region 0".to_string(),
        "Region 1".to_string(),
        "Region 2".to_string(),
        "Region 3".to_string(),
        "Region 4".to_string(),
    ];

    // Correlation matrix for Sales, Quantity, Discount, Profit
    // Sales-Quantity: 0.7, Sales-Profit: 0.6, Quantity-Profit: 0.5
    // Discount-Profit: -0.4, Discount-Sales: -0.2, Discount-Quantity: 0.1
    let correlation_matrix = vec![
        vec![1.0, 0.7, -0.2, 0.6],  // Sales
        vec![0.7, 1.0, 0.1, 0.5],   // Quantity
        vec![-0.2, 0.1, 1.0, -0.4], // Discount
        vec![0.6, 0.5, -0.4, 1.0],  // Profit
    ];

    // Pre-generate all correlated values using copula
    let correlated_values = if let Ok(copula) = GaussianCopula::new(correlation_matrix) {
        copula.sample_n(&mut rng, count)
    } else {
        // Fallback to independent uniform values
        (0..count)
            .map(|_| {
                vec![
                    rng.gen::<f64>(),
                    rng.gen::<f64>(),
                    rng.gen::<f64>(),
                    rng.gen::<f64>(),
                ]
            })
            .collect()
    };

    for (id, uniforms) in correlated_values.into_iter().enumerate() {
        let order_date = random_date_this_year(&mut rng);
        let ship_date = random_date_between(&mut rng, order_date);

        let sector = *sectors.choose(&mut rng).unwrap();
        let industries = US_SECTORS_MAP.get(sector).unwrap();
        let industry = *industries.choose(&mut rng).unwrap();

        // Transform uniform copula values to actual ranges
        // Sales: 100 to 10000 (log-normal-ish via quantile)
        let sales = (100.0 + uniforms[0] * 9900.0).round() as i32;
        // Quantity: 1 to 100
        let quantity = (1.0 + uniforms[1] * 99.0).round() as i32;
        // Discount: 0 to 50%
        let discount = (uniforms[2] * 50.0 * 100.0).round() / 100.0;
        // Profit: -500 to 3000 (can be negative)
        let profit = ((-500.0 + uniforms[3] * 3500.0) * 100.0).round() / 100.0;

        let row = SuperstoreRow {
            row_id: id as i32,
            order_id: generate_ein(&mut rng),
            order_date: order_date.format("%Y-%m-%d").to_string(),
            ship_date: ship_date.format("%Y-%m-%d").to_string(),
            ship_mode: SHIP_MODES.choose(&mut rng).unwrap().to_string(),
            customer_id: generate_license_plate(&mut rng),
            segment: SEGMENTS.choose(&mut rng).unwrap().to_string(),
            country: "US".to_string(),
            city: location_pool.random_city(&mut rng).to_string(),
            state: location_pool.random_state(&mut rng).to_string(),
            postal_code: location_pool.random_zip(&mut rng).to_string(),
            region: regions[rng.gen_range(0..5)].clone(),
            product_id: generate_bban(&mut rng),
            category: sector.to_string(),
            sub_category: industry.to_string(),
            sales,
            quantity,
            discount,
            profit,
        };
        data.push(row);
    }

    data
}

pub fn employees(count: usize, seed: Option<u64>, pool_size: Option<usize>) -> Vec<EmployeeRow> {
    let mut rng = create_rng(seed);
    let mut data = Vec::with_capacity(count);

    // Pre-generate pools for performance
    let actual_pool_size = pool_size.unwrap_or(DEFAULT_POOL_SIZE);
    let location_pool = LocationPool::generate(&mut rng, actual_pool_size);
    let name_pool = NamePool::generate(&mut rng, actual_pool_size);

    // Pre-generate region strings
    let regions: [String; 5] = [
        "Region 0".to_string(),
        "Region 1".to_string(),
        "Region 2".to_string(),
        "Region 3".to_string(),
        "Region 4".to_string(),
    ];

    for id in 0..count {
        let row = EmployeeRow {
            row_id: id as i32,
            employee_id: generate_license_plate(&mut rng),
            first_name: name_pool.first_names.choose(&mut rng).unwrap().clone(),
            surname: name_pool.last_names.choose(&mut rng).unwrap().clone(),
            prefix: PREFIXES.choose(&mut rng).unwrap().to_string(),
            suffix: SUFFIXES.choose(&mut rng).unwrap().to_string(),
            phone_number: name_pool.phone_numbers.choose(&mut rng).unwrap().clone(),
            email: name_pool.emails.choose(&mut rng).unwrap().clone(),
            ssn: generate_ssn(&mut rng),
            street: generate_street_address(&mut rng),
            city: location_pool.random_city(&mut rng).to_string(),
            postal_code: location_pool.random_zip(&mut rng).to_string(),
            region: regions[rng.gen_range(0..5)].clone(),
            state: location_pool.random_state(&mut rng).to_string(),
            country: "US".to_string(),
            start_date: random_date_30_years(&mut rng),
            date_of_birth: random_date_of_birth(&mut rng),
        };
        data.push(row);
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_superstore() {
        let data = superstore(100, None, None);
        assert_eq!(data.len(), 100);
        for (i, row) in data.iter().enumerate() {
            assert_eq!(row.row_id, i as i32);
            assert_eq!(row.country, "US");
            assert!(SHIP_MODES.contains(&row.ship_mode.as_str()));
            assert!(SEGMENTS.contains(&row.segment.as_str()));
        }
    }

    #[test]
    fn test_superstore_seeded() {
        let data1 = superstore(10, Some(12345), None);
        let data2 = superstore(10, Some(12345), None);
        // Same seed should produce same results
        for (r1, r2) in data1.iter().zip(data2.iter()) {
            assert_eq!(r1.order_id, r2.order_id);
            assert_eq!(r1.customer_id, r2.customer_id);
            assert_eq!(r1.city, r2.city);
            assert_eq!(r1.sales, r2.sales);
        }
    }

    #[test]
    fn test_employees() {
        let data = employees(100, None, None);
        assert_eq!(data.len(), 100);
        for (i, row) in data.iter().enumerate() {
            assert_eq!(row.row_id, i as i32);
            assert_eq!(row.country, "US");
        }
    }

    #[test]
    fn test_employees_seeded() {
        let data1 = employees(10, Some(54321), None);
        let data2 = employees(10, Some(54321), None);
        // Same seed should produce same results
        for (r1, r2) in data1.iter().zip(data2.iter()) {
            assert_eq!(r1.employee_id, r2.employee_id);
            assert_eq!(r1.first_name, r2.first_name);
            assert_eq!(r1.surname, r2.surname);
            assert_eq!(r1.ssn, r2.ssn);
        }
    }
}
