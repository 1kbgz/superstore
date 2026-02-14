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

// Realistic price points ($9.99, $19.99, etc.)
const PRICE_POINTS: [f64; 12] = [
    9.99, 14.99, 19.99, 24.99, 29.99, 39.99, 49.99, 79.99, 99.99, 149.99, 199.99, 299.99,
];

// =============================================================================
// Superstore Configuration Structs
// =============================================================================

/// Configuration for seasonal patterns in sales data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeasonalityConfig {
    pub enable: bool,
    pub q4_multiplier: f64,
    pub summer_multiplier: f64,
    pub back_to_school_multiplier: f64,
}

impl Default for SeasonalityConfig {
    fn default() -> Self {
        Self {
            enable: false,
            q4_multiplier: 1.5,
            summer_multiplier: 0.9,
            back_to_school_multiplier: 1.2,
        }
    }
}

/// Configuration for promotional effects
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromotionalConfig {
    pub enable: bool,
    pub discount_quantity_correlation: f64,
    pub price_elasticity: f64,
}

impl Default for PromotionalConfig {
    fn default() -> Self {
        Self {
            enable: false,
            discount_quantity_correlation: 0.5,
            price_elasticity: -0.8,
        }
    }
}

/// Configuration for customer behavior patterns
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CustomerConfig {
    pub enable_cohorts: bool,
    pub repeat_customer_rate: f64,
    pub vip_segment_rate: f64,
    pub vip_order_multiplier: f64,
}

impl Default for CustomerConfig {
    fn default() -> Self {
        Self {
            enable_cohorts: false,
            repeat_customer_rate: 0.3,
            vip_segment_rate: 0.1,
            vip_order_multiplier: 2.0,
        }
    }
}

/// Full superstore configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperstoreConfig {
    pub count: usize,
    pub pool_size: usize,
    pub seed: Option<u64>,
    pub min_sales: i32,
    pub max_sales: i32,
    pub min_quantity: i32,
    pub max_quantity: i32,
    pub max_discount_percent: f64,
    pub sales_quantity_correlation: f64,
    pub sales_profit_correlation: f64,
    pub discount_profit_correlation: f64,
    pub seasonality: SeasonalityConfig,
    pub promotions: PromotionalConfig,
    pub customers: CustomerConfig,
    pub regions: Vec<String>,
}

impl Default for SuperstoreConfig {
    fn default() -> Self {
        Self {
            count: 1000,
            pool_size: DEFAULT_POOL_SIZE,
            seed: None,
            min_sales: 100,
            max_sales: 10000,
            min_quantity: 1,
            max_quantity: 100,
            max_discount_percent: 50.0,
            sales_quantity_correlation: 0.7,
            sales_profit_correlation: 0.6,
            discount_profit_correlation: -0.4,
            seasonality: SeasonalityConfig::default(),
            promotions: PromotionalConfig::default(),
            customers: CustomerConfig::default(),
            regions: vec![
                "Region 1".to_string(),
                "Region 2".to_string(),
                "Region 3".to_string(),
                "Region 4".to_string(),
                "Region 5".to_string(),
            ],
        }
    }
}

/// Calculate seasonality multiplier based on month
fn get_seasonality_multiplier(month: u32, config: &SeasonalityConfig) -> f64 {
    if !config.enable {
        return 1.0;
    }

    match month {
        // Q4 holiday season (Oct-Dec)
        10 | 11 | 12 => config.q4_multiplier,
        // Summer slump (Jun-Jul)
        6 | 7 => config.summer_multiplier,
        // Back to school (Aug-Sep)
        8 | 9 => config.back_to_school_multiplier,
        // Normal months
        _ => 1.0,
    }
}

/// Apply promotional effects: higher discount -> higher quantity
fn apply_promotional_effects<R: Rng>(
    rng: &mut R,
    base_quantity: i32,
    discount: f64,
    config: &PromotionalConfig,
) -> i32 {
    if !config.enable {
        return base_quantity;
    }

    // Higher discounts increase quantity by the correlation factor
    // e.g., 50% discount with 0.5 correlation -> 25% quantity boost
    let discount_boost = 1.0 + (discount / 100.0) * config.discount_quantity_correlation;

    // Price elasticity adds some random variation
    let elasticity_factor = 1.0 + rng.gen_range(-0.1..0.1) * config.price_elasticity.abs();

    ((base_quantity as f64) * discount_boost * elasticity_factor).round() as i32
}

/// Generate a customer ID with cohort behavior
fn generate_customer_id<R: Rng>(
    rng: &mut R,
    customer_pool: &[String],
    config: &CustomerConfig,
) -> (String, bool) {
    if !config.enable_cohorts {
        return (generate_license_plate(rng), false);
    }

    // Determine if this is a repeat customer
    if rng.gen::<f64>() < config.repeat_customer_rate {
        // Pick from existing customer pool
        let customer_id = customer_pool.choose(rng).unwrap().clone();
        let is_vip = rng.gen::<f64>() < config.vip_segment_rate;
        (customer_id, is_vip)
    } else {
        // New customer
        let is_vip = rng.gen::<f64>() < config.vip_segment_rate;
        (generate_license_plate(rng), is_vip)
    }
}

/// Round to realistic price point ($X.99 style)
fn round_to_price_point(value: f64) -> f64 {
    // Find nearest price point
    let mut best = PRICE_POINTS[0];
    let mut best_diff = (value - best).abs();

    for &point in &PRICE_POINTS {
        let diff = (value - point).abs();
        if diff < best_diff {
            best = point;
            best_diff = diff;
        }
    }

    // Return either the price point or just make it end in .99
    if best_diff < value * 0.3 {
        best
    } else {
        // Just round to nearest .99
        (value.floor()) + 0.99
    }
}

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
    superstore_with_config(&SuperstoreConfig {
        count,
        pool_size: pool_size.unwrap_or(DEFAULT_POOL_SIZE),
        seed,
        ..Default::default()
    })
}

/// Generate superstore data with full configuration
pub fn superstore_with_config(config: &SuperstoreConfig) -> Vec<SuperstoreRow> {
    let mut rng = create_rng(config.seed);
    let mut data = Vec::with_capacity(config.count);

    let sectors: Vec<&str> = US_SECTORS.clone();

    // Pre-generate location pool for performance
    let location_pool = LocationPool::generate(&mut rng, config.pool_size);

    // Pre-generate customer pool for repeat customer simulation
    let customer_pool: Vec<String> = if config.customers.enable_cohorts {
        (0..100).map(|_| generate_license_plate(&mut rng)).collect()
    } else {
        Vec::new()
    };

    // Build correlation matrix from config
    //   [Sales, Quantity, Discount, Profit]
    // We use configured correlations for key relationships
    let sq = config.sales_quantity_correlation;
    let sp = config.sales_profit_correlation;
    let dp = config.discount_profit_correlation;
    // Derived correlations (keep matrix positive semi-definite)
    let qp = (sq * sp).clamp(-0.99, 0.99); // Quantity-Profit derived
    let ds = (-0.2_f64).clamp(-0.99, 0.99); // Discount-Sales weak negative
    let dq = (0.1_f64).clamp(-0.99, 0.99); // Discount-Quantity weak positive (more discount -> more bought)

    let correlation_matrix = vec![
        vec![1.0, sq, ds, sp], // Sales
        vec![sq, 1.0, dq, qp], // Quantity
        vec![ds, dq, 1.0, dp], // Discount
        vec![sp, qp, dp, 1.0], // Profit
    ];

    // Pre-generate all correlated values using copula
    let correlated_values = if let Ok(copula) = GaussianCopula::new(correlation_matrix) {
        copula.sample_n(&mut rng, config.count)
    } else {
        // Fallback to independent uniform values
        (0..config.count)
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

    let sales_range = (config.max_sales - config.min_sales) as f64;
    let quantity_range = (config.max_quantity - config.min_quantity) as f64;

    for (id, uniforms) in correlated_values.into_iter().enumerate() {
        let order_date = random_date_this_year(&mut rng);
        let ship_date = random_date_between(&mut rng, order_date);

        let sector = *sectors.choose(&mut rng).unwrap();
        let industries = US_SECTORS_MAP.get(sector).unwrap();
        let industry = *industries.choose(&mut rng).unwrap();

        // Calculate seasonality multiplier based on order date
        let month = order_date.month();
        let seasonality_mult = get_seasonality_multiplier(month, &config.seasonality);

        // Transform uniform copula values to actual ranges
        let base_sales = config.min_sales as f64 + uniforms[0] * sales_range;
        let sales_with_season = base_sales * seasonality_mult;
        let sales = round_to_price_point(sales_with_season);

        // Quantity with potential promotional boost
        let base_quantity = config.min_quantity as f64 + uniforms[1] * quantity_range;
        let discount = (uniforms[2] * config.max_discount_percent * 100.0).round() / 100.0;
        let quantity_with_promotion = apply_promotional_effects(
            &mut rng,
            base_quantity.round() as i32,
            discount,
            &config.promotions,
        );
        let quantity = quantity_with_promotion.clamp(config.min_quantity, config.max_quantity);

        // Customer with cohort behavior
        let (customer_id, is_vip) =
            generate_customer_id(&mut rng, &customer_pool, &config.customers);

        // VIP customers get bigger orders
        let vip_mult = if is_vip && config.customers.enable_cohorts {
            config.customers.vip_order_multiplier
        } else {
            1.0
        };
        let final_sales = (sales * vip_mult).round() as i32;
        let final_quantity = ((quantity as f64) * vip_mult.sqrt()).round() as i32;

        // Profit calculation (can be negative for high discount items)
        let base_profit = -500.0 + uniforms[3] * 3500.0;
        // High discounts hurt profit more
        let discount_penalty = (discount / 100.0) * 500.0;
        let profit = ((base_profit - discount_penalty) * seasonality_mult * 100.0).round() / 100.0;

        // Choose region from config
        let region = config
            .regions
            .choose(&mut rng)
            .unwrap_or(&config.regions[0])
            .clone();

        let row = SuperstoreRow {
            row_id: id as i32,
            order_id: generate_ein(&mut rng),
            order_date: order_date.format("%Y-%m-%d").to_string(),
            ship_date: ship_date.format("%Y-%m-%d").to_string(),
            ship_mode: SHIP_MODES.choose(&mut rng).unwrap().to_string(),
            customer_id,
            segment: SEGMENTS.choose(&mut rng).unwrap().to_string(),
            country: "US".to_string(),
            city: location_pool.random_city(&mut rng).to_string(),
            state: location_pool.random_state(&mut rng).to_string(),
            postal_code: location_pool.random_zip(&mut rng).to_string(),
            region,
            product_id: generate_bban(&mut rng),
            category: sector.to_string(),
            sub_category: industry.to_string(),
            sales: final_sales,
            quantity: final_quantity,
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
