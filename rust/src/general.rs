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

/// Costco-style item status based on price ending
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ItemStatus {
    /// Price ends in .99 - regular full price item
    Regular,
    /// Price ends in .49 or .79 - manufacturer sale or deal
    ManufacturerSale,
    /// Price ends in .97 - store clearance
    Clearance,
    /// Price ends in .88 - returned or floor model
    ReturnedFloorModel,
}

impl ItemStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ItemStatus::Regular => "Regular",
            ItemStatus::ManufacturerSale => "Manufacturer Sale",
            ItemStatus::Clearance => "Clearance",
            ItemStatus::ReturnedFloorModel => "Returned/Floor Model",
        }
    }

    /// Profit margin multiplier for this item status
    /// Regular items have full margins, sale/clearance have reduced margins
    pub fn profit_multiplier(&self) -> f64 {
        match self {
            ItemStatus::Regular => 1.0,
            ItemStatus::ManufacturerSale => 0.4, // Lower margins on sales
            ItemStatus::Clearance => 0.1,        // Very low margins
            ItemStatus::ReturnedFloorModel => 0.05, // Near-zero margins
        }
    }

    /// Get the price ending for this status
    fn price_ending(&self) -> f64 {
        match self {
            ItemStatus::Regular => 0.99,
            ItemStatus::ManufacturerSale => 0.49, // Could also be 0.79
            ItemStatus::Clearance => 0.97,
            ItemStatus::ReturnedFloorModel => 0.88,
        }
    }
}

// Base price points (will be adjusted with status-specific endings)
const BASE_PRICES: [f64; 12] = [
    9.0, 14.0, 19.0, 24.0, 29.0, 39.0, 49.0, 79.0, 99.0, 149.0, 199.0, 299.0,
];

// Item status distribution weights: [Regular, ManufacturerSale, Clearance, ReturnedFloorModel]
// Most items are regular, with smaller portions being sale/clearance
const ITEM_STATUS_WEIGHTS: [f64; 4] = [0.70, 0.15, 0.10, 0.05];

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

// =============================================================================
// Priority 4: Enhanced Superstore Features
// =============================================================================

/// Product bundle definition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProductBundle {
    pub name: String,
    pub categories: Vec<String>,
    pub discount_multiplier: f64,
}

/// Configuration for product bundling behavior
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BundlingConfig {
    pub enable: bool,
    pub bundle_probability: f64,
    pub bundles: Vec<ProductBundle>,
}

impl Default for BundlingConfig {
    fn default() -> Self {
        Self {
            enable: false,
            bundle_probability: 0.15,
            bundles: vec![
                ProductBundle {
                    name: "Home Office".to_string(),
                    categories: vec!["Technology".to_string(), "Office Supplies".to_string()],
                    discount_multiplier: 0.85,
                },
                ProductBundle {
                    name: "Entertainment Center".to_string(),
                    categories: vec!["Technology".to_string(), "Furniture".to_string()],
                    discount_multiplier: 0.88,
                },
                ProductBundle {
                    name: "Back to School".to_string(),
                    categories: vec!["Office Supplies".to_string(), "Technology".to_string()],
                    discount_multiplier: 0.80,
                },
            ],
        }
    }
}

/// Regional product preference weights
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionalPreference {
    pub region: String,
    pub category_weights: Vec<(String, f64)>,
}

/// Configuration for regional variations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionalConfig {
    pub enable: bool,
    pub preferences: Vec<RegionalPreference>,
}

impl Default for RegionalConfig {
    fn default() -> Self {
        Self {
            enable: false,
            preferences: vec![
                RegionalPreference {
                    region: "West".to_string(),
                    category_weights: vec![
                        ("Technology".to_string(), 1.4),
                        ("Furniture".to_string(), 1.0),
                        ("Office Supplies".to_string(), 0.8),
                    ],
                },
                RegionalPreference {
                    region: "East".to_string(),
                    category_weights: vec![
                        ("Technology".to_string(), 1.0),
                        ("Furniture".to_string(), 1.2),
                        ("Office Supplies".to_string(), 1.1),
                    ],
                },
                RegionalPreference {
                    region: "Central".to_string(),
                    category_weights: vec![
                        ("Technology".to_string(), 0.9),
                        ("Furniture".to_string(), 1.3),
                        ("Office Supplies".to_string(), 1.0),
                    ],
                },
                RegionalPreference {
                    region: "South".to_string(),
                    category_weights: vec![
                        ("Technology".to_string(), 1.1),
                        ("Furniture".to_string(), 1.1),
                        ("Office Supplies".to_string(), 1.0),
                    ],
                },
            ],
        }
    }
}

/// Configuration for inventory effects
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InventoryConfig {
    pub enable: bool,
    pub stock_out_probability: f64,
    pub backorder_delay_days: i32,
    pub low_stock_threshold: f64,
    pub low_stock_price_premium: f64,
}

impl Default for InventoryConfig {
    fn default() -> Self {
        Self {
            enable: false,
            stock_out_probability: 0.05,
            backorder_delay_days: 7,
            low_stock_threshold: 0.2,
            low_stock_price_premium: 1.05,
        }
    }
}

/// Payment method types with associated fraud rates
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PaymentMethod {
    CreditCard,
    DebitCard,
    PayPal,
    GiftCard,
    BankTransfer,
    CashOnDelivery,
}

impl PaymentMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            PaymentMethod::CreditCard => "Credit Card",
            PaymentMethod::DebitCard => "Debit Card",
            PaymentMethod::PayPal => "PayPal",
            PaymentMethod::GiftCard => "Gift Card",
            PaymentMethod::BankTransfer => "Bank Transfer",
            PaymentMethod::CashOnDelivery => "Cash on Delivery",
        }
    }

    pub fn fraud_rate(&self) -> f64 {
        match self {
            PaymentMethod::CreditCard => 0.02,
            PaymentMethod::DebitCard => 0.015,
            PaymentMethod::PayPal => 0.01,
            PaymentMethod::GiftCard => 0.03,
            PaymentMethod::BankTransfer => 0.005,
            PaymentMethod::CashOnDelivery => 0.001,
        }
    }

    pub fn processing_fee_rate(&self) -> f64 {
        match self {
            PaymentMethod::CreditCard => 0.029,
            PaymentMethod::DebitCard => 0.021,
            PaymentMethod::PayPal => 0.034,
            PaymentMethod::GiftCard => 0.0,
            PaymentMethod::BankTransfer => 0.005,
            PaymentMethod::CashOnDelivery => 0.0,
        }
    }
}

// Payment method distribution weights
const PAYMENT_METHOD_WEIGHTS: [f64; 6] = [0.40, 0.25, 0.15, 0.08, 0.07, 0.05];

/// Configuration for payment methods
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaymentConfig {
    pub enable: bool,
    pub fraud_simulation: bool,
}

impl Default for PaymentConfig {
    fn default() -> Self {
        Self {
            enable: false,
            fraud_simulation: true,
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
    // Priority 4 enhancements
    pub bundling: BundlingConfig,
    pub regional: RegionalConfig,
    pub inventory: InventoryConfig,
    pub payment: PaymentConfig,
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
                "West".to_string(),
                "East".to_string(),
                "Central".to_string(),
                "South".to_string(),
            ],
            bundling: BundlingConfig::default(),
            regional: RegionalConfig::default(),
            inventory: InventoryConfig::default(),
            payment: PaymentConfig::default(),
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

// =============================================================================
// Priority 4: Helper Functions
// =============================================================================

/// Generate a payment method based on weighted distribution
fn generate_payment_method<R: Rng>(rng: &mut R) -> PaymentMethod {
    let total: f64 = PAYMENT_METHOD_WEIGHTS.iter().sum();
    let roll = rng.gen::<f64>() * total;
    let mut cumulative = 0.0;

    for (i, &weight) in PAYMENT_METHOD_WEIGHTS.iter().enumerate() {
        cumulative += weight;
        if roll < cumulative {
            return match i {
                0 => PaymentMethod::CreditCard,
                1 => PaymentMethod::DebitCard,
                2 => PaymentMethod::PayPal,
                3 => PaymentMethod::GiftCard,
                4 => PaymentMethod::BankTransfer,
                _ => PaymentMethod::CashOnDelivery,
            };
        }
    }
    PaymentMethod::CreditCard
}

/// Check if this transaction is fraudulent based on payment method fraud rate
fn check_fraud<R: Rng>(rng: &mut R, payment_method: &PaymentMethod) -> bool {
    rng.gen::<f64>() < payment_method.fraud_rate()
}

/// Determine stock status and backorder days
fn determine_stock_status<R: Rng>(rng: &mut R, config: &InventoryConfig) -> (String, Option<i32>) {
    if !config.enable {
        return ("In Stock".to_string(), None);
    }

    let roll = rng.gen::<f64>();

    if roll < config.stock_out_probability {
        // Item is out of stock, will be backordered
        let delay = rng.gen_range(1..=config.backorder_delay_days);
        ("Backorder".to_string(), Some(delay))
    } else if roll < config.stock_out_probability + config.low_stock_threshold {
        // Low stock but available
        ("Low Stock".to_string(), None)
    } else {
        // Normal stock
        ("In Stock".to_string(), None)
    }
}

/// Apply regional preference to category selection - returns index of selected category
fn apply_regional_preference<R: Rng>(
    rng: &mut R,
    region: &str,
    categories: &[&str],
    config: &RegionalConfig,
) -> usize {
    if !config.enable || categories.is_empty() {
        return rng.gen_range(0..categories.len().max(1));
    }

    // Find preference for this region
    let preference = config.preferences.iter().find(|p| p.region == region);

    if let Some(pref) = preference {
        // Build weighted list
        let mut weights: Vec<f64> = Vec::with_capacity(categories.len());

        for &cat in categories {
            let weight = pref
                .category_weights
                .iter()
                .find(|(c, _)| c == cat)
                .map(|(_, w)| *w)
                .unwrap_or(1.0);
            weights.push(weight);
        }

        // Select based on weights
        let total_weight: f64 = weights.iter().sum();
        let roll = rng.gen::<f64>() * total_weight;
        let mut cumulative = 0.0;

        for (idx, weight) in weights.iter().enumerate() {
            cumulative += weight;
            if roll < cumulative {
                return idx;
            }
        }
    }

    rng.gen_range(0..categories.len().max(1))
}

/// Check if current order should be part of a bundle
fn check_bundle<R: Rng>(
    rng: &mut R,
    config: &BundlingConfig,
    order_id: &str,
) -> Option<(String, f64)> {
    if !config.enable {
        return None;
    }

    if rng.gen::<f64>() < config.bundle_probability {
        // Select a random bundle
        if let Some(bundle) = config.bundles.choose(rng) {
            // Create a unique bundle ID based on order
            let bundle_id = format!(
                "BDL-{}-{}",
                &order_id[..6],
                bundle
                    .name
                    .chars()
                    .take(3)
                    .collect::<String>()
                    .to_uppercase()
            );
            return Some((bundle_id, bundle.discount_multiplier));
        }
    }
    None
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

/// Generate item status based on weighted distribution
/// Can optionally bias toward sale/clearance for high-discount items
fn generate_item_status<R: Rng>(rng: &mut R, discount_factor: f64) -> ItemStatus {
    // Higher discounts increase chance of being a sale/clearance item
    let discount_bias = discount_factor.clamp(0.0, 1.0);

    // Adjust weights: high discount -> more likely to be sale/clearance
    let regular_weight = ITEM_STATUS_WEIGHTS[0] * (1.0 - discount_bias * 0.5);
    let sale_weight = ITEM_STATUS_WEIGHTS[1] * (1.0 + discount_bias * 1.5);
    let clearance_weight = ITEM_STATUS_WEIGHTS[2] * (1.0 + discount_bias * 2.0);
    let returned_weight = ITEM_STATUS_WEIGHTS[3] * (1.0 + discount_bias * 0.5);

    let total = regular_weight + sale_weight + clearance_weight + returned_weight;
    let roll = rng.gen::<f64>() * total;

    if roll < regular_weight {
        ItemStatus::Regular
    } else if roll < regular_weight + sale_weight {
        ItemStatus::ManufacturerSale
    } else if roll < regular_weight + sale_weight + clearance_weight {
        ItemStatus::Clearance
    } else {
        ItemStatus::ReturnedFloorModel
    }
}

/// Round to realistic Costco-style price point based on item status
/// Returns the price with appropriate ending (.99, .49/.79, .97, .88)
fn round_to_price_point_with_status(value: f64, status: &ItemStatus) -> f64 {
    // Find nearest base price
    let mut best_base = BASE_PRICES[0];
    let mut best_diff = (value - best_base).abs();

    for &base in &BASE_PRICES {
        let diff = (value - base).abs();
        if diff < best_diff {
            best_base = base;
            best_diff = diff;
        }
    }

    // Use the base price if close enough, otherwise use the value's floor
    let final_base = if best_diff < value * 0.3 {
        best_base
    } else {
        value.floor()
    };

    // Apply status-specific price ending
    final_base + status.price_ending()
}

/// Apply volume effects based on item status
/// Sale items have bimodal distribution: either high volume (good deal) or low volume (unwanted)
fn apply_item_status_volume_effect<R: Rng>(
    rng: &mut R,
    base_quantity: i32,
    status: &ItemStatus,
) -> i32 {
    match status {
        ItemStatus::Regular => base_quantity,
        ItemStatus::ManufacturerSale => {
            // Bimodal: 70% chance of high volume (good deal), 30% low volume
            if rng.gen::<f64>() < 0.7 {
                // Good deal - higher volume (1.3x to 2.0x)
                ((base_quantity as f64) * rng.gen_range(1.3..2.0)).round() as i32
            } else {
                // Not in demand - lower volume (0.5x to 0.8x)
                ((base_quantity as f64) * rng.gen_range(0.5..0.8)).round() as i32
            }
        }
        ItemStatus::Clearance => {
            // Bimodal: 40% high volume (finally affordable), 60% low (nobody wants it)
            if rng.gen::<f64>() < 0.4 {
                // Clearance deal hunters - moderate boost (1.2x to 1.5x)
                ((base_quantity as f64) * rng.gen_range(1.2..1.5)).round() as i32
            } else {
                // Unwanted items - low volume (0.3x to 0.6x)
                ((base_quantity as f64) * rng.gen_range(0.3..0.6)).round() as i32
            }
        }
        ItemStatus::ReturnedFloorModel => {
            // Generally low volume - these are one-off items (0.4x to 0.7x)
            // Occasionally someone snags a deal (1.0x)
            if rng.gen::<f64>() < 0.2 {
                base_quantity // Lucky find
            } else {
                ((base_quantity as f64) * rng.gen_range(0.4..0.7)).round() as i32
            }
        }
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
    pub item_status: String,
    pub item_price: f64,
    pub sales: i32,
    pub quantity: i32,
    pub discount: f64,
    pub profit: f64,
    // Priority 4 enhancements
    pub bundle_id: Option<String>,
    pub payment_method: Option<String>,
    pub is_fraud: Option<bool>,
    pub processing_fee: Option<f64>,
    pub backorder_days: Option<i32>,
    pub stock_status: Option<String>,
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
        let mut ship_date = random_date_between(&mut rng, order_date);

        // Choose region from config first (needed for regional preferences)
        let region = config
            .regions
            .choose(&mut rng)
            .unwrap_or(&config.regions[0])
            .clone();

        // Apply regional preference to sector selection
        let sector_idx = apply_regional_preference(&mut rng, &region, &sectors, &config.regional);
        let sector = sectors[sector_idx];
        let industries = US_SECTORS_MAP.get(sector).unwrap();
        let industry = *industries.choose(&mut rng).unwrap();

        // Calculate seasonality multiplier based on order date
        let month = order_date.month();
        let seasonality_mult = get_seasonality_multiplier(month, &config.seasonality);

        // Transform uniform copula values to actual ranges
        let base_sales = config.min_sales as f64 + uniforms[0] * sales_range;
        let sales_with_season = base_sales * seasonality_mult;

        // Generate item status with discount-biased distribution
        let discount = (uniforms[2] * config.max_discount_percent * 100.0).round() / 100.0;
        let discount_factor = discount / config.max_discount_percent;
        let item_status = generate_item_status(&mut rng, discount_factor);

        // Round to Costco-style price point based on item status
        let mut item_price = round_to_price_point_with_status(sales_with_season, &item_status);

        // Generate order ID early (needed for bundle ID)
        let order_id = generate_ein(&mut rng);

        // Check for product bundle
        let (bundle_id, bundle_discount) = if config.bundling.enable {
            if let Some((bid, disc)) = check_bundle(&mut rng, &config.bundling, &order_id) {
                // Apply bundle discount to price
                item_price *= disc;
                (Some(bid), disc)
            } else {
                (None, 1.0)
            }
        } else {
            (None, 1.0)
        };

        // Check inventory status
        let (stock_status, backorder_days) = determine_stock_status(&mut rng, &config.inventory);

        // If backordered, adjust ship date
        if let Some(delay) = backorder_days {
            ship_date = ship_date + chrono::Duration::days(delay as i64);
        }

        // Apply inventory low stock premium
        if config.inventory.enable && stock_status == "Low Stock" {
            item_price *= config.inventory.low_stock_price_premium;
        }

        // Quantity with promotional boost and item status effects
        let base_quantity = config.min_quantity as f64 + uniforms[1] * quantity_range;
        let quantity_with_promotion = apply_promotional_effects(
            &mut rng,
            base_quantity.round() as i32,
            discount,
            &config.promotions,
        );
        // Apply item status volume effects (bimodal for sale/clearance)
        let quantity_with_status =
            apply_item_status_volume_effect(&mut rng, quantity_with_promotion, &item_status);
        let quantity = quantity_with_status.clamp(config.min_quantity, config.max_quantity);

        // Customer with cohort behavior
        let (customer_id, is_vip) =
            generate_customer_id(&mut rng, &customer_pool, &config.customers);

        // VIP customers get bigger orders
        let vip_mult = if is_vip && config.customers.enable_cohorts {
            config.customers.vip_order_multiplier
        } else {
            1.0
        };
        let final_sales = (item_price * vip_mult).round() as i32;
        let final_quantity = ((quantity as f64) * vip_mult.sqrt()).round() as i32;

        // Payment method generation
        let (payment_method, is_fraud, processing_fee) = if config.payment.enable {
            let pm = generate_payment_method(&mut rng);
            let fraud = if config.payment.fraud_simulation {
                check_fraud(&mut rng, &pm)
            } else {
                false
            };
            let fee = (final_sales as f64) * pm.processing_fee_rate();
            (
                Some(pm.as_str().to_string()),
                Some(fraud),
                Some((fee * 100.0).round() / 100.0),
            )
        } else {
            (None, None, None)
        };

        // Profit calculation with item status correlation
        // Sale/clearance items have reduced profit margins
        let base_profit = -500.0 + uniforms[3] * 3500.0;
        // High discounts hurt profit more
        let discount_penalty = (discount / 100.0) * 500.0;
        // Apply item status profit multiplier (regular=1.0, sale=0.4, clearance=0.1, returned=0.05)
        let status_adjusted_profit =
            (base_profit - discount_penalty) * item_status.profit_multiplier();
        // Apply bundle discount effect on profit
        let bundle_adjusted_profit = status_adjusted_profit * bundle_discount;
        // Deduct processing fee if applicable
        let fee_adjusted_profit = bundle_adjusted_profit - processing_fee.unwrap_or(0.0);
        let profit = (fee_adjusted_profit * seasonality_mult * 100.0).round() / 100.0;

        let row = SuperstoreRow {
            row_id: id as i32,
            order_id,
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
            item_status: item_status.as_str().to_string(),
            item_price: (item_price * 100.0).round() / 100.0,
            sales: final_sales,
            quantity: final_quantity,
            discount,
            profit,
            // Priority 4 fields
            bundle_id,
            payment_method,
            is_fraud,
            processing_fee,
            backorder_days,
            stock_status: if config.inventory.enable {
                Some(stock_status)
            } else {
                None
            },
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
