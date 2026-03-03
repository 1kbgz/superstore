//! E-commerce data generator module.
//!
//! Generates realistic e-commerce data including:
//! - User sessions via MarkovChain state machines
//! - Shopping cart events with abandonment patterns
//! - Customer RFM (Recency, Frequency, Monetary) metrics
//! - Product catalog with categories and pricing
//! - Conversion funnels with realistic drop-off rates

use chrono::{Duration, NaiveDateTime, Utc};
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp, LogNormal};
use serde::{Deserialize, Serialize};

use crate::temporal::MarkovChain;

// =============================================================================
// Constants
// =============================================================================

const DEVICE_TYPES: &[&str] = &["desktop", "mobile", "tablet"];
const DEVICE_WEIGHTS: &[f64] = &[0.35, 0.55, 0.10];

const BROWSERS: &[&str] = &["Chrome", "Safari", "Firefox", "Edge", "Samsung Internet"];
const BROWSER_WEIGHTS: &[f64] = &[0.65, 0.18, 0.08, 0.05, 0.04];

const TRAFFIC_SOURCES: &[&str] = &[
    "organic",
    "paid_search",
    "social",
    "email",
    "direct",
    "referral",
    "affiliate",
];
const TRAFFIC_SOURCE_WEIGHTS: &[f64] = &[0.30, 0.20, 0.15, 0.10, 0.15, 0.05, 0.05];

const LANDING_PAGES: &[&str] = &[
    "/",
    "/products",
    "/category/electronics",
    "/category/clothing",
    "/category/home",
    "/sale",
    "/new-arrivals",
];

const PRODUCT_CATEGORIES: &[&str] = &[
    "Electronics",
    "Clothing",
    "Home & Garden",
    "Sports",
    "Beauty",
    "Books",
    "Toys",
    "Food",
];

const SESSION_STATES: &[&str] = &[
    "landing",
    "browse",
    "view_product",
    "add_to_cart",
    "view_cart",
    "checkout_start",
    "checkout_payment",
    "purchase",
    "exit",
];

// RFM segment names based on scores
const RFM_SEGMENTS: &[&str] = &[
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "Recent Customers",
    "Promising",
    "Customers Needing Attention",
    "About to Sleep",
    "At Risk",
    "Can't Lose Them",
    "Lost",
];

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for session behavior
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Average pages viewed per session
    pub avg_pages_per_session: f64,
    /// Probability of adding item to cart given product view
    pub cart_add_probability: f64,
    /// Probability of starting checkout given cart view
    pub checkout_start_probability: f64,
    /// Probability of completing purchase given checkout start
    pub purchase_completion_probability: f64,
    /// Average session duration in seconds
    pub avg_session_duration_seconds: u32,
    /// Enable session bounces (single-page visits)
    pub enable_bounces: bool,
    /// Bounce rate (probability of immediate exit)
    pub bounce_rate: f64,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            avg_pages_per_session: 5.0,
            cart_add_probability: 0.15,
            checkout_start_probability: 0.40,
            purchase_completion_probability: 0.65,
            avg_session_duration_seconds: 300,
            enable_bounces: true,
            bounce_rate: 0.35,
        }
    }
}

/// Configuration for cart behavior
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CartConfig {
    /// Average items per cart
    pub avg_items_per_cart: f64,
    /// Probability of removing an item from cart
    pub remove_probability: f64,
    /// Probability of updating quantity
    pub quantity_update_probability: f64,
    /// Maximum items per cart
    pub max_items: u32,
    /// Enable cart abandonment simulation
    pub enable_abandonment: bool,
    /// Cart abandonment rate
    pub abandonment_rate: f64,
}

impl Default for CartConfig {
    fn default() -> Self {
        Self {
            avg_items_per_cart: 2.5,
            remove_probability: 0.10,
            quantity_update_probability: 0.05,
            max_items: 20,
            enable_abandonment: true,
            abandonment_rate: 0.70,
        }
    }
}

/// Configuration for product catalog
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatalogConfig {
    /// Number of unique products
    pub num_products: usize,
    /// Minimum product price
    pub min_price: f64,
    /// Maximum product price
    pub max_price: f64,
    /// Price follows log-normal distribution (realistic skew)
    pub lognormal_prices: bool,
    /// Categories to use (defaults to standard categories)
    pub categories: Vec<String>,
}

impl Default for CatalogConfig {
    fn default() -> Self {
        Self {
            num_products: 500,
            min_price: 5.0,
            max_price: 1000.0,
            lognormal_prices: true,
            categories: PRODUCT_CATEGORIES.iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Configuration for RFM (Recency, Frequency, Monetary) analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RfmConfig {
    /// Enable RFM metrics calculation
    pub enable: bool,
    /// Days to look back for recency
    pub recency_window_days: u32,
    /// Number of RFM score buckets (typically 5)
    pub num_buckets: u32,
    /// Pareto distribution shape for customer value (80/20 rule)
    pub pareto_shape: f64,
}

impl Default for RfmConfig {
    fn default() -> Self {
        Self {
            enable: true,
            recency_window_days: 365,
            num_buckets: 5,
            pareto_shape: 1.5,
        }
    }
}

/// Configuration for conversion funnel
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunnelConfig {
    /// Enable funnel stage tracking
    pub enable: bool,
    /// Custom funnel stages (overrides defaults)
    pub stages: Vec<String>,
    /// Time-of-day effects on conversions
    pub time_of_day_effects: bool,
    /// Day-of-week effects on conversions
    pub day_of_week_effects: bool,
}

impl Default for FunnelConfig {
    fn default() -> Self {
        Self {
            enable: true,
            stages: vec![
                "visit".to_string(),
                "view_product".to_string(),
                "add_to_cart".to_string(),
                "checkout".to_string(),
                "purchase".to_string(),
            ],
            time_of_day_effects: true,
            day_of_week_effects: true,
        }
    }
}

/// Full e-commerce configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EcommerceConfig {
    /// Number of sessions to generate
    pub sessions: usize,
    /// Number of unique customers
    pub customers: usize,
    /// Random seed
    pub seed: Option<u64>,
    /// Start date for data generation
    pub start_date: Option<String>,
    /// Number of days to generate
    pub days: u32,
    /// Session configuration
    pub session: SessionConfig,
    /// Cart configuration
    pub cart: CartConfig,
    /// Catalog configuration
    pub catalog: CatalogConfig,
    /// RFM configuration
    pub rfm: RfmConfig,
    /// Funnel configuration
    pub funnel: FunnelConfig,
}

impl Default for EcommerceConfig {
    fn default() -> Self {
        Self {
            sessions: 10000,
            customers: 2000,
            seed: None,
            start_date: None,
            days: 30,
            session: SessionConfig::default(),
            cart: CartConfig::default(),
            catalog: CatalogConfig::default(),
            rfm: RfmConfig::default(),
            funnel: FunnelConfig::default(),
        }
    }
}

// =============================================================================
// Data Structures
// =============================================================================

/// A product in the catalog
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Product {
    pub product_id: String,
    pub name: String,
    pub category: String,
    pub subcategory: String,
    pub price: f64,
    pub rating: f64,
    pub review_count: u32,
    pub in_stock: bool,
}

/// A user session
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Session {
    pub session_id: String,
    pub user_id: String,
    pub start_time: String,
    pub end_time: String,
    pub duration_seconds: u32,
    pub device_type: String,
    pub browser: String,
    pub traffic_source: String,
    pub landing_page: String,
    pub pages_viewed: u32,
    pub bounced: bool,
    pub converted: bool,
    pub total_value: f64,
}

/// A cart event (add, remove, update)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CartEvent {
    pub event_id: String,
    pub session_id: String,
    pub user_id: String,
    pub timestamp: String,
    pub event_type: String,
    pub product_id: String,
    pub quantity: u32,
    pub unit_price: f64,
    pub total_price: f64,
}

/// A completed order
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Order {
    pub order_id: String,
    pub user_id: String,
    pub session_id: String,
    pub order_time: String,
    pub total_items: u32,
    pub subtotal: f64,
    pub discount: f64,
    pub tax: f64,
    pub shipping: f64,
    pub total: f64,
    pub payment_method: String,
    pub status: String,
}

/// Order line item
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderItem {
    pub order_id: String,
    pub product_id: String,
    pub quantity: u32,
    pub unit_price: f64,
    pub discount: f64,
    pub total: f64,
}

/// Customer with RFM metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Customer {
    pub customer_id: String,
    pub email: String,
    pub first_order_date: Option<String>,
    pub last_order_date: Option<String>,
    pub total_orders: u32,
    pub total_spent: f64,
    pub avg_order_value: f64,
    // RFM metrics
    pub rfm_recency: u32,
    pub rfm_frequency: u32,
    pub rfm_monetary: f64,
    pub rfm_score: String,
    pub rfm_segment: String,
}

/// Funnel event for conversion tracking
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunnelEvent {
    pub event_id: String,
    pub session_id: String,
    pub user_id: String,
    pub timestamp: String,
    pub stage: String,
    pub stage_number: u32,
    pub time_in_stage_seconds: u32,
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    }
}

fn generate_id<R: Rng>(rng: &mut R, prefix: &str) -> String {
    format!("{}-{:08x}", prefix, rng.gen::<u32>())
}

fn generate_email<R: Rng>(rng: &mut R) -> String {
    let names = &[
        "john", "jane", "mike", "emma", "alex", "sarah", "chris", "lisa",
    ];
    let domains = &["gmail.com", "yahoo.com", "outlook.com", "email.com"];
    format!(
        "{}{}@{}",
        names.choose(rng).unwrap(),
        rng.gen_range(1..9999),
        domains.choose(rng).unwrap()
    )
}

fn parse_start_date(date_str: &Option<String>) -> NaiveDateTime {
    if let Some(ref s) = date_str {
        NaiveDateTime::parse_from_str(&format!("{} 00:00:00", s), "%Y-%m-%d %H:%M:%S")
            .unwrap_or_else(|_| Utc::now().naive_utc())
    } else {
        Utc::now().naive_utc() - Duration::days(30)
    }
}

fn weighted_choice<'a, R: Rng>(rng: &mut R, items: &[&'a str], weights: &[f64]) -> &'a str {
    let total: f64 = weights.iter().sum();
    let roll = rng.gen::<f64>() * total;
    let mut cumulative = 0.0;

    for (i, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if roll < cumulative {
            return items[i];
        }
    }
    items[0]
}

/// Build a session state transition matrix for MarkovChain
/// Ensures all rows sum to 1.0
fn build_session_transition_matrix(config: &SessionConfig) -> Vec<Vec<f64>> {
    // States: landing, browse, view_product, add_to_cart, view_cart,
    //         checkout_start, checkout_payment, purchase, exit
    // indices:  0        1         2              3            4
    //           5               6                 7        8

    let bounce = if config.enable_bounces {
        config.bounce_rate
    } else {
        0.0
    };

    let cart_prob = config.cart_add_probability;
    let checkout_prob = config.checkout_start_probability;
    let purchase_prob = config.purchase_completion_probability;

    // Helper to normalize a row to sum to 1.0
    fn normalize(row: Vec<f64>) -> Vec<f64> {
        let sum: f64 = row.iter().sum();
        if sum == 0.0 {
            // If all zeros, put all probability on last state (exit)
            let mut result = vec![0.0; row.len()];
            result[row.len() - 1] = 1.0;
            result
        } else {
            row.iter().map(|x| x / sum).collect()
        }
    }

    vec![
        // From landing: bounce or continue browsing
        normalize(vec![
            0.0,                           // stay at landing
            0.5 * (1.0 - bounce),          // browse
            0.3 * (1.0 - bounce),          // view_product
            0.0,                           // add_to_cart
            0.0,                           // view_cart
            0.0,                           // checkout_start
            0.0,                           // checkout_payment
            0.0,                           // purchase
            bounce + 0.2 * (1.0 - bounce), // exit
        ]),
        // From browse
        normalize(vec![0.0, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]),
        // From view_product
        normalize(vec![
            0.0,              // landing
            0.25,             // browse (continue shopping)
            0.20,             // view_product (view another)
            cart_prob,        // add_to_cart
            0.0,              // view_cart
            0.0,              // checkout_start
            0.0,              // checkout_payment
            0.0,              // purchase
            0.55 - cart_prob, // exit
        ]),
        // From add_to_cart
        normalize(vec![0.0, 0.1, 0.2, 0.1, 0.4, 0.0, 0.0, 0.0, 0.2]),
        // From view_cart
        normalize(vec![
            0.0,                 // landing
            0.1,                 // browse
            0.15,                // view_product
            0.05,                // add_to_cart
            0.1,                 // view_cart
            checkout_prob,       // checkout_start
            0.0,                 // checkout_payment
            0.0,                 // purchase
            0.6 - checkout_prob, // exit
        ]),
        // From checkout_start
        normalize(vec![
            0.0,                        // landing
            0.0,                        // browse
            0.0,                        // view_product
            0.0,                        // add_to_cart
            0.15,                       // view_cart (go back)
            0.0,                        // checkout_start
            purchase_prob * 0.9,        // checkout_payment
            0.0,                        // purchase
            0.85 - purchase_prob * 0.9, // exit (abandon)
        ]),
        // From checkout_payment
        normalize(vec![
            0.0,                  // landing
            0.0,                  // browse
            0.0,                  // view_product
            0.0,                  // add_to_cart
            0.05,                 // view_cart
            0.0,                  // checkout_start
            0.0,                  // checkout_payment
            purchase_prob,        // purchase
            0.95 - purchase_prob, // exit
        ]),
        // From purchase (terminal state -> exit)
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        // From exit (absorbing state)
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
}

/// Calculate RFM score bucket (1-5 scale)
fn rfm_bucket(value: f64, min: f64, max: f64, num_buckets: u32, invert: bool) -> u32 {
    if max <= min {
        return num_buckets;
    }
    let normalized = ((value - min) / (max - min)).clamp(0.0, 1.0);
    let bucket = (normalized * (num_buckets as f64)).floor() as u32;
    let bucket = bucket.min(num_buckets - 1) + 1;
    if invert {
        num_buckets + 1 - bucket
    } else {
        bucket
    }
}

/// Determine RFM segment based on scores
fn get_rfm_segment(r: u32, f: u32, m: u32) -> &'static str {
    // Simplified RFM segmentation logic
    let _avg = (r + f + m) as f64 / 3.0;

    if r >= 4 && f >= 4 && m >= 4 {
        "Champions"
    } else if f >= 4 && r >= 3 {
        "Loyal Customers"
    } else if r >= 4 && f <= 2 {
        "Recent Customers"
    } else if r >= 3 && f >= 2 && m >= 2 {
        "Potential Loyalists"
    } else if r >= 3 && f <= 2 && m <= 2 {
        "Promising"
    } else if r == 2 && f >= 2 {
        "Customers Needing Attention"
    } else if r == 2 && f <= 2 {
        "About to Sleep"
    } else if r == 1 && f >= 3 && m >= 3 {
        "Can't Lose Them"
    } else if r == 1 && f >= 2 {
        "At Risk"
    } else {
        "Lost"
    }
}

// =============================================================================
// Product Catalog Generator
// =============================================================================

/// Generate a product catalog
pub fn generate_catalog(config: &EcommerceConfig) -> Vec<Product> {
    let mut rng = create_rng(config.seed);
    let mut products = Vec::with_capacity(config.catalog.num_products);

    let categories = if config.catalog.categories.is_empty() {
        PRODUCT_CATEGORIES.iter().map(|s| s.to_string()).collect()
    } else {
        config.catalog.categories.clone()
    };

    let subcategories: Vec<Vec<&str>> = vec![
        vec!["Phones", "Laptops", "Tablets", "Accessories", "Cameras"],
        vec!["Men's", "Women's", "Kids", "Shoes", "Accessories"],
        vec!["Furniture", "Decor", "Kitchen", "Garden", "Bedding"],
        vec!["Fitness", "Outdoor", "Team Sports", "Water Sports", "Gear"],
        vec!["Skincare", "Makeup", "Hair Care", "Fragrance", "Tools"],
        vec![
            "Fiction",
            "Non-Fiction",
            "Children's",
            "Textbooks",
            "Comics",
        ],
        vec!["Games", "Dolls", "Building", "Outdoor", "Educational"],
        vec!["Snacks", "Beverages", "Organic", "International", "Pantry"],
    ];

    // Price distribution
    let ln_mean = ((config.catalog.max_price + config.catalog.min_price) / 2.0).ln();
    let ln_std = 1.0;
    let price_dist = LogNormal::new(ln_mean - ln_std * ln_std / 2.0, ln_std).unwrap();

    for i in 0..config.catalog.num_products {
        let cat_idx = rng.gen_range(0..categories.len());
        let category = categories[cat_idx].clone();
        let subcats = &subcategories[cat_idx % subcategories.len()];
        let subcategory = subcats.choose(&mut rng).unwrap().to_string();

        let price = if config.catalog.lognormal_prices {
            price_dist
                .sample(&mut rng)
                .clamp(config.catalog.min_price, config.catalog.max_price)
        } else {
            rng.gen_range(config.catalog.min_price..config.catalog.max_price)
        };

        // Round to common price points
        let price = (price * 100.0).round() / 100.0;
        let price = if price > 10.0 {
            price.floor() + 0.99
        } else {
            price
        };

        products.push(Product {
            product_id: format!("PROD-{:06}", i + 1),
            name: format!("{} {} Item {}", category, subcategory, i + 1),
            category,
            subcategory,
            price,
            rating: 3.0 + rng.gen::<f64>() * 2.0,
            review_count: rng.gen_range(0..5000),
            in_stock: rng.gen::<f64>() > 0.05,
        });
    }

    products
}

// =============================================================================
// Session Generator
// =============================================================================

/// Generate user sessions with MarkovChain-based navigation
pub fn generate_sessions(config: &EcommerceConfig) -> Vec<Session> {
    let mut rng = create_rng(config.seed);
    let mut sessions = Vec::with_capacity(config.sessions);

    let start_time = parse_start_date(&config.start_date);
    let transition_matrix = build_session_transition_matrix(&config.session);
    let states: Vec<String> = SESSION_STATES.iter().map(|s| s.to_string()).collect();
    let mut mc = MarkovChain::new(transition_matrix, states).unwrap();

    // Pre-generate customer IDs
    let customer_ids: Vec<String> = (0..config.customers)
        .map(|i| format!("CUST-{:06}", i + 1))
        .collect();

    let time_dist =
        Exp::new(1.0 / (config.days as f64 * 86400.0 / config.sessions as f64)).unwrap();

    let mut current_time = start_time;

    for _i in 0..config.sessions {
        let session_id = generate_id(&mut rng, "SES");
        let user_id = customer_ids.choose(&mut rng).unwrap().clone();

        // Time of session
        let time_delta = time_dist.sample(&mut rng) as i64;
        current_time = current_time + Duration::seconds(time_delta);

        let device = weighted_choice(&mut rng, DEVICE_TYPES, DEVICE_WEIGHTS);
        let browser = weighted_choice(&mut rng, BROWSERS, BROWSER_WEIGHTS);
        let traffic_source = weighted_choice(&mut rng, TRAFFIC_SOURCES, TRAFFIC_SOURCE_WEIGHTS);
        let landing = LANDING_PAGES.choose(&mut rng).unwrap();

        // Simulate session via MarkovChain
        let mut pages_viewed = 1u32;
        let mut converted = false;
        let mut total_value = 0.0;

        // Reset to landing state for each session
        mc.set_state(0).unwrap();

        // Check for immediate bounce
        let bounced =
            config.session.enable_bounces && rng.gen::<f64>() < config.session.bounce_rate;

        if !bounced {
            // Simulate navigation
            let max_steps = 50;
            for _ in 0..max_steps {
                let state_name = mc.next(&mut rng).to_string();
                pages_viewed += 1;

                if state_name == "purchase" {
                    converted = true;
                    // Generate order value
                    total_value = 20.0 + rng.gen::<f64>() * 200.0;
                    break;
                }
                if state_name == "exit" {
                    break;
                }
            }
        }

        let duration = if bounced {
            rng.gen_range(5..30)
        } else {
            let base = config.session.avg_session_duration_seconds as f64;
            (base * (0.5 + rng.gen::<f64>())).round() as u32
        };

        let end_time = current_time + Duration::seconds(duration as i64);

        sessions.push(Session {
            session_id,
            user_id,
            start_time: current_time.format("%Y-%m-%d %H:%M:%S").to_string(),
            end_time: end_time.format("%Y-%m-%d %H:%M:%S").to_string(),
            duration_seconds: duration,
            device_type: device.to_string(),
            browser: browser.to_string(),
            traffic_source: traffic_source.to_string(),
            landing_page: landing.to_string(),
            pages_viewed,
            bounced,
            converted,
            total_value,
        });
    }

    sessions
}

// =============================================================================
// Cart Events Generator
// =============================================================================

/// Generate cart events based on sessions
pub fn generate_cart_events(
    sessions: &[Session],
    products: &[Product],
    config: &EcommerceConfig,
) -> Vec<CartEvent> {
    let mut rng = create_rng(config.seed.map(|s| s + 1));
    let mut events = Vec::new();

    for session in sessions {
        // Skip bounced sessions
        if session.bounced {
            continue;
        }

        // Probability of cart activity
        if rng.gen::<f64>() > config.session.cart_add_probability * 2.0 {
            continue;
        }

        let num_items = (config.cart.avg_items_per_cart * (0.5 + rng.gen::<f64>())).round() as u32;
        let num_items = num_items.min(config.cart.max_items).max(1);

        let session_start =
            NaiveDateTime::parse_from_str(&session.start_time, "%Y-%m-%d %H:%M:%S").unwrap();
        let mut current_time = session_start;

        for _ in 0..num_items {
            let product = products.choose(&mut rng).unwrap();
            let quantity = rng.gen_range(1..=3);

            current_time = current_time + Duration::seconds(rng.gen_range(10..120));

            events.push(CartEvent {
                event_id: generate_id(&mut rng, "EVT"),
                session_id: session.session_id.clone(),
                user_id: session.user_id.clone(),
                timestamp: current_time.format("%Y-%m-%d %H:%M:%S").to_string(),
                event_type: "add_to_cart".to_string(),
                product_id: product.product_id.clone(),
                quantity,
                unit_price: product.price,
                total_price: product.price * quantity as f64,
            });

            // Possible remove
            if rng.gen::<f64>() < config.cart.remove_probability {
                current_time = current_time + Duration::seconds(rng.gen_range(30..180));
                events.push(CartEvent {
                    event_id: generate_id(&mut rng, "EVT"),
                    session_id: session.session_id.clone(),
                    user_id: session.user_id.clone(),
                    timestamp: current_time.format("%Y-%m-%d %H:%M:%S").to_string(),
                    event_type: "remove_from_cart".to_string(),
                    product_id: product.product_id.clone(),
                    quantity,
                    unit_price: product.price,
                    total_price: product.price * quantity as f64,
                });
            }
        }

        // Checkout events for converted sessions
        if session.converted {
            current_time = current_time + Duration::seconds(rng.gen_range(30..120));
            events.push(CartEvent {
                event_id: generate_id(&mut rng, "EVT"),
                session_id: session.session_id.clone(),
                user_id: session.user_id.clone(),
                timestamp: current_time.format("%Y-%m-%d %H:%M:%S").to_string(),
                event_type: "checkout_start".to_string(),
                product_id: "".to_string(),
                quantity: 0,
                unit_price: 0.0,
                total_price: session.total_value,
            });

            current_time = current_time + Duration::seconds(rng.gen_range(60..300));
            events.push(CartEvent {
                event_id: generate_id(&mut rng, "EVT"),
                session_id: session.session_id.clone(),
                user_id: session.user_id.clone(),
                timestamp: current_time.format("%Y-%m-%d %H:%M:%S").to_string(),
                event_type: "checkout_complete".to_string(),
                product_id: "".to_string(),
                quantity: 0,
                unit_price: 0.0,
                total_price: session.total_value,
            });
        } else if !events.is_empty() && rng.gen::<f64>() < config.cart.abandonment_rate {
            // Abandoned cart
            current_time = current_time + Duration::seconds(rng.gen_range(300..1800));
            events.push(CartEvent {
                event_id: generate_id(&mut rng, "EVT"),
                session_id: session.session_id.clone(),
                user_id: session.user_id.clone(),
                timestamp: current_time.format("%Y-%m-%d %H:%M:%S").to_string(),
                event_type: "cart_abandoned".to_string(),
                product_id: "".to_string(),
                quantity: 0,
                unit_price: 0.0,
                total_price: 0.0,
            });
        }
    }

    events
}

// =============================================================================
// Orders Generator
// =============================================================================

/// Generate orders from converted sessions
pub fn generate_orders(sessions: &[Session], config: &EcommerceConfig) -> Vec<Order> {
    let mut rng = create_rng(config.seed.map(|s| s + 2));
    let mut orders = Vec::new();

    let payment_methods = &[
        "credit_card",
        "debit_card",
        "paypal",
        "apple_pay",
        "google_pay",
        "bank_transfer",
    ];
    let payment_weights = &[0.40, 0.20, 0.15, 0.10, 0.10, 0.05];

    for session in sessions.iter().filter(|s| s.converted) {
        let items = rng.gen_range(1..=5);
        let subtotal = session.total_value;
        let discount = if rng.gen::<f64>() < 0.3 {
            subtotal * rng.gen_range(0.05..0.20)
        } else {
            0.0
        };
        let tax = (subtotal - discount) * 0.08;
        let shipping: f64 = if subtotal > 50.0 && rng.gen::<f64>() > 0.3 {
            0.0
        } else {
            rng.gen_range(5.0..15.0)
        };

        let payment = weighted_choice(&mut rng, payment_methods, payment_weights);

        orders.push(Order {
            order_id: generate_id(&mut rng, "ORD"),
            user_id: session.user_id.clone(),
            session_id: session.session_id.clone(),
            order_time: session.end_time.clone(),
            total_items: items,
            subtotal,
            discount: (discount * 100.0).round() / 100.0,
            tax: (tax * 100.0).round() / 100.0,
            shipping: (shipping * 100.0).round() / 100.0,
            total: ((subtotal - discount + tax + shipping) * 100.0).round() / 100.0,
            payment_method: payment.to_string(),
            status: "completed".to_string(),
        });
    }

    orders
}

// =============================================================================
// Customer RFM Generator
// =============================================================================

/// Generate customers with RFM metrics
pub fn generate_customers(orders: &[Order], config: &EcommerceConfig) -> Vec<Customer> {
    let mut rng = create_rng(config.seed.map(|s| s + 3));

    // Aggregate order data by customer
    let mut customer_data: std::collections::HashMap<String, (Vec<&Order>, f64)> =
        std::collections::HashMap::new();

    for order in orders {
        let entry = customer_data
            .entry(order.user_id.clone())
            .or_insert((Vec::new(), 0.0));
        entry.0.push(order);
        entry.1 += order.total;
    }

    let now = Utc::now().naive_utc();
    let mut customers = Vec::new();

    // Calculate RFM buckets
    let mut recencies: Vec<i64> = Vec::new();
    let mut frequencies: Vec<u32> = Vec::new();
    let mut monetaries: Vec<f64> = Vec::new();

    for (_, (orders_list, total)) in &customer_data {
        let last_order = orders_list
            .iter()
            .filter_map(|o| NaiveDateTime::parse_from_str(&o.order_time, "%Y-%m-%d %H:%M:%S").ok())
            .max();

        if let Some(last) = last_order {
            recencies.push((now - last).num_days());
        }
        frequencies.push(orders_list.len() as u32);
        monetaries.push(*total);
    }

    let r_min = *recencies.iter().min().unwrap_or(&0) as f64;
    let r_max = *recencies.iter().max().unwrap_or(&365) as f64;
    let f_min = *frequencies.iter().min().unwrap_or(&0) as f64;
    let f_max = *frequencies.iter().max().unwrap_or(&10) as f64;
    let m_min = monetaries.iter().cloned().fold(f64::INFINITY, f64::min);
    let m_max = monetaries.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    for (user_id, (orders_list, total_spent)) in customer_data {
        let first_order = orders_list
            .iter()
            .filter_map(|o| NaiveDateTime::parse_from_str(&o.order_time, "%Y-%m-%d %H:%M:%S").ok())
            .min()
            .map(|d| d.format("%Y-%m-%d").to_string());

        let last_order = orders_list
            .iter()
            .filter_map(|o| NaiveDateTime::parse_from_str(&o.order_time, "%Y-%m-%d %H:%M:%S").ok())
            .max();

        let recency_days = last_order.map(|d| (now - d).num_days()).unwrap_or(365) as u32;

        let frequency = orders_list.len() as u32;
        let avg_order_value = if frequency > 0 {
            total_spent / frequency as f64
        } else {
            0.0
        };

        let r_score = rfm_bucket(
            recency_days as f64,
            r_min,
            r_max,
            config.rfm.num_buckets,
            true,
        );
        let f_score = rfm_bucket(
            frequency as f64,
            f_min,
            f_max,
            config.rfm.num_buckets,
            false,
        );
        let m_score = rfm_bucket(total_spent, m_min, m_max, config.rfm.num_buckets, false);

        let rfm_score = format!("{}{}{}", r_score, f_score, m_score);
        let rfm_segment = get_rfm_segment(r_score, f_score, m_score).to_string();

        customers.push(Customer {
            customer_id: user_id.clone(),
            email: generate_email(&mut rng),
            first_order_date: first_order,
            last_order_date: last_order.map(|d| d.format("%Y-%m-%d").to_string()),
            total_orders: frequency,
            total_spent: (total_spent * 100.0).round() / 100.0,
            avg_order_value: (avg_order_value * 100.0).round() / 100.0,
            rfm_recency: recency_days,
            rfm_frequency: frequency,
            rfm_monetary: total_spent,
            rfm_score,
            rfm_segment,
        });
    }

    customers
}

// =============================================================================
// Funnel Events Generator
// =============================================================================

/// Generate conversion funnel events
pub fn generate_funnel_events(sessions: &[Session], config: &EcommerceConfig) -> Vec<FunnelEvent> {
    let mut rng = create_rng(config.seed.map(|s| s + 4));
    let mut events = Vec::new();

    let stages = if config.funnel.stages.is_empty() {
        vec![
            "visit",
            "view_product",
            "add_to_cart",
            "checkout",
            "purchase",
        ]
    } else {
        config
            .funnel
            .stages
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
    };

    for session in sessions {
        let session_start =
            NaiveDateTime::parse_from_str(&session.start_time, "%Y-%m-%d %H:%M:%S").unwrap();
        let mut current_time = session_start;

        // Determine how far into funnel based on session state
        let max_stage = if session.bounced {
            0
        } else if session.converted {
            stages.len() - 1
        } else {
            // Based on pages viewed, estimate stage
            let stage_estimate = (session.pages_viewed as f64 / 2.0).floor() as usize;
            stage_estimate.min(stages.len() - 2).max(1)
        };

        for (idx, &stage) in stages.iter().enumerate() {
            if idx > max_stage {
                break;
            }

            let time_in_stage = rng.gen_range(10..120);
            events.push(FunnelEvent {
                event_id: generate_id(&mut rng, "FNL"),
                session_id: session.session_id.clone(),
                user_id: session.user_id.clone(),
                timestamp: current_time.format("%Y-%m-%d %H:%M:%S").to_string(),
                stage: stage.to_string(),
                stage_number: idx as u32,
                time_in_stage_seconds: time_in_stage,
            });

            current_time = current_time + Duration::seconds(time_in_stage as i64);
        }
    }

    events
}

// =============================================================================
// Main Generator Functions
// =============================================================================

/// Generate complete e-commerce dataset
pub fn ecommerce(config: &EcommerceConfig) -> EcommerceData {
    let products = generate_catalog(config);
    let sessions = generate_sessions(config);
    let cart_events = generate_cart_events(&sessions, &products, config);
    let orders = generate_orders(&sessions, config);
    let customers = generate_customers(&orders, config);
    let funnel_events = if config.funnel.enable {
        generate_funnel_events(&sessions, config)
    } else {
        Vec::new()
    };

    EcommerceData {
        products,
        sessions,
        cart_events,
        orders,
        customers,
        funnel_events,
    }
}

/// Complete e-commerce dataset
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EcommerceData {
    pub products: Vec<Product>,
    pub sessions: Vec<Session>,
    pub cart_events: Vec<CartEvent>,
    pub orders: Vec<Order>,
    pub customers: Vec<Customer>,
    pub funnel_events: Vec<FunnelEvent>,
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Generate sessions only
pub fn sessions(count: usize, seed: Option<u64>) -> Vec<Session> {
    let config = EcommerceConfig {
        sessions: count,
        seed,
        ..Default::default()
    };
    generate_sessions(&config)
}

/// Generate product catalog only
pub fn products(count: usize, seed: Option<u64>) -> Vec<Product> {
    let config = EcommerceConfig {
        seed,
        catalog: CatalogConfig {
            num_products: count,
            ..Default::default()
        },
        ..Default::default()
    };
    generate_catalog(&config)
}
