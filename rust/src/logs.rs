//! Web server and application logs generator module.
//!
//! Generates realistic log data with:
//! - Poisson-driven request arrival times
//! - HTTP status code sequences via Markov chains
//! - Log-normal latency distributions with outliers
//! - Error bursts via clustered Poisson
//! - Realistic user agents, IPs, and request paths

use chrono::{Duration, NaiveDateTime, Utc};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp, LogNormal};
use serde::{Deserialize, Serialize};

use crate::temporal::MarkovChain;

// =============================================================================
// Constants
// =============================================================================

const USER_AGENTS: &[&str] = &[
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
    "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/120.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "curl/8.4.0",
    "python-requests/2.31.0",
    "Go-http-client/2.0",
    "PostmanRuntime/7.35.0",
    "Apache-HttpClient/4.5.14",
];

const HTTP_METHODS: &[&str] = &["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"];
const HTTP_METHOD_WEIGHTS: &[f64] = &[0.65, 0.20, 0.05, 0.03, 0.02, 0.03, 0.02];

const LOG_LEVELS: &[&str] = &["DEBUG", "INFO", "WARN", "ERROR", "FATAL"];

// Common request paths by category
const API_PATHS: &[&str] = &[
    "/api/v1/users",
    "/api/v1/users/{id}",
    "/api/v1/products",
    "/api/v1/products/{id}",
    "/api/v1/orders",
    "/api/v1/orders/{id}",
    "/api/v1/cart",
    "/api/v1/checkout",
    "/api/v1/search",
    "/api/v1/auth/login",
    "/api/v1/auth/logout",
    "/api/v1/auth/refresh",
    "/api/v2/graphql",
    "/health",
    "/metrics",
    "/ready",
];

const STATIC_PATHS: &[&str] = &[
    "/static/js/main.js",
    "/static/css/styles.css",
    "/static/images/logo.png",
    "/favicon.ico",
    "/robots.txt",
    "/sitemap.xml",
];

// =============================================================================
// Configuration
// =============================================================================

/// HTTP log format type
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    /// Combined Log Format (Apache/nginx style)
    Combined,
    /// Common Log Format
    Common,
    /// JSON structured logs
    Json,
    /// Application event logs
    Application,
}

impl Default for LogFormat {
    fn default() -> Self {
        LogFormat::Combined
    }
}

/// Configuration for error bursts
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErrorBurstConfig {
    pub enable: bool,
    pub burst_probability: f64,
    pub burst_duration_seconds: u32,
    pub burst_error_rate: f64,
}

impl Default for ErrorBurstConfig {
    fn default() -> Self {
        Self {
            enable: true,
            burst_probability: 0.02,
            burst_duration_seconds: 30,
            burst_error_rate: 0.8,
        }
    }
}

/// Configuration for latency simulation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatencyConfig {
    pub base_latency_ms: f64,
    pub latency_stddev: f64,
    pub slow_request_probability: f64,
    pub slow_request_multiplier: f64,
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            base_latency_ms: 50.0,
            latency_stddev: 0.8,
            slow_request_probability: 0.05,
            slow_request_multiplier: 10.0,
        }
    }
}

/// Configuration for logs generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogsConfig {
    pub count: usize,
    pub seed: Option<u64>,
    pub start_time: Option<String>,
    pub format: LogFormat,
    pub requests_per_second: f64,
    pub success_rate: f64,
    pub error_burst: ErrorBurstConfig,
    pub latency: LatencyConfig,
    pub include_user_agent: bool,
    pub include_referer: bool,
    pub unique_ips: usize,
    pub unique_users: usize,
    pub api_path_ratio: f64,
}

impl Default for LogsConfig {
    fn default() -> Self {
        Self {
            count: 1000,
            seed: None,
            start_time: None,
            format: LogFormat::Combined,
            requests_per_second: 100.0,
            success_rate: 0.95,
            error_burst: ErrorBurstConfig::default(),
            latency: LatencyConfig::default(),
            include_user_agent: true,
            include_referer: true,
            unique_ips: 1000,
            unique_users: 500,
            api_path_ratio: 0.7,
        }
    }
}

// =============================================================================
// Log Entry Structures
// =============================================================================

/// A single web server log entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub ip_address: String,
    pub user_id: Option<String>,
    pub method: String,
    pub path: String,
    pub status_code: u16,
    pub response_bytes: u32,
    pub latency_ms: f64,
    pub user_agent: Option<String>,
    pub referer: Option<String>,
}

/// Application event log entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppLogEntry {
    pub timestamp: String,
    pub level: String,
    pub logger: String,
    pub message: String,
    pub thread_id: u32,
    pub trace_id: Option<String>,
    pub span_id: Option<String>,
    pub exception: Option<String>,
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

fn generate_ip_pool(rng: &mut StdRng, count: usize) -> Vec<String> {
    (0..count)
        .map(|_| {
            format!(
                "{}.{}.{}.{}",
                rng.gen_range(1..255),
                rng.gen_range(0..256),
                rng.gen_range(0..256),
                rng.gen_range(1..255)
            )
        })
        .collect()
}

fn generate_user_pool(rng: &mut StdRng, count: usize) -> Vec<String> {
    (0..count)
        .map(|_| format!("user_{:06}", rng.gen_range(100000..999999)))
        .collect()
}

fn weighted_choice<R: Rng>(rng: &mut R, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    let mut r = rng.gen::<f64>() * total;
    for (i, w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

fn create_status_code_chain() -> MarkovChain {
    // States: 200, 201, 204, 301, 302, 400, 401, 403, 404, 500, 502, 503
    // Transition probabilities favor staying in success states
    let states = vec![
        "200".to_string(),
        "201".to_string(),
        "204".to_string(),
        "301".to_string(),
        "302".to_string(),
        "400".to_string(),
        "401".to_string(),
        "403".to_string(),
        "404".to_string(),
        "500".to_string(),
        "502".to_string(),
        "503".to_string(),
    ];

    // Transition matrix: rows are from-state, cols are to-state
    // Heavy bias toward 200 (success)
    let matrix = vec![
        vec![
            0.85, 0.03, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.005, 0.005,
        ], // from 200
        vec![
            0.80, 0.10, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.005, 0.0025, 0.0025,
        ], // from 201
        vec![
            0.85, 0.05, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.005, 0.0025, 0.0025,
        ], // from 204
        vec![
            0.70, 0.02, 0.01, 0.15, 0.05, 0.02, 0.01, 0.01, 0.02, 0.005, 0.0025, 0.0025,
        ], // from 301
        vec![
            0.70, 0.02, 0.01, 0.05, 0.15, 0.02, 0.01, 0.01, 0.02, 0.005, 0.0025, 0.0025,
        ], // from 302
        vec![
            0.60, 0.02, 0.01, 0.01, 0.01, 0.25, 0.02, 0.02, 0.04, 0.01, 0.005, 0.005,
        ], // from 400
        vec![
            0.50, 0.02, 0.01, 0.01, 0.01, 0.05, 0.30, 0.05, 0.03, 0.01, 0.005, 0.005,
        ], // from 401
        vec![
            0.50, 0.02, 0.01, 0.01, 0.01, 0.05, 0.05, 0.30, 0.03, 0.01, 0.005, 0.005,
        ], // from 403
        vec![
            0.60, 0.02, 0.01, 0.02, 0.02, 0.05, 0.02, 0.02, 0.20, 0.02, 0.01, 0.01,
        ], // from 404
        vec![
            0.40, 0.02, 0.01, 0.01, 0.01, 0.05, 0.02, 0.02, 0.05, 0.30, 0.05, 0.06,
        ], // from 500
        vec![
            0.40, 0.02, 0.01, 0.01, 0.01, 0.03, 0.02, 0.02, 0.03, 0.10, 0.25, 0.10,
        ], // from 502
        vec![
            0.40, 0.02, 0.01, 0.01, 0.01, 0.03, 0.02, 0.02, 0.03, 0.10, 0.10, 0.25,
        ], // from 503
    ];

    MarkovChain::new(matrix, states).expect("Invalid Markov chain")
}

fn generate_path<R: Rng>(rng: &mut R, api_ratio: f64) -> String {
    if rng.gen::<f64>() < api_ratio {
        let path = API_PATHS.choose(rng).unwrap_or(&"/api/v1/users");
        // Replace {id} placeholders with random IDs
        path.replace("{id}", &format!("{}", rng.gen_range(1..10000)))
    } else {
        STATIC_PATHS
            .choose(rng)
            .unwrap_or(&"/static/js/main.js")
            .to_string()
    }
}

fn generate_latency<R: Rng>(rng: &mut R, config: &LatencyConfig) -> f64 {
    // Use log-normal for realistic latency distribution
    let mu = config.base_latency_ms.ln();
    let sigma = config.latency_stddev;
    let dist = LogNormal::new(mu, sigma).unwrap_or_else(|_| LogNormal::new(3.9, 0.8).unwrap());
    let mut latency: f64 = dist.sample(rng);

    // Occasional slow requests
    if rng.gen::<f64>() < config.slow_request_probability {
        latency *= config.slow_request_multiplier;
    }

    latency.max(1.0).min(30000.0) // Clamp between 1ms and 30s
}

fn generate_response_bytes<R: Rng>(rng: &mut R, status: u16, path: &str) -> u32 {
    match status {
        204 => 0,
        301 | 302 => rng.gen_range(0..200),
        400..=499 => rng.gen_range(100..1000),
        500..=599 => rng.gen_range(100..500),
        _ => {
            // Response size depends on path type
            if path.contains("/static/") || path.contains(".js") || path.contains(".css") {
                rng.gen_range(1000..500000)
            } else if path.contains("/api/") {
                rng.gen_range(100..50000)
            } else {
                rng.gen_range(500..10000)
            }
        }
    }
}

// =============================================================================
// Main Generator
// =============================================================================

/// Generate web server log entries
pub fn generate_logs(config: &LogsConfig) -> Vec<LogEntry> {
    let mut rng = create_rng(config.seed);
    let ip_pool = generate_ip_pool(&mut rng, config.unique_ips);
    let user_pool = generate_user_pool(&mut rng, config.unique_users);
    let mut status_chain = create_status_code_chain();

    // Parse start time or use now
    let start = config
        .start_time
        .as_ref()
        .and_then(|s| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").ok())
        .unwrap_or_else(|| Utc::now().naive_utc() - Duration::hours(1));

    // Inter-arrival time distribution (exponential for Poisson process)
    let mean_interval_ms = 1000.0 / config.requests_per_second;
    let exp_dist = Exp::new(1.0 / mean_interval_ms).unwrap_or_else(|_| Exp::new(0.01).unwrap());

    let mut current_time = start;
    let mut entries = Vec::with_capacity(config.count);

    // Error burst state
    let mut in_burst = false;
    let mut burst_end_time = start;

    for _ in 0..config.count {
        // Advance time using exponential inter-arrival
        let interval_ms: f64 = exp_dist.sample(&mut rng);
        current_time += Duration::milliseconds(interval_ms as i64);

        // Check for error burst
        if config.error_burst.enable {
            if !in_burst && rng.gen::<f64>() < config.error_burst.burst_probability / 100.0 {
                in_burst = true;
                burst_end_time = current_time
                    + Duration::seconds(config.error_burst.burst_duration_seconds as i64);
            }
            if in_burst && current_time > burst_end_time {
                in_burst = false;
            }
        }

        // Generate status code
        let status_str = if in_burst && rng.gen::<f64>() < config.error_burst.burst_error_rate {
            // During burst, mostly 5xx errors
            *["500", "502", "503"].choose(&mut rng).unwrap_or(&"500")
        } else if rng.gen::<f64>() > config.success_rate {
            // Normal error rate
            status_chain.next(&mut rng)
        } else {
            status_chain.next(&mut rng)
        };
        let status_code: u16 = status_str.parse().unwrap_or(200);

        // Generate other fields
        let method_idx = weighted_choice(&mut rng, HTTP_METHOD_WEIGHTS);
        let method = HTTP_METHODS[method_idx].to_string();
        let path = generate_path(&mut rng, config.api_path_ratio);
        let latency = generate_latency(&mut rng, &config.latency);
        let response_bytes = generate_response_bytes(&mut rng, status_code, &path);

        let ip = ip_pool.choose(&mut rng).cloned().unwrap_or_default();
        let user_id = if rng.gen::<f64>() < 0.6 {
            user_pool.choose(&mut rng).cloned()
        } else {
            None
        };

        let user_agent = if config.include_user_agent {
            USER_AGENTS.choose(&mut rng).map(|s| s.to_string())
        } else {
            None
        };

        let referer = if config.include_referer && rng.gen::<f64>() < 0.3 {
            Some(format!(
                "https://example.com{}",
                API_PATHS.choose(&mut rng).unwrap_or(&"/")
            ))
        } else {
            None
        };

        entries.push(LogEntry {
            timestamp: current_time.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(),
            ip_address: ip,
            user_id,
            method,
            path,
            status_code,
            response_bytes,
            latency_ms: (latency * 100.0).round() / 100.0,
            user_agent,
            referer,
        });
    }

    entries
}

/// Generate application event logs
pub fn generate_app_logs(config: &LogsConfig) -> Vec<AppLogEntry> {
    let mut rng = create_rng(config.seed);

    let start = config
        .start_time
        .as_ref()
        .and_then(|s| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").ok())
        .unwrap_or_else(|| Utc::now().naive_utc() - Duration::hours(1));

    let mean_interval_ms = 1000.0 / config.requests_per_second;
    let exp_dist = Exp::new(1.0 / mean_interval_ms).unwrap_or_else(|_| Exp::new(0.01).unwrap());

    let mut current_time = start;
    let mut entries = Vec::with_capacity(config.count);

    let loggers = vec![
        "com.app.service.UserService",
        "com.app.service.OrderService",
        "com.app.controller.ApiController",
        "com.app.repository.UserRepository",
        "com.app.cache.RedisCache",
        "org.springframework.web.servlet.DispatcherServlet",
        "io.netty.channel.nio.NioEventLoop",
    ];

    let messages = vec![
        "Processing request",
        "Request completed successfully",
        "Cache hit for key",
        "Cache miss, fetching from database",
        "Database query executed",
        "Connection pool exhausted, waiting",
        "Retrying failed operation",
        "Circuit breaker opened",
        "Rate limit exceeded",
        "Authentication successful",
        "Authorization check passed",
        "Validation error",
        "Resource not found",
        "Internal error occurred",
        "Timeout waiting for response",
    ];

    let exceptions = vec![
        "java.lang.NullPointerException",
        "java.sql.SQLException: Connection timeout",
        "java.io.IOException: Connection reset",
        "java.lang.OutOfMemoryError: Java heap space",
        "org.springframework.dao.DataAccessException",
    ];

    // Level weights: DEBUG, INFO, WARN, ERROR, FATAL
    let level_weights = vec![0.1, 0.7, 0.12, 0.07, 0.01];

    for _ in 0..config.count {
        let interval_ms: f64 = exp_dist.sample(&mut rng);
        current_time += Duration::milliseconds(interval_ms as i64);

        let level_idx = weighted_choice(&mut rng, &level_weights);
        let level = LOG_LEVELS[level_idx].to_string();

        let exception = if level == "ERROR" || level == "FATAL" {
            if rng.gen::<f64>() < 0.7 {
                exceptions.choose(&mut rng).map(|s| s.to_string())
            } else {
                None
            }
        } else {
            None
        };

        let trace_id = if rng.gen::<f64>() < 0.8 {
            Some(format!("{:032x}", rng.gen::<u128>()))
        } else {
            None
        };

        let span_id = trace_id
            .as_ref()
            .map(|_| format!("{:016x}", rng.gen::<u64>()));

        entries.push(AppLogEntry {
            timestamp: current_time.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(),
            level,
            logger: loggers.choose(&mut rng).unwrap_or(&"app").to_string(),
            message: messages.choose(&mut rng).unwrap_or(&"Event").to_string(),
            thread_id: rng.gen_range(1..100),
            trace_id,
            span_id,
            exception,
        });
    }

    entries
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_logs_default() {
        let config = LogsConfig::default();
        let logs = generate_logs(&config);
        assert_eq!(logs.len(), 1000);
    }

    #[test]
    fn test_generate_logs_seeded() {
        let config = LogsConfig {
            count: 100,
            seed: Some(42),
            start_time: Some("2024-01-01 00:00:00".to_string()),
            ..Default::default()
        };
        let logs1 = generate_logs(&config);
        let logs2 = generate_logs(&config);
        assert_eq!(logs1.len(), logs2.len());
        assert_eq!(logs1[0].timestamp, logs2[0].timestamp);
        assert_eq!(logs1[0].ip_address, logs2[0].ip_address);
    }

    #[test]
    fn test_generate_logs_status_codes() {
        let config = LogsConfig {
            count: 1000,
            seed: Some(42),
            success_rate: 0.9,
            ..Default::default()
        };
        let logs = generate_logs(&config);

        let success_count = logs
            .iter()
            .filter(|l| l.status_code >= 200 && l.status_code < 300)
            .count();
        // Should have roughly 90% success with some variance
        assert!(success_count > 800);
    }

    #[test]
    fn test_generate_app_logs() {
        let config = LogsConfig {
            count: 100,
            seed: Some(42),
            format: LogFormat::Application,
            ..Default::default()
        };
        let logs = generate_app_logs(&config);
        assert_eq!(logs.len(), 100);
        assert!(logs.iter().any(|l| l.level == "INFO"));
    }

    #[test]
    fn test_latency_distribution() {
        let config = LogsConfig {
            count: 1000,
            seed: Some(42),
            ..Default::default()
        };
        let logs = generate_logs(&config);

        let avg_latency: f64 = logs.iter().map(|l| l.latency_ms).sum::<f64>() / logs.len() as f64;
        // Average latency should be reasonable
        assert!(avg_latency > 10.0 && avg_latency < 500.0);
    }

    #[test]
    fn test_ip_pool() {
        let mut rng = create_rng(Some(42));
        let ips = generate_ip_pool(&mut rng, 100);
        assert_eq!(ips.len(), 100);
        // IPs should be unique (with high probability)
        let unique: std::collections::HashSet<_> = ips.iter().collect();
        assert!(unique.len() > 95);
    }
}
