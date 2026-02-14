//! Financial data generator module.
//!
//! Generates realistic financial data including:
//! - Stock prices via Geometric Brownian Motion with jumps
//! - OHLCV bars with realistic intraday relationships
//! - Options data with Black-Scholes Greeks
//! - Correlated multi-asset returns via GaussianCopula

use chrono::{Datelike, Duration as ChronoDuration, NaiveDate, NaiveDateTime, Weekday};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, LogNormal, Normal};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::copulas::GaussianCopula;

// =============================================================================
// Constants
// =============================================================================

const TICKER_PREFIXES: &[&str] = &[
    "A", "B", "C", "D", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
];

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for stock price generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StockConfig {
    /// Annual drift (expected return), e.g., 0.08 for 8%
    pub annual_drift: f64,
    /// Annual volatility, e.g., 0.20 for 20%
    pub annual_volatility: f64,
    /// Initial stock price
    pub initial_price: f64,
    /// Enable jump diffusion
    pub enable_jumps: bool,
    /// Jump probability per day
    pub jump_probability: f64,
    /// Jump mean (log)
    pub jump_mean: f64,
    /// Jump standard deviation (log)
    pub jump_stddev: f64,
}

impl Default for StockConfig {
    fn default() -> Self {
        Self {
            annual_drift: 0.08,
            annual_volatility: 0.20,
            initial_price: 100.0,
            enable_jumps: false,
            jump_probability: 0.02,
            jump_mean: 0.0,
            jump_stddev: 0.05,
        }
    }
}

/// Configuration for OHLCV bar generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OhlcvConfig {
    /// Average daily volume
    pub avg_volume: u64,
    /// Volume volatility (as fraction of avg)
    pub volume_volatility: f64,
    /// Intraday volatility multiplier
    pub intraday_volatility: f64,
    /// Enable volume-price correlation (high volume on big moves)
    pub volume_price_correlation: f64,
}

impl Default for OhlcvConfig {
    fn default() -> Self {
        Self {
            avg_volume: 1_000_000,
            volume_volatility: 0.5,
            intraday_volatility: 0.02,
            volume_price_correlation: 0.3,
        }
    }
}

/// Configuration for options generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptionsConfig {
    /// Risk-free rate for Black-Scholes
    pub risk_free_rate: f64,
    /// Dividend yield
    pub dividend_yield: f64,
    /// Days to expiration options
    pub expirations: Vec<u32>,
    /// Strike price offsets (as % of spot, e.g., 0.9, 0.95, 1.0, 1.05, 1.1)
    pub strike_offsets: Vec<f64>,
}

impl Default for OptionsConfig {
    fn default() -> Self {
        Self {
            risk_free_rate: 0.05,
            dividend_yield: 0.02,
            expirations: vec![7, 14, 30, 60, 90],
            strike_offsets: vec![0.90, 0.95, 0.97, 1.0, 1.03, 1.05, 1.10],
        }
    }
}

/// Full finance configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinanceConfig {
    /// Number of trading days
    pub ndays: usize,
    /// Number of assets
    pub n_assets: usize,
    /// Random seed
    pub seed: Option<u64>,
    /// Start date
    pub start_date: Option<String>,
    /// Asset correlation (for multi-asset generation)
    pub asset_correlation: f64,
    /// Stock configuration
    pub stock: StockConfig,
    /// OHLCV configuration
    pub ohlcv: OhlcvConfig,
    /// Options configuration
    pub options: OptionsConfig,
    /// Custom ticker symbols
    pub tickers: Vec<String>,
}

impl Default for FinanceConfig {
    fn default() -> Self {
        Self {
            ndays: 252,
            n_assets: 1,
            seed: None,
            start_date: None,
            asset_correlation: 0.5,
            stock: StockConfig::default(),
            ohlcv: OhlcvConfig::default(),
            options: OptionsConfig::default(),
            tickers: vec!["AAPL".to_string()],
        }
    }
}

// =============================================================================
// Data Structures
// =============================================================================

/// Single OHLCV bar
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OhlcvBar {
    pub date: String,
    pub ticker: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
    pub vwap: f64,
    pub returns: f64,
}

/// Options quote with Greeks
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptionQuote {
    pub date: String,
    pub ticker: String,
    pub option_type: String,
    pub underlying_price: f64,
    pub strike: f64,
    pub expiration: String,
    pub days_to_expiry: u32,
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub implied_vol: f64,
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

fn generate_ticker<R: Rng>(rng: &mut R) -> String {
    let len = rng.gen_range(2..=4);
    (0..len)
        .map(|_| *TICKER_PREFIXES.choose(rng).unwrap())
        .collect()
}

fn make_date_index(ndays: usize, start: NaiveDateTime) -> Vec<NaiveDateTime> {
    let mut dates = Vec::with_capacity(ndays);
    let mut current = start;

    while dates.len() < ndays {
        let weekday = current.weekday();
        if weekday != Weekday::Sat && weekday != Weekday::Sun {
            dates.push(current);
        }
        current += ChronoDuration::days(1);
    }

    dates
}

/// Standard normal CDF approximation
fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Error function approximation (Abramowitz and Stegun)
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Standard normal PDF
fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Black-Scholes option pricing
fn black_scholes(
    spot: f64,
    strike: f64,
    time_to_expiry: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    volatility: f64,
    is_call: bool,
) -> (f64, f64, f64, f64, f64) {
    // Returns: (price, delta, gamma, theta, vega)

    if time_to_expiry <= 0.0 {
        // At expiry
        let intrinsic = if is_call {
            (spot - strike).max(0.0)
        } else {
            (strike - spot).max(0.0)
        };
        return (intrinsic, if is_call { 1.0 } else { -1.0 }, 0.0, 0.0, 0.0);
    }

    let d1 = ((spot / strike).ln()
        + (risk_free_rate - dividend_yield + 0.5 * volatility.powi(2)) * time_to_expiry)
        / (volatility * time_to_expiry.sqrt());
    let d2 = d1 - volatility * time_to_expiry.sqrt();

    let discount = (-risk_free_rate * time_to_expiry).exp();
    let div_discount = (-dividend_yield * time_to_expiry).exp();

    let (price, delta) = if is_call {
        let price = spot * div_discount * norm_cdf(d1) - strike * discount * norm_cdf(d2);
        let delta = div_discount * norm_cdf(d1);
        (price, delta)
    } else {
        let price = strike * discount * norm_cdf(-d2) - spot * div_discount * norm_cdf(-d1);
        let delta = -div_discount * norm_cdf(-d1);
        (price, delta)
    };

    // Gamma (same for calls and puts)
    let gamma = div_discount * norm_pdf(d1) / (spot * volatility * time_to_expiry.sqrt());

    // Theta (per day, not per year)
    let theta_common =
        -spot * div_discount * norm_pdf(d1) * volatility / (2.0 * time_to_expiry.sqrt());
    let theta = if is_call {
        (theta_common - risk_free_rate * strike * discount * norm_cdf(d2)
            + dividend_yield * spot * div_discount * norm_cdf(d1))
            / 365.0
    } else {
        (theta_common + risk_free_rate * strike * discount * norm_cdf(-d2)
            - dividend_yield * spot * div_discount * norm_cdf(-d1))
            / 365.0
    };

    // Vega (per 1% vol move)
    let vega = spot * div_discount * norm_pdf(d1) * time_to_expiry.sqrt() / 100.0;

    (price, delta, gamma, theta, vega)
}

// =============================================================================
// Generators
// =============================================================================

/// Generate stock prices using Geometric Brownian Motion with optional jumps
pub fn generate_stock_prices(config: &FinanceConfig) -> Vec<OhlcvBar> {
    let mut rng = create_rng(config.seed);

    let start = config
        .start_date
        .as_ref()
        .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .unwrap_or_else(|| NaiveDate::from_ymd_opt(2024, 1, 2).unwrap())
        .and_hms_opt(0, 0, 0)
        .unwrap();

    let dates = make_date_index(config.ndays, start);
    let ticker = config
        .tickers
        .first()
        .cloned()
        .unwrap_or_else(|| generate_ticker(&mut rng));

    // Convert annual params to daily
    let daily_drift = config.stock.annual_drift / 252.0;
    let daily_vol = config.stock.annual_volatility / 252.0_f64.sqrt();

    let normal = Normal::new(0.0, 1.0).expect("Invalid normal params");
    let volume_dist = LogNormal::new(
        (config.ohlcv.avg_volume as f64).ln(),
        config.ohlcv.volume_volatility,
    )
    .expect("Invalid lognormal params");

    let mut prices = Vec::with_capacity(config.ndays);
    let mut prev_close = config.stock.initial_price;

    for date in dates {
        // GBM: S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        let z: f64 = normal.sample(&mut rng);
        let mut log_return = (daily_drift - 0.5 * daily_vol.powi(2)) + daily_vol * z;

        // Add jump if enabled
        if config.stock.enable_jumps && rng.gen::<f64>() < config.stock.jump_probability {
            let jump_normal = Normal::new(config.stock.jump_mean, config.stock.jump_stddev)
                .expect("Invalid jump params");
            log_return += jump_normal.sample(&mut rng);
        }

        // Calculate OHLC
        let returns = log_return.exp() - 1.0;
        let close = prev_close * log_return.exp();

        // Generate intraday high/low
        let intraday_vol = config.ohlcv.intraday_volatility;
        let high_factor = 1.0 + rng.gen::<f64>() * intraday_vol;
        let low_factor = 1.0 - rng.gen::<f64>() * intraday_vol;

        let (open, high, low) = if returns > 0.0 {
            // Up day: open near low, close near high
            let open = prev_close * (1.0 + rng.gen::<f64>() * returns * 0.3);
            let high = close * high_factor;
            let low = open * low_factor;
            (open, high, low.min(open))
        } else {
            // Down day: open near high, close near low
            let open = prev_close * (1.0 + rng.gen::<f64>() * returns.abs() * 0.3);
            let high = open * high_factor;
            let low = close * low_factor;
            (open, high.max(open), low)
        };

        // Volume with price correlation
        let abs_return = returns.abs();
        let volume_mult = 1.0 + config.ohlcv.volume_price_correlation * abs_return * 10.0;
        let volume = (volume_dist.sample(&mut rng) * volume_mult) as u64;

        // VWAP approximation
        let vwap = (open + high + low + close) / 4.0;

        prices.push(OhlcvBar {
            date: date.format("%Y-%m-%d").to_string(),
            ticker: ticker.clone(),
            open: (open * 100.0).round() / 100.0,
            high: (high * 100.0).round() / 100.0,
            low: (low * 100.0).round() / 100.0,
            close: (close * 100.0).round() / 100.0,
            volume,
            vwap: (vwap * 100.0).round() / 100.0,
            returns: (returns * 10000.0).round() / 10000.0,
        });

        prev_close = close;
    }

    prices
}

/// Generate correlated multi-asset OHLCV data
pub fn generate_multi_asset_prices(config: &FinanceConfig) -> Vec<OhlcvBar> {
    let mut rng = create_rng(config.seed);
    let n_assets = config.n_assets;

    if n_assets <= 1 {
        return generate_stock_prices(config);
    }

    // Generate tickers
    let tickers: Vec<String> = if config.tickers.len() >= n_assets {
        config.tickers.iter().take(n_assets).cloned().collect()
    } else {
        (0..n_assets).map(|_| generate_ticker(&mut rng)).collect()
    };

    // Build correlation matrix
    let corr = config.asset_correlation.clamp(-0.99, 0.99);
    let mut matrix = Vec::with_capacity(n_assets);
    for i in 0..n_assets {
        let mut row = vec![corr; n_assets];
        row[i] = 1.0;
        matrix.push(row);
    }

    // Generate correlated returns using copula
    let copula = GaussianCopula::new(matrix).expect("Invalid correlation matrix");
    let correlated_uniforms = copula.sample_n(&mut rng, config.ndays);

    let start = config
        .start_date
        .as_ref()
        .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .unwrap_or_else(|| NaiveDate::from_ymd_opt(2024, 1, 2).unwrap())
        .and_hms_opt(0, 0, 0)
        .unwrap();

    let dates = make_date_index(config.ndays, start);
    let daily_drift = config.stock.annual_drift / 252.0;
    let daily_vol = config.stock.annual_volatility / 252.0_f64.sqrt();

    let normal = Normal::new(0.0, 1.0).expect("Invalid normal params");
    let volume_dist = LogNormal::new(
        (config.ohlcv.avg_volume as f64).ln(),
        config.ohlcv.volume_volatility,
    )
    .expect("Invalid lognormal params");

    let mut all_bars = Vec::with_capacity(config.ndays * n_assets);
    let mut prev_closes: Vec<f64> = vec![config.stock.initial_price; n_assets];

    for (day_idx, date) in dates.iter().enumerate() {
        let uniforms = &correlated_uniforms[day_idx];

        for (asset_idx, ticker) in tickers.iter().enumerate() {
            // Transform uniform to standard normal
            let z = inv_norm_cdf(uniforms[asset_idx]);
            let mut log_return = (daily_drift - 0.5 * daily_vol.powi(2)) + daily_vol * z;

            // Add jump if enabled
            if config.stock.enable_jumps && rng.gen::<f64>() < config.stock.jump_probability {
                let jump_normal = Normal::new(config.stock.jump_mean, config.stock.jump_stddev)
                    .expect("Invalid jump params");
                log_return += jump_normal.sample(&mut rng);
            }

            let returns = log_return.exp() - 1.0;
            let close = prev_closes[asset_idx] * log_return.exp();

            // Intraday high/low
            let intraday_vol = config.ohlcv.intraday_volatility;
            let high_factor = 1.0 + rng.gen::<f64>() * intraday_vol;
            let low_factor = 1.0 - rng.gen::<f64>() * intraday_vol;

            let (open, high, low) = if returns > 0.0 {
                let open = prev_closes[asset_idx] * (1.0 + rng.gen::<f64>() * returns * 0.3);
                let high = close * high_factor;
                let low = open * low_factor;
                (open, high, low.min(open))
            } else {
                let open = prev_closes[asset_idx] * (1.0 + rng.gen::<f64>() * returns.abs() * 0.3);
                let high = open * high_factor;
                let low = close * low_factor;
                (open, high.max(open), low)
            };

            let abs_return = returns.abs();
            let volume_mult = 1.0 + config.ohlcv.volume_price_correlation * abs_return * 10.0;
            let volume = (volume_dist.sample(&mut rng) * volume_mult) as u64;
            let vwap = (open + high + low + close) / 4.0;

            all_bars.push(OhlcvBar {
                date: date.format("%Y-%m-%d").to_string(),
                ticker: ticker.clone(),
                open: (open * 100.0).round() / 100.0,
                high: (high * 100.0).round() / 100.0,
                low: (low * 100.0).round() / 100.0,
                close: (close * 100.0).round() / 100.0,
                volume,
                vwap: (vwap * 100.0).round() / 100.0,
                returns: (returns * 10000.0).round() / 10000.0,
            });

            prev_closes[asset_idx] = close;
        }
    }

    all_bars
}

/// Inverse normal CDF (quantile function) approximation
fn inv_norm_cdf(p: f64) -> f64 {
    // Use rational approximation (Abramowitz and Stegun)
    let p = p.clamp(1e-10, 1.0 - 1e-10);

    let a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];
    let b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    let c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ];
    let d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Generate options chain with Greeks
pub fn generate_options_chain(
    config: &FinanceConfig,
    spot_price: f64,
    date: &str,
) -> Vec<OptionQuote> {
    let mut options = Vec::new();
    let ticker = config
        .tickers
        .first()
        .cloned()
        .unwrap_or_else(|| "SPY".to_string());

    let base_date = NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .unwrap_or_else(|_| NaiveDate::from_ymd_opt(2024, 1, 2).unwrap());

    for &days_to_expiry in &config.options.expirations {
        let expiry_date = base_date + ChronoDuration::days(days_to_expiry as i64);
        let time_to_expiry = days_to_expiry as f64 / 365.0;

        for &strike_offset in &config.options.strike_offsets {
            let strike = (spot_price * strike_offset * 100.0).round() / 100.0;

            for is_call in [true, false] {
                let (price, delta, gamma, theta, vega) = black_scholes(
                    spot_price,
                    strike,
                    time_to_expiry,
                    config.options.risk_free_rate,
                    config.options.dividend_yield,
                    config.stock.annual_volatility,
                    is_call,
                );

                options.push(OptionQuote {
                    date: date.to_string(),
                    ticker: ticker.clone(),
                    option_type: if is_call {
                        "call".to_string()
                    } else {
                        "put".to_string()
                    },
                    underlying_price: (spot_price * 100.0).round() / 100.0,
                    strike,
                    expiration: expiry_date.format("%Y-%m-%d").to_string(),
                    days_to_expiry,
                    price: (price * 100.0).round() / 100.0,
                    delta: (delta * 10000.0).round() / 10000.0,
                    gamma: (gamma * 10000.0).round() / 10000.0,
                    theta: (theta * 10000.0).round() / 10000.0,
                    vega: (vega * 10000.0).round() / 10000.0,
                    implied_vol: config.stock.annual_volatility,
                });
            }
        }
    }

    options
}

/// Generate full finance dataset: prices + options
pub fn generate_finance_data(config: &FinanceConfig) -> (Vec<OhlcvBar>, Vec<OptionQuote>) {
    let prices = if config.n_assets > 1 {
        generate_multi_asset_prices(config)
    } else {
        generate_stock_prices(config)
    };

    // Generate options for the last date
    let options = if let Some(last_bar) = prices.last() {
        generate_options_chain(config, last_bar.close, &last_bar.date)
    } else {
        Vec::new()
    };

    (prices, options)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_stock_prices() {
        let config = FinanceConfig {
            ndays: 100,
            seed: Some(42),
            ..Default::default()
        };
        let prices = generate_stock_prices(&config);
        assert_eq!(prices.len(), 100);

        // Check OHLC relationships
        for bar in &prices {
            assert!(bar.high >= bar.low);
            assert!(bar.high >= bar.open);
            assert!(bar.high >= bar.close);
            assert!(bar.low <= bar.open);
            assert!(bar.low <= bar.close);
            assert!(bar.volume > 0);
        }
    }

    #[test]
    fn test_generate_stock_prices_seeded() {
        let config = FinanceConfig {
            ndays: 50,
            seed: Some(42),
            ..Default::default()
        };
        let prices1 = generate_stock_prices(&config);
        let prices2 = generate_stock_prices(&config);

        assert_eq!(prices1.len(), prices2.len());
        assert_eq!(prices1[0].close, prices2[0].close);
        assert_eq!(prices1[10].close, prices2[10].close);
    }

    #[test]
    fn test_generate_multi_asset() {
        let config = FinanceConfig {
            ndays: 50,
            n_assets: 3,
            seed: Some(42),
            tickers: vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()],
            ..Default::default()
        };
        let prices = generate_multi_asset_prices(&config);

        // Should have 50 days * 3 assets = 150 bars
        assert_eq!(prices.len(), 150);

        // Check all tickers present
        let tickers: std::collections::HashSet<_> =
            prices.iter().map(|b| b.ticker.as_str()).collect();
        assert!(tickers.contains("AAPL"));
        assert!(tickers.contains("GOOGL"));
        assert!(tickers.contains("MSFT"));
    }

    #[test]
    fn test_black_scholes() {
        // Test call option
        let (price, delta, gamma, theta, vega) = black_scholes(
            100.0, // spot
            100.0, // strike (ATM)
            0.25,  // 3 months
            0.05,  // 5% risk-free
            0.02,  // 2% dividend
            0.20,  // 20% vol
            true,  // call
        );

        // ATM call should have positive price and ~0.5 delta
        assert!(price > 0.0);
        assert!(delta > 0.4 && delta < 0.6);
        assert!(gamma > 0.0);
        assert!(theta < 0.0); // Theta is negative (time decay)
        assert!(vega > 0.0);
    }

    #[test]
    fn test_generate_options_chain() {
        let config = FinanceConfig::default();
        let options = generate_options_chain(&config, 100.0, "2024-01-15");

        // Should have: 5 expirations * 7 strikes * 2 (call/put) = 70 options
        assert_eq!(options.len(), 70);

        // Check calls and puts present
        let calls: Vec<_> = options.iter().filter(|o| o.option_type == "call").collect();
        let puts: Vec<_> = options.iter().filter(|o| o.option_type == "put").collect();
        assert_eq!(calls.len(), 35);
        assert_eq!(puts.len(), 35);
    }

    #[test]
    fn test_norm_cdf() {
        // Test known values
        assert!((norm_cdf(0.0) - 0.5).abs() < 0.001);
        assert!((norm_cdf(-3.0) - 0.00135).abs() < 0.001);
        assert!((norm_cdf(3.0) - 0.99865).abs() < 0.001);
    }

    #[test]
    fn test_with_jumps() {
        let config = FinanceConfig {
            ndays: 100,
            seed: Some(42),
            stock: StockConfig {
                enable_jumps: true,
                jump_probability: 0.1, // High probability to see some jumps
                ..Default::default()
            },
            ..Default::default()
        };
        let prices = generate_stock_prices(&config);
        assert_eq!(prices.len(), 100);
    }
}
