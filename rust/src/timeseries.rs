use chrono::{Datelike, Duration as ChronoDuration, NaiveDate, NaiveDateTime, Weekday};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::temporal::{MarkovChain, AR1};

const ALPHABET: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

// =============================================================================
// Timeseries Configuration Structs
// =============================================================================

/// Configuration for regime-switching behavior
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegimeConfig {
    pub enable: bool,
    pub n_regimes: usize,
    pub regime_persistence: f64,
    pub volatility_multipliers: Vec<f64>,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            enable: false,
            n_regimes: 2,
            regime_persistence: 0.95,
            volatility_multipliers: vec![1.0, 2.5],
        }
    }
}

/// Configuration for jump diffusion
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JumpConfig {
    pub enable: bool,
    pub jump_probability: f64,
    pub jump_mean: f64,
    pub jump_stddev: f64,
}

impl Default for JumpConfig {
    fn default() -> Self {
        Self {
            enable: false,
            jump_probability: 0.01,
            jump_mean: 0.0,
            jump_stddev: 0.05,
        }
    }
}

// =============================================================================
// Priority 5: Enhanced Timeseries Features
// =============================================================================

/// Configuration for GARCH-like volatility clustering
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GarchConfig {
    pub enable: bool,
    pub alpha: f64,  // Weight on past squared returns
    pub beta: f64,   // Weight on past variance (persistence)
    pub omega: f64,  // Long-run variance constant
}

impl Default for GarchConfig {
    fn default() -> Self {
        Self {
            enable: false,
            alpha: 0.1,
            beta: 0.85,
            omega: 0.05,
        }
    }
}

/// Configuration for mean reversion (Ornstein-Uhlenbeck process)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeanReversionConfig {
    pub enable: bool,
    pub theta: f64,  // Speed of mean reversion (higher = faster)
    pub mu: f64,     // Long-run mean
    pub sigma: f64,  // Volatility
}

impl Default for MeanReversionConfig {
    fn default() -> Self {
        Self {
            enable: false,
            theta: 0.15,
            mu: 0.0,
            sigma: 0.2,
        }
    }
}

/// Configuration for intraday patterns (U-shaped volatility)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntradayConfig {
    pub enable: bool,
    pub opening_volatility_mult: f64,  // Multiplier at market open
    pub midday_volatility_mult: f64,   // Multiplier at midday (lowest)
    pub closing_volatility_mult: f64,  // Multiplier at market close
}

impl Default for IntradayConfig {
    fn default() -> Self {
        Self {
            enable: false,
            opening_volatility_mult: 1.5,
            midday_volatility_mult: 0.7,
            closing_volatility_mult: 1.3,
        }
    }
}

/// Configuration for event windows (abnormal returns around dates)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventWindowConfig {
    pub enable: bool,
    pub event_indices: Vec<usize>,       // Indices where events occur
    pub pre_event_window: usize,          // Days before event
    pub post_event_window: usize,         // Days after event
    pub abnormal_return_mean: f64,        // Mean of abnormal return
    pub abnormal_return_stddev: f64,      // Stddev of abnormal return
}

impl Default for EventWindowConfig {
    fn default() -> Self {
        Self {
            enable: false,
            event_indices: vec![],
            pre_event_window: 5,
            post_event_window: 5,
            abnormal_return_mean: 0.02,
            abnormal_return_stddev: 0.03,
        }
    }
}

/// Financial metrics output
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinancialMetrics {
    pub alpha: f64,        // Jensen's alpha (excess return)
    pub beta: f64,         // Market sensitivity
    pub sharpe_ratio: f64, // Risk-adjusted return
    pub volatility: f64,   // Annualized volatility
    pub max_drawdown: f64, // Maximum peak-to-trough decline
}

impl Default for FinancialMetrics {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            beta: 1.0,
            sharpe_ratio: 0.0,
            volatility: 0.0,
            max_drawdown: 0.0,
        }
    }
}

/// Full timeseries configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeseriesConfig {
    pub nper: usize,
    pub ncol: usize,
    pub freq: String,
    pub seed: Option<u64>,
    pub ar_phi: f64,
    pub sigma: f64,
    pub drift: f64,
    pub cumulative: bool,
    pub use_fat_tails: bool,
    pub degrees_freedom: f64,
    pub cross_correlation: f64,
    pub regimes: RegimeConfig,
    pub jumps: JumpConfig,
    // Priority 5 enhancements
    pub garch: GarchConfig,
    pub mean_reversion: MeanReversionConfig,
    pub intraday: IntradayConfig,
    pub event_windows: EventWindowConfig,
    pub compute_metrics: bool,
}

impl Default for TimeseriesConfig {
    fn default() -> Self {
        Self {
            nper: 30,
            ncol: 4,
            freq: "B".to_string(),
            seed: None,
            ar_phi: 0.95,
            sigma: 1.0,
            drift: 0.0,
            cumulative: true,
            use_fat_tails: false,
            degrees_freedom: 5.0,
            cross_correlation: 0.0,
            regimes: RegimeConfig::default(),
            jumps: JumpConfig::default(),
            garch: GarchConfig::default(),
            mean_reversion: MeanReversionConfig::default(),
            intraday: IntradayConfig::default(),
            event_windows: EventWindowConfig::default(),
            compute_metrics: false,
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn get_cols(k: usize) -> Vec<char> {
    ALPHABET.chars().take(k).collect()
}

/// Create an RNG from an optional seed
fn create_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

/// Generate a Student-t random variate using the ratio of normals method
/// This is more efficient than the inverse CDF method for most df values
fn sample_student_t<R: Rng>(rng: &mut R, df: f64) -> f64 {
    // Use the fact that T = Z / sqrt(V/df) where Z ~ N(0,1) and V ~ Chi^2(df)
    // Chi^2(df) = sum of df squared standard normals
    let normal = Normal::new(0.0, 1.0).expect("Invalid normal params");
    let z: f64 = normal.sample(rng);

    // For Chi^2, we use df/2 independent pairs
    let mut chi_sq = 0.0;
    let n_samples = (df / 2.0).ceil() as usize;
    for _ in 0..n_samples {
        let x: f64 = normal.sample(rng);
        let y: f64 = normal.sample(rng);
        chi_sq += x * x + y * y;
    }
    // Adjust for actual df
    chi_sq = chi_sq * (df / (2.0 * n_samples as f64));

    z / (chi_sq / df).sqrt()
}

/// Generate a single innovation (normal or Student-t)
fn sample_innovation<R: Rng>(rng: &mut R, sigma: f64, use_fat_tails: bool, df: f64) -> f64 {
    if use_fat_tails && df > 2.0 {
        // Scale Student-t to have the same variance as N(0, sigma^2)
        // Var(T_df) = df / (df - 2) for df > 2
        let scale = sigma * ((df - 2.0) / df).sqrt();
        sample_student_t(rng, df) * scale
    } else {
        let normal = Normal::new(0.0, sigma).expect("Invalid normal params");
        normal.sample(rng)
    }
}

/// Create a regime-switching Markov chain
fn create_regime_chain(config: &RegimeConfig) -> Option<MarkovChain> {
    if !config.enable {
        return None;
    }

    let n = config.n_regimes;
    let p_stay = config.regime_persistence;
    let p_switch = (1.0 - p_stay) / (n - 1).max(1) as f64;

    // Build transition matrix
    let mut matrix = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = vec![p_switch; n];
        row[i] = p_stay;
        matrix.push(row);
    }

    let states: Vec<String> = (0..n).map(|i| format!("regime_{}", i)).collect();
    MarkovChain::new(matrix, states).ok()
}

// =============================================================================
// Priority 5: Helper Functions
// =============================================================================

/// GARCH(1,1) volatility model
/// sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
fn apply_garch_volatility<R: Rng>(
    _rng: &mut R,
    innovations: &mut [f64],
    config: &GarchConfig,
) {
    if !config.enable || innovations.is_empty() {
        return;
    }

    let mut variance = config.omega / (1.0 - config.alpha - config.beta); // Long-run variance
    
    for i in 0..innovations.len() {
        let prev_return_sq = if i > 0 {
            innovations[i - 1].powi(2)
        } else {
            variance
        };
        
        // Update variance: GARCH(1,1) equation
        variance = config.omega + config.alpha * prev_return_sq + config.beta * variance;
        
        // Scale the innovation by the time-varying volatility
        innovations[i] *= variance.sqrt();
    }
}

/// Ornstein-Uhlenbeck (mean-reverting) process
/// dX_t = theta * (mu - X_t) * dt + sigma * dW_t
fn generate_ornstein_uhlenbeck<R: Rng>(
    rng: &mut R,
    n: usize,
    config: &MeanReversionConfig,
) -> Vec<f64> {
    let normal = Normal::new(0.0, 1.0).expect("Invalid normal params");
    let mut values = Vec::with_capacity(n);
    let mut x = config.mu; // Start at long-run mean
    
    let dt = 1.0; // Daily timestep
    
    for _ in 0..n {
        let dw: f64 = normal.sample(rng);
        // Euler-Maruyama discretization
        x += config.theta * (config.mu - x) * dt + config.sigma * dt.sqrt() * dw;
        values.push(x);
    }
    
    values
}

/// Get intraday volatility multiplier based on position in trading day
/// Creates a U-shaped pattern: high at open, low at midday, high at close
fn get_intraday_volatility_mult(index: usize, total: usize, config: &IntradayConfig) -> f64 {
    if !config.enable || total == 0 {
        return 1.0;
    }
    
    // Normalize position to [0, 1]
    let t = (index as f64) / (total as f64);
    
    // U-shaped curve: higher at endpoints, lower in middle
    // Using a quadratic: volatility = a + b*(t - 0.5)^2
    // At t=0 and t=1: volatility is high
    // At t=0.5: volatility is low
    
    let deviation = (t - 0.5).abs() * 2.0; // 0 at midday, 1 at open/close
    
    // Interpolate between midday (min) and open/close (max)
    let opening_blend = if t < 0.5 {
        1.0 - t * 2.0 // 1.0 at open, 0.0 at midday
    } else {
        0.0
    };
    let closing_blend = if t >= 0.5 {
        (t - 0.5) * 2.0 // 0.0 at midday, 1.0 at close
    } else {
        0.0
    };
    
    let base = config.midday_volatility_mult;
    let opening_contrib = opening_blend * (config.opening_volatility_mult - base);
    let closing_contrib = closing_blend * (config.closing_volatility_mult - base);
    
    base + opening_contrib + closing_contrib + deviation * 0.1
}

/// Apply abnormal returns around event dates
fn apply_event_windows<R: Rng>(
    rng: &mut R,
    values: &mut [f64],
    config: &EventWindowConfig,
) {
    if !config.enable || config.event_indices.is_empty() {
        return;
    }
    
    let normal = Normal::new(config.abnormal_return_mean, config.abnormal_return_stddev)
        .expect("Invalid normal params");
    
    for &event_idx in &config.event_indices {
        // Calculate window boundaries
        let start_idx = event_idx.saturating_sub(config.pre_event_window);
        let end_idx = (event_idx + config.post_event_window + 1).min(values.len());
        
        // Apply abnormal returns in the window
        for (i, value) in values.iter_mut().enumerate().take(end_idx).skip(start_idx) {
            let abnormal_return: f64 = normal.sample(rng);
            
            // Strongest effect at the event, decaying away from it
            let distance = (i as i32 - event_idx as i32).abs() as f64;
            let decay = 1.0 / (1.0 + distance * 0.3);
            
            *value += abnormal_return * decay;
        }
    }
}

/// Calculate financial metrics from a return series
fn calculate_financial_metrics(returns: &[f64], market_returns: Option<&[f64]>, risk_free_rate: f64) -> FinancialMetrics {
    if returns.is_empty() {
        return FinancialMetrics::default();
    }
    
    let n = returns.len() as f64;
    
    // Calculate mean return
    let mean_return: f64 = returns.iter().sum::<f64>() / n;
    
    // Calculate volatility (annualized, assuming daily returns)
    let variance: f64 = returns.iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>() / (n - 1.0).max(1.0);
    let volatility = variance.sqrt() * (252.0_f64).sqrt(); // Annualize
    
    // Calculate Sharpe ratio
    let excess_return = mean_return - risk_free_rate / 252.0; // Daily risk-free rate
    let sharpe_ratio = if variance > 0.0 {
        excess_return / variance.sqrt() * (252.0_f64).sqrt()
    } else {
        0.0
    };
    
    // Calculate max drawdown
    let mut cumulative = Vec::with_capacity(returns.len());
    let mut cum = 0.0;
    for r in returns {
        cum += r;
        cumulative.push(cum);
    }
    
    let mut max_drawdown = 0.0;
    let mut peak = f64::NEG_INFINITY;
    for &val in &cumulative {
        if val > peak {
            peak = val;
        }
        let drawdown = peak - val;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }
    
    // Calculate beta and alpha if market returns provided
    let (alpha, beta) = if let Some(market) = market_returns {
        if market.len() == returns.len() && market.len() > 1 {
            // Calculate market mean
            let market_mean: f64 = market.iter().sum::<f64>() / market.len() as f64;
            
            // Calculate covariance and market variance
            let covariance: f64 = returns.iter()
                .zip(market.iter())
                .map(|(r, m)| (r - mean_return) * (m - market_mean))
                .sum::<f64>() / (n - 1.0);
            
            let market_variance: f64 = market.iter()
                .map(|m| (m - market_mean).powi(2))
                .sum::<f64>() / (n - 1.0).max(1.0);
            
            let b = if market_variance > 0.0 {
                covariance / market_variance
            } else {
                1.0
            };
            
            // Alpha = mean_return - risk_free - beta * (market_mean - risk_free)
            let a = mean_return - risk_free_rate / 252.0 - b * (market_mean - risk_free_rate / 252.0);
            
            (a * 252.0, b) // Annualize alpha
        } else {
            (0.0, 1.0)
        }
    } else {
        (0.0, 1.0)
    };
    
    FinancialMetrics {
        alpha,
        beta,
        sharpe_ratio,
        volatility,
        max_drawdown,
    }
}

fn make_date_index(k: usize, freq: &str) -> Vec<NaiveDateTime> {
    let start = NaiveDate::from_ymd_opt(2000, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let mut dates = Vec::with_capacity(k);
    let mut current = start;

    match freq {
        "B" => {
            // Business day frequency
            while dates.len() < k {
                let weekday = current.weekday();
                if weekday != Weekday::Sat && weekday != Weekday::Sun {
                    dates.push(current);
                }
                current += ChronoDuration::days(1);
            }
        }
        "D" => {
            // Daily frequency
            for i in 0..k {
                dates.push(start + ChronoDuration::days(i as i64));
            }
        }
        "W" => {
            // Weekly frequency
            for i in 0..k {
                dates.push(start + ChronoDuration::weeks(i as i64));
            }
        }
        "M" => {
            // Monthly frequency (approximate)
            for i in 0..k {
                dates.push(start + ChronoDuration::days((i * 30) as i64));
            }
        }
        _ => {
            // Default to business day
            while dates.len() < k {
                let weekday = current.weekday();
                if weekday != Weekday::Sat && weekday != Weekday::Sun {
                    dates.push(current);
                }
                current += ChronoDuration::days(1);
            }
        }
    }

    dates
}

fn make_time_series_with_rng<R: Rng>(
    rng: &mut R,
    nper: usize,
    freq: &str,
) -> (Vec<NaiveDateTime>, Vec<f64>) {
    // Delegate to config-based version with defaults
    let config = TimeseriesConfig {
        nper,
        freq: freq.to_string(),
        ..Default::default()
    };
    make_time_series_with_config_inner(rng, &config)
}

/// Enhanced time series generation with full config support
fn make_time_series_with_config_inner<R: Rng>(
    rng: &mut R,
    config: &TimeseriesConfig,
) -> (Vec<NaiveDateTime>, Vec<f64>) {
    let dates = make_date_index(config.nper, &config.freq);

    // If mean reversion is enabled, use Ornstein-Uhlenbeck process instead
    if config.mean_reversion.enable {
        let values = generate_ornstein_uhlenbeck(rng, config.nper, &config.mean_reversion);
        return (dates, values);
    }

    // Set up regime chain if enabled
    let mut regime_chain = create_regime_chain(&config.regimes);
    let mut current_regime = 0usize;

    // Generate innovations
    let mut innovations = Vec::with_capacity(config.nper);

    for i in 0..config.nper {
        // Update regime if we have regime switching
        if let Some(ref mut chain) = regime_chain {
            let state = chain.next(rng);
            // Parse regime number from state name
            current_regime = state
                .strip_prefix("regime_")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        }

        // Get volatility multiplier for current regime
        let vol_mult = if config.regimes.enable {
            *config
                .regimes
                .volatility_multipliers
                .get(current_regime)
                .unwrap_or(&1.0)
        } else {
            1.0
        };

        // Apply intraday volatility pattern
        let intraday_mult = get_intraday_volatility_mult(i, config.nper, &config.intraday);

        // Sample innovation (normal or Student-t for fat tails)
        let effective_sigma = config.sigma * vol_mult * intraday_mult;
        let mut innovation = sample_innovation(
            rng,
            effective_sigma,
            config.use_fat_tails,
            config.degrees_freedom,
        );

        // Add jump component if enabled
        if config.jumps.enable && rng.gen::<f64>() < config.jumps.jump_probability {
            let jump_dist = Normal::new(config.jumps.jump_mean, config.jumps.jump_stddev)
                .expect("Invalid jump params");
            innovation += jump_dist.sample(rng);
        }

        // Add drift
        innovation += config.drift;

        innovations.push(innovation);
    }

    // Apply GARCH volatility clustering
    apply_garch_volatility(rng, &mut innovations, &config.garch);

    // Apply AR(1) dynamics
    let mut ar1 = AR1::new(config.ar_phi, 1.0, 0.0).expect("Invalid AR1 parameters");
    let ar_weights = ar1.sample_n(rng, config.nper);

    // Blend AR weights with innovations
    let mut values: Vec<f64> = innovations
        .iter()
        .zip(ar_weights.iter())
        .map(|(&inn, &ar)| inn * (1.0 - config.ar_phi.abs()) + ar * config.ar_phi.abs())
        .collect();

    // Apply event window effects
    apply_event_windows(rng, &mut values, &config.event_windows);

    // Optionally compute cumulative sum for trending time series
    if config.cumulative {
        let cumsum: Vec<f64> = values
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();
        (dates, cumsum)
    } else {
        (dates, values)
    }
}

/// Generate time series data with full configuration support
pub fn get_time_series_with_config(config: &TimeseriesConfig) -> TimeSeriesDataWithMetrics {
    let mut rng = create_rng(config.seed);
    let cols = get_cols(config.ncol);
    let index = make_date_index(config.nper, &config.freq);
    let mut columns = Vec::with_capacity(config.ncol);
    let mut metrics_map = HashMap::new();

    // For cross-correlated series, generate a common factor
    let common_factor: Vec<f64> = if config.cross_correlation > 0.0 {
        let (_, factor) = make_time_series_with_config_inner(&mut rng, config);
        factor
    } else {
        vec![]
    };

    // Generate market returns for beta calculation (first column acts as market)
    let mut market_returns: Option<Vec<f64>> = None;

    for (col_idx, c) in cols.iter().enumerate() {
        let (_, mut values) = make_time_series_with_config_inner(&mut rng, config);

        // Blend with common factor for cross-correlation
        if config.cross_correlation > 0.0 && !common_factor.is_empty() {
            let rho = config.cross_correlation;
            values = values
                .iter()
                .zip(common_factor.iter())
                .map(|(&v, &f)| (1.0 - rho.sqrt()) * v + rho.sqrt() * f)
                .collect();
        }

        // Calculate returns for metrics
        if config.compute_metrics {
            let returns: Vec<f64> = if config.cumulative && values.len() > 1 {
                // For cumulative series, compute returns from differences
                values.windows(2).map(|w| w[1] - w[0]).collect()
            } else {
                values.clone()
            };

            // First column acts as market proxy
            if col_idx == 0 {
                market_returns = Some(returns.clone());
            }

            let metrics = calculate_financial_metrics(
                &returns,
                if col_idx > 0 { market_returns.as_deref() } else { None },
                0.02, // 2% annual risk-free rate
            );
            metrics_map.insert(*c, metrics);
        }

        columns.push(TimeSeriesColumn { name: *c, values });
    }

    TimeSeriesDataWithMetrics {
        index,
        columns,
        metrics: if config.compute_metrics { Some(metrics_map) } else { None },
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeSeriesColumn {
    pub name: char,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeSeriesData {
    pub index: Vec<NaiveDateTime>,
    pub columns: Vec<TimeSeriesColumn>,
}

/// Extended time series data with optional financial metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeSeriesDataWithMetrics {
    pub index: Vec<NaiveDateTime>,
    pub columns: Vec<TimeSeriesColumn>,
    pub metrics: Option<HashMap<char, FinancialMetrics>>,
}

impl From<TimeSeriesDataWithMetrics> for TimeSeriesData {
    fn from(data: TimeSeriesDataWithMetrics) -> Self {
        TimeSeriesData {
            index: data.index,
            columns: data.columns,
        }
    }
}

pub fn get_time_series_data(
    nper: usize,
    freq: &str,
    ncol: usize,
    seed: Option<u64>,
) -> HashMap<char, (Vec<NaiveDateTime>, Vec<f64>)> {
    let mut rng = create_rng(seed);
    let cols = get_cols(ncol);
    let mut data = HashMap::new();

    for c in cols {
        data.insert(c, make_time_series_with_rng(&mut rng, nper, freq));
    }

    data
}

pub fn get_time_series(nper: usize, freq: &str, ncol: usize, seed: Option<u64>) -> TimeSeriesData {
    let mut rng = create_rng(seed);
    let cols = get_cols(ncol);
    let index = make_date_index(nper, freq);
    let mut columns = Vec::with_capacity(ncol);

    for c in cols {
        let (_, values) = make_time_series_with_rng(&mut rng, nper, freq);
        columns.push(TimeSeriesColumn { name: c, values });
    }

    TimeSeriesData { index, columns }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_cols() {
        assert_eq!(get_cols(4), vec!['A', 'B', 'C', 'D']);
        assert_eq!(
            get_cols(10),
            vec!['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        );
    }

    #[test]
    fn test_make_date_index() {
        let dates = make_date_index(10, "B");
        assert_eq!(dates.len(), 10);
        // First business day from Jan 1, 2000 (Saturday) should be Jan 3, 2000 (Monday)
        assert_eq!(
            dates[0].date(),
            NaiveDate::from_ymd_opt(2000, 1, 3).unwrap()
        );
    }

    #[test]
    fn test_get_time_series() {
        let data = get_time_series(30, "B", 4, None);
        assert_eq!(data.index.len(), 30);
        assert_eq!(data.columns.len(), 4);
        assert_eq!(data.columns[0].name, 'A');
        assert_eq!(data.columns[1].name, 'B');
        assert_eq!(data.columns[2].name, 'C');
        assert_eq!(data.columns[3].name, 'D');
    }

    #[test]
    fn test_get_time_series_seeded() {
        let data1 = get_time_series(10, "D", 2, Some(99999));
        let data2 = get_time_series(10, "D", 2, Some(99999));
        // Same seed should produce same results
        assert_eq!(data1.columns[0].values, data2.columns[0].values);
        assert_eq!(data1.columns[1].values, data2.columns[1].values);
    }

    #[test]
    fn test_get_time_series_data_seeded() {
        let data1 = get_time_series_data(10, "D", 2, Some(88888));
        let data2 = get_time_series_data(10, "D", 2, Some(88888));
        // Same seed should produce same results
        assert_eq!(data1.get(&'A').unwrap().1, data2.get(&'A').unwrap().1);
        assert_eq!(data1.get(&'B').unwrap().1, data2.get(&'B').unwrap().1);
    }
}
