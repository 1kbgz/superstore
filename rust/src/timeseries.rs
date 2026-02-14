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

    // Set up regime chain if enabled
    let mut regime_chain = create_regime_chain(&config.regimes);
    let mut current_regime = 0usize;

    // Generate innovations
    let mut innovations = Vec::with_capacity(config.nper);

    for _ in 0..config.nper {
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

        // Sample innovation (normal or Student-t for fat tails)
        let effective_sigma = config.sigma * vol_mult;
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

    // Apply AR(1) dynamics
    let mut ar1 = AR1::new(config.ar_phi, 1.0, 0.0).expect("Invalid AR1 parameters");
    let ar_weights = ar1.sample_n(rng, config.nper);

    // Blend AR weights with innovations
    let values: Vec<f64> = innovations
        .iter()
        .zip(ar_weights.iter())
        .map(|(&inn, &ar)| inn * (1.0 - config.ar_phi.abs()) + ar * config.ar_phi.abs())
        .collect();

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
pub fn get_time_series_with_config(config: &TimeseriesConfig) -> TimeSeriesData {
    let mut rng = create_rng(config.seed);
    let cols = get_cols(config.ncol);
    let index = make_date_index(config.nper, &config.freq);
    let mut columns = Vec::with_capacity(config.ncol);

    // For cross-correlated series, generate a common factor
    let common_factor: Vec<f64> = if config.cross_correlation > 0.0 {
        let (_, factor) = make_time_series_with_config_inner(&mut rng, config);
        factor
    } else {
        vec![]
    };

    for c in cols {
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

        columns.push(TimeSeriesColumn { name: c, values });
    }

    TimeSeriesData { index, columns }
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
