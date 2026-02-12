use chrono::{Datelike, Duration as ChronoDuration, NaiveDate, NaiveDateTime, Weekday};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::temporal::AR1;

const ALPHABET: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

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
    let dates = make_date_index(nper, freq);

    // Use AR(1) model with phi=0.95 for realistic temporal autocorrelation
    // High phi means values persist (smooth time series)
    let mut ar1 = AR1::new(0.95, 1.0, 0.0).expect("Invalid AR1 parameters");
    let values = ar1.sample_n(rng, nper);

    // Compute cumulative sum for a trending time series (like stock prices)
    let cumsum: Vec<f64> = values
        .iter()
        .scan(0.0, |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect();

    (dates, cumsum)
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
