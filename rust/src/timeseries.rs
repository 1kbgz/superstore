use chrono::{Datelike, Duration as ChronoDuration, NaiveDate, NaiveDateTime, Weekday};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const ALPHABET: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

fn get_cols(k: usize) -> Vec<char> {
    ALPHABET.chars().take(k).collect()
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

fn make_time_series(nper: usize, freq: &str) -> (Vec<NaiveDateTime>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let dates = make_date_index(nper, freq);

    // Generate cumulative sum of random normals with mean 0.2 and std 1
    let mut values = Vec::with_capacity(nper);
    let mut cumsum = 0.0;
    for _ in 0..nper {
        // Approximate normal distribution using Box-Muller
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let value = 0.2 + z; // mean 0.2, std 1
        cumsum += value;
        values.push(cumsum);
    }

    (dates, values)
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
) -> HashMap<char, (Vec<NaiveDateTime>, Vec<f64>)> {
    let cols = get_cols(ncol);
    let mut data = HashMap::new();

    for c in cols {
        data.insert(c, make_time_series(nper, freq));
    }

    data
}

pub fn get_time_series(nper: usize, freq: &str, ncol: usize) -> TimeSeriesData {
    let cols = get_cols(ncol);
    let index = make_date_index(nper, freq);
    let mut columns = Vec::with_capacity(ncol);

    for c in cols {
        let (_, values) = make_time_series(nper, freq);
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
        let data = get_time_series(30, "B", 4);
        assert_eq!(data.index.len(), 30);
        assert_eq!(data.columns.len(), 4);
        assert_eq!(data.columns[0].name, 'A');
        assert_eq!(data.columns[1].name, 'B');
        assert_eq!(data.columns[2].name, 'C');
        assert_eq!(data.columns[3].name, 'D');
    }
}
