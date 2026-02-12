//! Weather data generator module.
//!
//! Generates realistic outdoor sensor data with:
//! - Day/night temperature cycles
//! - Seasonal variations
//! - Weather events (rain, snow, clouds)
//! - Occasional outlier events (sensor errors, extreme weather)

use chrono::{Datelike, NaiveDate, NaiveDateTime, Timelike, Utc};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Climate zone affecting weather patterns
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ClimateZone {
    Tropical,
    Subtropical,
    Temperate,
    Continental,
    Polar,
    Arid,
    Mediterranean,
}

impl Default for ClimateZone {
    fn default() -> Self {
        ClimateZone::Temperate
    }
}

/// Weather event types
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum WeatherEvent {
    Clear,
    Cloudy,
    Rain,
    HeavyRain,
    Snow,
    Storm,
    Heatwave,
    ColdSnap,
    Fog,
}

impl Default for WeatherEvent {
    fn default() -> Self {
        WeatherEvent::Clear
    }
}

/// Configuration for weather generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeatherConfig {
    pub count: usize,
    pub seed: Option<u64>,
    pub start_date: Option<String>,
    pub frequency_minutes: u32,
    pub climate_zone: ClimateZone,
    pub latitude: f64,
    pub base_temp_celsius: f64,
    pub temp_daily_amplitude: f64,
    pub temp_seasonal_amplitude: f64,
    pub temp_noise_stddev: f64,
    pub base_humidity_percent: f64,
    pub humidity_temp_correlation: f64,
    pub precipitation_probability: f64,
    pub enable_weather_events: bool,
    pub event_probability: f64,
    pub outlier_probability: f64,
    pub sensor_drift: bool,
    pub sensor_drift_rate: f64,
}

impl Default for WeatherConfig {
    fn default() -> Self {
        WeatherConfig {
            count: 1000,
            seed: None,
            start_date: None,
            frequency_minutes: 15,
            climate_zone: ClimateZone::Temperate,
            latitude: 40.0,
            base_temp_celsius: 15.0,
            temp_daily_amplitude: 10.0,
            temp_seasonal_amplitude: 15.0,
            temp_noise_stddev: 2.0,
            base_humidity_percent: 60.0,
            humidity_temp_correlation: -0.3,
            precipitation_probability: 0.15,
            enable_weather_events: true,
            event_probability: 0.05,
            outlier_probability: 0.01,
            sensor_drift: false,
            sensor_drift_rate: 0.001,
        }
    }
}

/// A single weather reading from an outdoor sensor
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeatherReading {
    pub timestamp: String,
    pub temperature_celsius: f64,
    pub humidity_percent: f64,
    pub precipitation_mm: f64,
    pub weather_event: String,
    pub is_outlier: bool,
}

/// Weather state for persistence between readings
struct WeatherState {
    current_event: WeatherEvent,
    event_duration: i32,
    cumulative_drift: f64,
    ar1_state: f64,
}

impl Default for WeatherState {
    fn default() -> Self {
        WeatherState {
            current_event: WeatherEvent::Clear,
            event_duration: 0,
            cumulative_drift: 0.0,
            ar1_state: 0.0,
        }
    }
}

/// Create an RNG from an optional seed
fn create_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

/// Parse a date string or use default
fn parse_start_date(start_date: &Option<String>) -> NaiveDateTime {
    if let Some(date_str) = start_date {
        if let Ok(date) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
            return date.and_hms_opt(0, 0, 0).unwrap();
        }
    }
    // Default: 30 days ago
    let today = Utc::now().naive_utc();
    today - chrono::Duration::days(30)
}

/// Calculate day of year as fraction (0.0 to 1.0)
fn day_of_year_fraction(date: &NaiveDateTime) -> f64 {
    let day_of_year = date.ordinal() as f64;
    day_of_year / 365.25
}

/// Calculate hour of day as fraction (0.0 to 1.0)
fn hour_of_day_fraction(date: &NaiveDateTime) -> f64 {
    let hour = date.hour() as f64;
    let minute = date.minute() as f64;
    (hour + minute / 60.0) / 24.0
}

/// Calculate seasonal temperature adjustment
fn seasonal_adjustment(day_fraction: f64, amplitude: f64, latitude: f64) -> f64 {
    // Peak summer around day 172 (June 21) in Northern Hemisphere
    // We want cos to be +1 on June 21 (day 172), so shift by ~0.47
    // For Southern Hemisphere, shift by additional 0.5 (6 months)
    let phase_shift = if latitude >= 0.0 { 0.0 } else { 0.5 };
    // day_fraction 0.47 should give angle 0 (cos=1)
    let seasonal_angle = 2.0 * PI * (day_fraction - 0.47 + phase_shift);
    amplitude * seasonal_angle.cos()
}

/// Calculate daily temperature cycle adjustment
fn daily_adjustment(hour_fraction: f64, amplitude: f64) -> f64 {
    // Peak temperature around 3 PM (hour 15), minimum around 5 AM
    let daily_angle = 2.0 * PI * (hour_fraction - 0.625);
    amplitude * daily_angle.cos()
}

/// Get climate-specific temperature modifiers
fn climate_modifiers(climate: &ClimateZone) -> (f64, f64, f64) {
    // Returns (temp_offset, humidity_base, precip_base)
    match climate {
        ClimateZone::Tropical => (10.0, 75.0, 0.25),
        ClimateZone::Subtropical => (5.0, 65.0, 0.20),
        ClimateZone::Temperate => (0.0, 60.0, 0.15),
        ClimateZone::Continental => (-5.0, 55.0, 0.12),
        ClimateZone::Polar => (-20.0, 70.0, 0.10),
        ClimateZone::Arid => (5.0, 25.0, 0.03),
        ClimateZone::Mediterranean => (3.0, 50.0, 0.10),
    }
}

/// Weather event effects on temperature and humidity
fn event_effects(event: &WeatherEvent) -> (f64, f64, f64) {
    // Returns (temp_modifier, humidity_modifier, precipitation_mm)
    match event {
        WeatherEvent::Clear => (0.0, -5.0, 0.0),
        WeatherEvent::Cloudy => (-2.0, 5.0, 0.0),
        WeatherEvent::Rain => (-4.0, 20.0, 2.5),
        WeatherEvent::HeavyRain => (-6.0, 30.0, 15.0),
        WeatherEvent::Snow => (-8.0, 15.0, 5.0),
        WeatherEvent::Storm => (-5.0, 25.0, 25.0),
        WeatherEvent::Heatwave => (8.0, -15.0, 0.0),
        WeatherEvent::ColdSnap => (-12.0, 10.0, 0.0),
        WeatherEvent::Fog => (-1.0, 25.0, 0.5),
    }
}

/// Sample a new weather event based on current conditions
fn sample_weather_event<R: Rng>(
    rng: &mut R,
    temperature: f64,
    climate: &ClimateZone,
) -> WeatherEvent {
    let r: f64 = rng.gen();

    // Adjust probabilities based on temperature
    if temperature < -5.0 {
        // Cold weather events more likely
        if r < 0.3 {
            WeatherEvent::Snow
        } else if r < 0.4 {
            WeatherEvent::ColdSnap
        } else if r < 0.5 {
            WeatherEvent::Cloudy
        } else if r < 0.55 {
            WeatherEvent::Fog
        } else {
            WeatherEvent::Clear
        }
    } else if temperature > 30.0 {
        // Hot weather events more likely
        if r < 0.2 {
            WeatherEvent::Heatwave
        } else if r < 0.35 {
            WeatherEvent::Storm
        } else if r < 0.45 {
            WeatherEvent::Clear
        } else if r < 0.6 {
            WeatherEvent::Cloudy
        } else {
            WeatherEvent::Clear
        }
    } else {
        // Normal temperature range
        match climate {
            ClimateZone::Tropical | ClimateZone::Subtropical => {
                if r < 0.2 {
                    WeatherEvent::Rain
                } else if r < 0.3 {
                    WeatherEvent::HeavyRain
                } else if r < 0.35 {
                    WeatherEvent::Storm
                } else if r < 0.5 {
                    WeatherEvent::Cloudy
                } else {
                    WeatherEvent::Clear
                }
            }
            ClimateZone::Arid => {
                if r < 0.02 {
                    WeatherEvent::Rain
                } else if r < 0.15 {
                    WeatherEvent::Cloudy
                } else {
                    WeatherEvent::Clear
                }
            }
            _ => {
                if r < 0.15 {
                    WeatherEvent::Rain
                } else if r < 0.2 {
                    WeatherEvent::HeavyRain
                } else if r < 0.35 {
                    WeatherEvent::Cloudy
                } else if r < 0.4 {
                    WeatherEvent::Fog
                } else {
                    WeatherEvent::Clear
                }
            }
        }
    }
}

/// Generate weather data based on configuration
pub fn generate_weather(config: &WeatherConfig) -> Vec<WeatherReading> {
    let mut rng = create_rng(config.seed);
    let start_datetime = parse_start_date(&config.start_date);
    let (climate_temp_offset, climate_humidity, climate_precip) =
        climate_modifiers(&config.climate_zone);

    let mut readings = Vec::with_capacity(config.count);
    let mut state = WeatherState::default();

    // AR(1) coefficient for temperature persistence
    let ar_phi = 0.85;

    for i in 0..config.count {
        // Calculate current timestamp
        let minutes_offset = (i as i64) * (config.frequency_minutes as i64);
        let current_time = start_datetime + chrono::Duration::minutes(minutes_offset);

        let day_frac = day_of_year_fraction(&current_time);
        let hour_frac = hour_of_day_fraction(&current_time);

        // Base temperature calculation
        let seasonal_temp =
            seasonal_adjustment(day_frac, config.temp_seasonal_amplitude, config.latitude);
        let daily_temp = daily_adjustment(hour_frac, config.temp_daily_amplitude);

        // AR(1) noise for realistic persistence
        let innovation: f64 = rng.gen::<f64>() * 2.0 - 1.0; // [-1, 1]
        state.ar1_state =
            ar_phi * state.ar1_state + (1.0 - ar_phi) * innovation * config.temp_noise_stddev;

        let mut temperature = config.base_temp_celsius
            + climate_temp_offset
            + seasonal_temp
            + daily_temp
            + state.ar1_state;

        // Weather event handling
        if config.enable_weather_events {
            if state.event_duration <= 0 {
                // Check for new event
                if rng.gen::<f64>() < config.event_probability {
                    state.current_event =
                        sample_weather_event(&mut rng, temperature, &config.climate_zone);
                    // Event duration: 4 to 24 readings (1-6 hours at 15 min intervals)
                    state.event_duration = rng.gen_range(4..24);
                } else {
                    state.current_event = WeatherEvent::Clear;
                }
            }
            state.event_duration -= 1;
        }

        let (event_temp_mod, event_humid_mod, event_precip) = event_effects(&state.current_event);
        temperature += event_temp_mod;

        // Calculate humidity (inversely correlated with temperature)
        let temp_deviation = temperature - config.base_temp_celsius;
        let base_humidity = config.base_humidity_percent + climate_humidity - 60.0;
        let mut humidity = base_humidity
            + config.humidity_temp_correlation * temp_deviation * 2.0
            + event_humid_mod
            + (rng.gen::<f64>() - 0.5) * 10.0;
        humidity = humidity.clamp(5.0, 100.0);

        // Calculate precipitation
        let mut precipitation = if state.current_event == WeatherEvent::Clear {
            if rng.gen::<f64>() < config.precipitation_probability * climate_precip / 0.15 {
                rng.gen::<f64>() * 2.0 // Light random precipitation
            } else {
                0.0
            }
        } else {
            event_precip * (0.5 + rng.gen::<f64>())
        };

        // Check if it's too warm for snow
        if temperature > 2.0 && state.current_event == WeatherEvent::Snow {
            // Convert snow to rain
            precipitation *= 0.1; // Snow water equivalent
        }

        // Sensor drift
        if config.sensor_drift {
            state.cumulative_drift += config.sensor_drift_rate * (rng.gen::<f64>() - 0.3);
            temperature += state.cumulative_drift;
        }

        // Outliers (sensor errors)
        let is_outlier = rng.gen::<f64>() < config.outlier_probability;
        if is_outlier {
            // Add significant random error
            let error_type: u8 = rng.gen_range(0..4);
            match error_type {
                0 => temperature += rng.gen_range(20.0..40.0), // Spike high
                1 => temperature -= rng.gen_range(20.0..40.0), // Spike low
                2 => humidity = rng.gen_range(0.0..10.0),      // Humidity sensor fail
                _ => temperature = -99.9,                      // Sensor offline reading
            }
        }

        // Round to realistic precision
        temperature = (temperature * 10.0).round() / 10.0;
        humidity = (humidity * 10.0).round() / 10.0;
        precipitation = (precipitation * 100.0).round() / 100.0;

        let reading = WeatherReading {
            timestamp: current_time.format("%Y-%m-%d %H:%M:%S").to_string(),
            temperature_celsius: temperature,
            humidity_percent: humidity,
            precipitation_mm: precipitation,
            weather_event: format!("{:?}", state.current_event).to_lowercase(),
            is_outlier,
        };
        readings.push(reading);
    }

    readings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_weather_default() {
        let config = WeatherConfig::default();
        let readings = generate_weather(&config);
        assert_eq!(readings.len(), 1000);

        for reading in &readings {
            assert!(!reading.timestamp.is_empty());
            // Temperature should be reasonable (allowing for outliers)
            // Most readings should be between -50 and 50
        }
    }

    #[test]
    fn test_generate_weather_seeded() {
        let mut config = WeatherConfig::default();
        config.count = 100;
        config.seed = Some(42);

        let readings1 = generate_weather(&config);
        let readings2 = generate_weather(&config);

        // Same seed should produce same results
        for (r1, r2) in readings1.iter().zip(readings2.iter()) {
            assert_eq!(r1.timestamp, r2.timestamp);
            assert_eq!(r1.temperature_celsius, r2.temperature_celsius);
            assert_eq!(r1.humidity_percent, r2.humidity_percent);
        }
    }

    #[test]
    fn test_climate_zones() {
        let climates = vec![ClimateZone::Tropical, ClimateZone::Arid, ClimateZone::Polar];

        for climate in climates {
            let mut config = WeatherConfig::default();
            config.count = 100;
            config.climate_zone = climate;
            config.seed = Some(42);

            let readings = generate_weather(&config);
            assert_eq!(readings.len(), 100);
        }
    }

    #[test]
    fn test_weather_events() {
        let mut config = WeatherConfig::default();
        config.count = 500;
        config.enable_weather_events = true;
        config.event_probability = 0.2; // High probability to ensure events occur
        config.seed = Some(42);

        let readings = generate_weather(&config);

        // Should have some precipitation
        let has_precipitation = readings.iter().any(|r| r.precipitation_mm > 0.0);
        assert!(has_precipitation, "Should have some precipitation readings");
    }

    #[test]
    fn test_outlier_generation() {
        let mut config = WeatherConfig::default();
        config.count = 1000;
        config.outlier_probability = 0.1; // 10% outlier rate
        config.seed = Some(42);

        let readings = generate_weather(&config);
        let outlier_count = readings.iter().filter(|r| r.is_outlier).count();

        // Should have roughly 10% outliers (allow some variance)
        assert!(outlier_count > 50, "Should have at least 50 outliers");
        assert!(outlier_count < 200, "Should have less than 200 outliers");
    }

    #[test]
    fn test_seasonal_adjustment() {
        // Summer day (June 21, day ~172)
        let summer = 172.0 / 365.25;
        let summer_adj = seasonal_adjustment(summer, 15.0, 40.0);

        // Winter day (December 21, day ~355)
        let winter = 355.0 / 365.25;
        let winter_adj = seasonal_adjustment(winter, 15.0, 40.0);

        // Summer should be warmer than winter in Northern Hemisphere
        assert!(summer_adj > winter_adj);
    }

    #[test]
    fn test_daily_adjustment() {
        // 3 PM (15:00) should be warmest
        let afternoon = 15.0 / 24.0;
        let afternoon_adj = daily_adjustment(afternoon, 10.0);

        // 5 AM should be coolest
        let early_morning = 5.0 / 24.0;
        let morning_adj = daily_adjustment(early_morning, 10.0);

        assert!(afternoon_adj > morning_adj);
    }
}
