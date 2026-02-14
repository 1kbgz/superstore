use chrono::{Datelike, Duration as ChronoDuration, NaiveDateTime, Timelike, Utc};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use uuid::Uuid;

const REGIONS: [&str; 3] = ["na", "eu", "ap"];
const ZONES: [&str; 4] = ["A", "B", "C", "D"];

// =============================================================================
// IoT Telemetry Configuration
// =============================================================================

/// Machine operational state for state machine transitions
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MachineState {
    Healthy,
    Degraded,
    Critical,
    Maintenance,
    Recovering,
    Failed,
}

impl Default for MachineState {
    fn default() -> Self {
        MachineState::Healthy
    }
}

impl MachineState {
    pub fn as_str(&self) -> &'static str {
        match self {
            MachineState::Healthy => "healthy",
            MachineState::Degraded => "degraded",
            MachineState::Critical => "critical",
            MachineState::Maintenance => "maintenance",
            MachineState::Recovering => "recovering",
            MachineState::Failed => "failed",
        }
    }
}

/// Anomaly type for telemetry injection
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnomalyType {
    None,
    CpuSpike,
    MemoryLeak,
    NetworkSaturation,
    DiskFull,
    MultiResource,
}

impl Default for AnomalyType {
    fn default() -> Self {
        AnomalyType::None
    }
}

/// Configuration for anomaly injection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnomalyConfig {
    pub enable: bool,
    pub cpu_spike_probability: f64,
    pub cpu_spike_magnitude: f64,
    pub memory_leak_probability: f64,
    pub memory_leak_rate: f64,
    pub network_saturation_probability: f64,
    pub disk_fill_probability: f64,
    pub multi_resource_probability: f64,
    pub anomaly_duration_min: u32,
    pub anomaly_duration_max: u32,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            enable: false,
            cpu_spike_probability: 0.02,
            cpu_spike_magnitude: 40.0,
            memory_leak_probability: 0.01,
            memory_leak_rate: 0.5,
            network_saturation_probability: 0.01,
            disk_fill_probability: 0.005,
            multi_resource_probability: 0.005,
            anomaly_duration_min: 5,
            anomaly_duration_max: 30,
        }
    }
}

/// Configuration for sensor drift simulation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensorDriftConfig {
    pub enable: bool,
    pub drift_rate: f64,
    pub drift_bias: f64,
    pub recalibration_probability: f64,
}

impl Default for SensorDriftConfig {
    fn default() -> Self {
        Self {
            enable: false,
            drift_rate: 0.001,
            drift_bias: 0.7,
            recalibration_probability: 0.01,
        }
    }
}

/// Configuration for temporal patterns (diurnal cycles)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalConfig {
    pub enable: bool,
    pub diurnal_amplitude: f64,
    pub peak_hour: u32,
    pub weekend_reduction: f64,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            enable: false,
            diurnal_amplitude: 0.3,
            peak_hour: 14,
            weekend_reduction: 0.4,
        }
    }
}

/// Configuration for failure cascades
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FailureCascadeConfig {
    pub enable: bool,
    pub cascade_probability: f64,
    pub cascade_delay_readings: u32,
    pub zone_correlation: f64,
}

impl Default for FailureCascadeConfig {
    fn default() -> Self {
        Self {
            enable: false,
            cascade_probability: 0.3,
            cascade_delay_readings: 5,
            zone_correlation: 0.7,
        }
    }
}

/// Configuration for maintenance windows
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaintenanceConfig {
    pub enable: bool,
    pub window_probability: f64,
    pub window_duration_min: u32,
    pub window_duration_max: u32,
    pub scheduled_hours: Vec<u32>,
}

impl Default for MaintenanceConfig {
    fn default() -> Self {
        Self {
            enable: false,
            window_probability: 0.02,
            window_duration_min: 10,
            window_duration_max: 60,
            scheduled_hours: vec![2, 3, 4],
        }
    }
}

/// Configuration for degradation curves
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DegradationConfig {
    pub enable: bool,
    pub degradation_rate: f64,
    pub failure_threshold: f64,
    pub recovery_rate: f64,
}

impl Default for DegradationConfig {
    fn default() -> Self {
        Self {
            enable: false,
            degradation_rate: 0.002,
            failure_threshold: 0.95,
            recovery_rate: 0.1,
        }
    }
}

/// Full IoT telemetry configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TelemetryConfig {
    pub machine_count: usize,
    pub readings_per_machine: usize,
    pub seed: Option<u64>,
    pub start_time: Option<String>,
    pub frequency_seconds: u32,
    pub anomalies: AnomalyConfig,
    pub sensor_drift: SensorDriftConfig,
    pub temporal: TemporalConfig,
    pub failure_cascade: FailureCascadeConfig,
    pub maintenance: MaintenanceConfig,
    pub degradation: DegradationConfig,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            machine_count: 10,
            readings_per_machine: 100,
            seed: None,
            start_time: None,
            frequency_seconds: 60,
            anomalies: AnomalyConfig::default(),
            sensor_drift: SensorDriftConfig::default(),
            temporal: TemporalConfig::default(),
            failure_cascade: FailureCascadeConfig::default(),
            maintenance: MaintenanceConfig::default(),
            degradation: DegradationConfig::default(),
        }
    }
}

// =============================================================================
// Preset Scenarios - "Interesting Defaults"
// =============================================================================

impl TelemetryConfig {
    /// Normal operation - baseline behavior with no anomalies
    pub fn normal() -> Self {
        Self::default()
    }

    /// Scenario: CPU spikes occurring periodically
    pub fn cpu_spikes() -> Self {
        Self {
            anomalies: AnomalyConfig {
                enable: true,
                cpu_spike_probability: 0.08,
                cpu_spike_magnitude: 50.0,
                anomaly_duration_min: 3,
                anomaly_duration_max: 15,
                ..Default::default()
            },
            temporal: TemporalConfig {
                enable: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Scenario: Gradual memory leak building over time
    pub fn memory_leak() -> Self {
        Self {
            anomalies: AnomalyConfig {
                enable: true,
                memory_leak_probability: 0.15,
                memory_leak_rate: 0.8,
                anomaly_duration_min: 50,
                anomaly_duration_max: 100,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Scenario: Network saturation during peak hours
    pub fn network_congestion() -> Self {
        Self {
            anomalies: AnomalyConfig {
                enable: true,
                network_saturation_probability: 0.1,
                anomaly_duration_min: 10,
                anomaly_duration_max: 30,
                ..Default::default()
            },
            temporal: TemporalConfig {
                enable: true,
                diurnal_amplitude: 0.5,
                peak_hour: 11,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Scenario: Disk filling up over time
    pub fn disk_pressure() -> Self {
        Self {
            anomalies: AnomalyConfig {
                enable: true,
                disk_fill_probability: 0.1,
                anomaly_duration_min: 100,
                anomaly_duration_max: 200,
                ..Default::default()
            },
            degradation: DegradationConfig {
                enable: true,
                degradation_rate: 0.005,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Scenario: Cascading failures across a data center zone
    pub fn cascade_failure() -> Self {
        Self {
            anomalies: AnomalyConfig {
                enable: true,
                multi_resource_probability: 0.05,
                anomaly_duration_min: 10,
                anomaly_duration_max: 50,
                ..Default::default()
            },
            failure_cascade: FailureCascadeConfig {
                enable: true,
                cascade_probability: 0.5,
                cascade_delay_readings: 3,
                zone_correlation: 0.8,
            },
            ..Default::default()
        }
    }

    /// Scenario: Scheduled maintenance windows with recovery
    pub fn maintenance_window() -> Self {
        Self {
            maintenance: MaintenanceConfig {
                enable: true,
                window_probability: 0.1,
                window_duration_min: 15,
                window_duration_max: 45,
                scheduled_hours: vec![2, 3, 4, 5],
            },
            degradation: DegradationConfig {
                enable: true,
                recovery_rate: 0.15,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Scenario: Sensor calibration drift over extended period
    pub fn sensor_drift_scenario() -> Self {
        Self {
            sensor_drift: SensorDriftConfig {
                enable: true,
                drift_rate: 0.003,
                drift_bias: 0.8,
                recalibration_probability: 0.005,
            },
            ..Default::default()
        }
    }

    /// Scenario: Gradual degradation leading to failure and recovery
    pub fn degradation_cycle() -> Self {
        Self {
            degradation: DegradationConfig {
                enable: true,
                degradation_rate: 0.008,
                failure_threshold: 0.9,
                recovery_rate: 0.2,
            },
            anomalies: AnomalyConfig {
                enable: true,
                cpu_spike_probability: 0.03,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Scenario: Realistic production environment with all patterns
    pub fn production() -> Self {
        Self {
            machine_count: 50,
            readings_per_machine: 288, // 24 hours at 5-min intervals
            frequency_seconds: 300,
            anomalies: AnomalyConfig {
                enable: true,
                cpu_spike_probability: 0.02,
                cpu_spike_magnitude: 30.0,
                memory_leak_probability: 0.005,
                memory_leak_rate: 0.3,
                network_saturation_probability: 0.01,
                disk_fill_probability: 0.002,
                multi_resource_probability: 0.001,
                anomaly_duration_min: 3,
                anomaly_duration_max: 20,
            },
            sensor_drift: SensorDriftConfig {
                enable: true,
                drift_rate: 0.0005,
                drift_bias: 0.6,
                recalibration_probability: 0.001,
            },
            temporal: TemporalConfig {
                enable: true,
                diurnal_amplitude: 0.25,
                peak_hour: 14,
                weekend_reduction: 0.35,
            },
            failure_cascade: FailureCascadeConfig {
                enable: true,
                cascade_probability: 0.2,
                cascade_delay_readings: 4,
                zone_correlation: 0.6,
            },
            maintenance: MaintenanceConfig {
                enable: true,
                window_probability: 0.01,
                window_duration_min: 20,
                window_duration_max: 60,
                scheduled_hours: vec![2, 3, 4],
            },
            degradation: DegradationConfig {
                enable: true,
                degradation_rate: 0.001,
                failure_threshold: 0.95,
                recovery_rate: 0.08,
            },
            ..Default::default()
        }
    }

    /// Scenario: Chaos engineering - high anomaly rates for testing
    pub fn chaos() -> Self {
        Self {
            anomalies: AnomalyConfig {
                enable: true,
                cpu_spike_probability: 0.15,
                cpu_spike_magnitude: 60.0,
                memory_leak_probability: 0.1,
                memory_leak_rate: 1.5,
                network_saturation_probability: 0.1,
                disk_fill_probability: 0.08,
                multi_resource_probability: 0.05,
                anomaly_duration_min: 2,
                anomaly_duration_max: 10,
            },
            failure_cascade: FailureCascadeConfig {
                enable: true,
                cascade_probability: 0.6,
                cascade_delay_readings: 2,
                zone_correlation: 0.9,
            },
            ..Default::default()
        }
    }
}

// =============================================================================
// Telemetry Reading Structure
// =============================================================================

/// A single telemetry reading from a machine
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TelemetryReading {
    pub timestamp: String,
    pub machine_id: String,
    pub kind: String,
    pub cores: i32,
    pub region: String,
    pub zone: String,
    pub cpu: f64,
    pub mem: f64,
    pub free: f64,
    pub network: f64,
    pub disk: f64,
    pub state: String,
    pub anomaly_type: String,
    pub health_score: f64,
}

// =============================================================================
// Internal State Structs
// =============================================================================

/// State for a single machine
struct MachineSimState {
    state: MachineState,
    anomaly: AnomalyType,
    anomaly_remaining: u32,
    cpu_drift: f64,
    mem_drift: f64,
    cumulative_mem_leak: f64,
    cumulative_disk_fill: f64,
    degradation_level: f64,
    maintenance_remaining: u32,
    prev_cpu: f64,
    prev_mem: f64,
    prev_network: f64,
    prev_disk: f64,
}

impl Default for MachineSimState {
    fn default() -> Self {
        Self {
            state: MachineState::Healthy,
            anomaly: AnomalyType::None,
            anomaly_remaining: 0,
            cpu_drift: 0.0,
            mem_drift: 0.0,
            cumulative_mem_leak: 0.0,
            cumulative_disk_fill: 0.0,
            degradation_level: 0.0,
            maintenance_remaining: 0,
            prev_cpu: 0.0,
            prev_mem: 0.0,
            prev_network: 0.0,
            prev_disk: 0.0,
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Parse start time or use default
fn parse_start_time(start_time: &Option<String>) -> NaiveDateTime {
    if let Some(time_str) = start_time {
        if let Ok(dt) = NaiveDateTime::parse_from_str(time_str, "%Y-%m-%d %H:%M:%S") {
            return dt;
        }
    }
    Utc::now().naive_utc() - chrono::Duration::hours(24)
}

/// Calculate diurnal adjustment factor (0.0 to 1.0 centered)
fn diurnal_factor(hour: u32, peak_hour: u32, amplitude: f64) -> f64 {
    let hour_diff = ((hour as i32) - (peak_hour as i32)).abs() as f64;
    let normalized = hour_diff / 12.0; // 0 at peak, 1 at 12 hours away
    let factor = 1.0 - amplitude * (normalized * PI).cos().abs();
    factor.clamp(0.5, 1.5)
}

/// Check if it's a weekend
fn is_weekend(datetime: &NaiveDateTime) -> bool {
    let weekday = datetime.weekday().num_days_from_monday();
    weekday >= 5 // Saturday = 5, Sunday = 6
}

// =============================================================================
// Main Telemetry Generation
// =============================================================================

/// Generate IoT telemetry data with configurable behaviors
pub fn generate_telemetry(config: &TelemetryConfig) -> Vec<TelemetryReading> {
    let mut rng = create_rng(config.seed);
    let start_time = parse_start_time(&config.start_time);

    // Generate machines
    let machines_list = machines(config.machine_count, config.seed);

    // Initialize state for each machine
    let mut machine_states: Vec<MachineSimState> = machines_list
        .iter()
        .map(|_| MachineSimState::default())
        .collect();

    // Track cascade failures by zone
    let mut zone_failures: std::collections::HashMap<String, u32> =
        std::collections::HashMap::new();

    let mut readings = Vec::with_capacity(config.machine_count * config.readings_per_machine);

    for reading_idx in 0..config.readings_per_machine {
        let current_time = start_time
            + chrono::Duration::seconds((reading_idx as i64) * (config.frequency_seconds as i64));

        // Calculate temporal factors
        let hour = current_time.hour();
        let temporal_factor = if config.temporal.enable {
            let mut factor = diurnal_factor(
                hour,
                config.temporal.peak_hour,
                config.temporal.diurnal_amplitude,
            );
            if is_weekend(&current_time) {
                factor *= 1.0 - config.temporal.weekend_reduction;
            }
            factor
        } else {
            1.0
        };

        // Check for maintenance window
        let in_maintenance_window =
            config.maintenance.enable && config.maintenance.scheduled_hours.contains(&hour);

        for (machine_idx, machine) in machines_list.iter().enumerate() {
            let state = &mut machine_states[machine_idx];

            // Handle maintenance windows
            if in_maintenance_window && state.maintenance_remaining == 0 {
                if rng.gen::<f64>() < config.maintenance.window_probability {
                    state.maintenance_remaining = rng.gen_range(
                        config.maintenance.window_duration_min
                            ..=config.maintenance.window_duration_max,
                    );
                    state.state = MachineState::Maintenance;
                }
            }

            if state.maintenance_remaining > 0 {
                state.maintenance_remaining -= 1;
                if state.maintenance_remaining == 0 {
                    state.state = MachineState::Recovering;
                    // Reset degradation after maintenance
                    state.degradation_level *= 0.2;
                    state.cumulative_mem_leak = 0.0;
                    state.cumulative_disk_fill *= 0.5;
                }
            }

            // Handle anomaly injection
            if config.anomalies.enable
                && state.anomaly_remaining == 0
                && state.state != MachineState::Maintenance
            {
                let anomaly_roll: f64 = rng.gen();
                if anomaly_roll < config.anomalies.cpu_spike_probability {
                    state.anomaly = AnomalyType::CpuSpike;
                    state.anomaly_remaining = rng.gen_range(
                        config.anomalies.anomaly_duration_min
                            ..=config.anomalies.anomaly_duration_max,
                    );
                } else if anomaly_roll
                    < config.anomalies.cpu_spike_probability
                        + config.anomalies.memory_leak_probability
                {
                    state.anomaly = AnomalyType::MemoryLeak;
                    state.anomaly_remaining = rng.gen_range(
                        config.anomalies.anomaly_duration_min
                            ..=config.anomalies.anomaly_duration_max,
                    );
                } else if anomaly_roll
                    < config.anomalies.cpu_spike_probability
                        + config.anomalies.memory_leak_probability
                        + config.anomalies.network_saturation_probability
                {
                    state.anomaly = AnomalyType::NetworkSaturation;
                    state.anomaly_remaining = rng.gen_range(
                        config.anomalies.anomaly_duration_min
                            ..=config.anomalies.anomaly_duration_max,
                    );
                } else if anomaly_roll
                    < config.anomalies.cpu_spike_probability
                        + config.anomalies.memory_leak_probability
                        + config.anomalies.network_saturation_probability
                        + config.anomalies.disk_fill_probability
                {
                    state.anomaly = AnomalyType::DiskFull;
                    state.anomaly_remaining = rng.gen_range(
                        config.anomalies.anomaly_duration_min
                            ..=config.anomalies.anomaly_duration_max,
                    );
                } else if anomaly_roll
                    < config.anomalies.cpu_spike_probability
                        + config.anomalies.memory_leak_probability
                        + config.anomalies.network_saturation_probability
                        + config.anomalies.disk_fill_probability
                        + config.anomalies.multi_resource_probability
                {
                    state.anomaly = AnomalyType::MultiResource;
                    state.anomaly_remaining = rng.gen_range(
                        config.anomalies.anomaly_duration_min
                            ..=config.anomalies.anomaly_duration_max,
                    );
                }
            }

            // Check for cascade failures
            if config.failure_cascade.enable {
                if let Some(&delay) = zone_failures.get(&machine.zone) {
                    if delay == 0 && rng.gen::<f64>() < config.failure_cascade.zone_correlation {
                        state.anomaly = AnomalyType::MultiResource;
                        state.anomaly_remaining = rng.gen_range(5..20);
                    }
                }
            }

            // Decrement anomaly timer
            if state.anomaly_remaining > 0 {
                state.anomaly_remaining -= 1;
                if state.anomaly_remaining == 0 {
                    state.anomaly = AnomalyType::None;
                }
            }

            // Apply sensor drift
            if config.sensor_drift.enable {
                let drift_change = config.sensor_drift.drift_rate
                    * (rng.gen::<f64>() - (1.0 - config.sensor_drift.drift_bias));
                state.cpu_drift += drift_change;
                state.mem_drift += drift_change * 0.8;

                // Occasional recalibration
                if rng.gen::<f64>() < config.sensor_drift.recalibration_probability {
                    state.cpu_drift *= 0.1;
                    state.mem_drift *= 0.1;
                }
            }

            // Apply degradation
            if config.degradation.enable {
                if state.state != MachineState::Maintenance && state.state != MachineState::Failed {
                    state.degradation_level += config.degradation.degradation_rate;
                    if state.degradation_level > config.degradation.failure_threshold {
                        state.state = MachineState::Failed;
                        // Trigger cascade
                        if config.failure_cascade.enable
                            && rng.gen::<f64>() < config.failure_cascade.cascade_probability
                        {
                            zone_failures.insert(
                                machine.zone.clone(),
                                config.failure_cascade.cascade_delay_readings,
                            );
                        }
                    }
                }

                // Recovery for recovering/failed states
                if state.state == MachineState::Recovering || state.state == MachineState::Failed {
                    state.degradation_level -= config.degradation.recovery_rate;
                    if state.degradation_level < 0.1 {
                        state.degradation_level = 0.0;
                        state.state = MachineState::Healthy;
                    } else if state.state == MachineState::Failed && state.degradation_level < 0.5 {
                        state.state = MachineState::Recovering;
                    }
                }
            }

            // Calculate base metrics based on machine type
            let (base_cpu, base_mem, base_network, base_disk) = match machine.kind.as_str() {
                "core" => (35.0, 45.0, 50.0, 40.0),
                "edge" => (20.0, 40.0, 60.0, 55.0),
                _ => (70.0, 75.0, 70.0, 65.0), // worker
            };

            // Apply temporal factor
            let mut cpu = base_cpu * temporal_factor;
            let mut mem = base_mem * temporal_factor;
            let mut network = base_network * temporal_factor;
            let mut disk = base_disk + state.cumulative_disk_fill;

            // Apply anomaly effects
            match state.anomaly {
                AnomalyType::CpuSpike => {
                    cpu += config.anomalies.cpu_spike_magnitude * rng.gen_range(0.7..1.3);
                }
                AnomalyType::MemoryLeak => {
                    state.cumulative_mem_leak += config.anomalies.memory_leak_rate;
                    mem += state.cumulative_mem_leak;
                }
                AnomalyType::NetworkSaturation => {
                    network = 95.0 + rng.gen_range(0.0..5.0);
                }
                AnomalyType::DiskFull => {
                    state.cumulative_disk_fill += 0.5;
                    disk = 90.0 + state.cumulative_disk_fill.min(10.0);
                }
                AnomalyType::MultiResource => {
                    cpu += 30.0 * rng.gen_range(0.5..1.5);
                    mem += 25.0 * rng.gen_range(0.5..1.5);
                    network += 20.0 * rng.gen_range(0.5..1.5);
                }
                AnomalyType::None => {}
            }

            // Apply degradation to performance
            cpu += state.degradation_level * 20.0;
            mem += state.degradation_level * 15.0;

            // Apply sensor drift
            cpu += state.cpu_drift;
            mem += state.mem_drift;

            // Add random noise
            cpu += rng.gen_range(-5.0..5.0);
            mem += rng.gen_range(-3.0..3.0);
            network += rng.gen_range(-8.0..8.0);
            disk += rng.gen_range(-1.0..1.0);

            // Maintenance state: low utilization
            if state.state == MachineState::Maintenance {
                cpu = rng.gen_range(0.0..5.0);
                mem = rng.gen_range(5.0..15.0);
                network = rng.gen_range(0.0..3.0);
            }

            // Failed state: either zero or erratic
            if state.state == MachineState::Failed {
                if rng.gen::<f64>() < 0.7 {
                    cpu = 0.0;
                    mem = 0.0;
                    network = 0.0;
                } else {
                    cpu = rng.gen_range(0.0..100.0);
                    mem = rng.gen_range(0.0..100.0);
                }
            }

            // Clamp values
            cpu = clip(cpu, 0.0, 100.0);
            mem = clip(mem, 0.0, 100.0);
            network = clip(network, 0.0, 100.0);
            disk = clip(disk, 0.0, 100.0);
            let free = 100.0 - mem;

            // Update machine state based on metrics
            if state.state != MachineState::Maintenance && state.state != MachineState::Failed {
                if cpu > 90.0 || mem > 90.0 || disk > 95.0 {
                    state.state = MachineState::Critical;
                } else if cpu > 75.0 || mem > 80.0 || disk > 85.0 {
                    state.state = MachineState::Degraded;
                } else if state.state != MachineState::Recovering {
                    state.state = MachineState::Healthy;
                }
            }

            // Calculate health score
            let health_score = (1.0
                - (cpu / 100.0).powi(2) * 0.3
                - (mem / 100.0).powi(2) * 0.3
                - (disk / 100.0).powi(2) * 0.2
                - state.degradation_level * 0.2)
                .clamp(0.0, 1.0);

            // Store previous values
            state.prev_cpu = cpu;
            state.prev_mem = mem;
            state.prev_network = network;
            state.prev_disk = disk;

            let reading = TelemetryReading {
                timestamp: current_time.format("%Y-%m-%d %H:%M:%S").to_string(),
                machine_id: machine.machine_id.clone(),
                kind: machine.kind.clone(),
                cores: machine.cores,
                region: machine.region.clone(),
                zone: machine.zone.clone(),
                cpu,
                mem,
                free,
                network,
                disk,
                state: state.state.as_str().to_string(),
                anomaly_type: format!("{:?}", state.anomaly).to_lowercase(),
                health_score: (health_score * 100.0).round() / 100.0,
            };
            readings.push(reading);
        }

        // Update zone failure timers
        for delay in zone_failures.values_mut() {
            if *delay > 0 {
                *delay -= 1;
            }
        }
    }

    readings
}

// Adjectives and nouns for generating coolname-style names
const ADJECTIVES: [&str; 20] = [
    "brave", "calm", "dark", "eager", "fast", "gentle", "happy", "jolly", "kind", "lively",
    "merry", "nice", "proud", "quick", "rapid", "sharp", "smart", "swift", "warm", "wise",
];

const NOUNS: [&str; 20] = [
    "ant", "bear", "cat", "deer", "eagle", "fox", "goat", "hawk", "iguana", "jaguar", "koala",
    "lion", "mouse", "newt", "owl", "panda", "quail", "rabbit", "snake", "tiger",
];

/// Create an RNG from an optional seed
fn create_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

fn generate_name<R: Rng>(rng: &mut R) -> String {
    let adj = ADJECTIVES.choose(rng).unwrap();
    let noun = NOUNS.choose(rng).unwrap();
    format!("{}-{}", adj, noun)
}

fn generate_id() -> String {
    let uuid = Uuid::new_v4().to_string();
    uuid.rsplit('-').next().unwrap().to_string()
}

fn generate_id_seeded<R: Rng>(rng: &mut R) -> String {
    // Generate a UUID-like string using the seeded RNG
    let hex_chars: Vec<char> = "0123456789abcdef".chars().collect();
    (0..12).map(|_| *hex_chars.choose(rng).unwrap()).collect()
}

fn clip(value: f64, min: f64, max: f64) -> f64 {
    ((value.max(min).min(max) * 100.0).round()) / 100.0
}

fn randrange<R: Rng>(rng: &mut R, low: f64, high: f64) -> f64 {
    rng.gen::<f64>() * (high - low) + low
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Machine {
    pub machine_id: String,
    pub kind: String,
    pub cores: i32,
    pub region: String,
    pub zone: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Usage {
    pub machine_id: String,
    pub kind: String,
    pub cores: i32,
    pub region: String,
    pub zone: String,
    pub cpu: f64,
    pub mem: f64,
    pub free: f64,
    pub network: f64,
    pub disk: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Status {
    pub machine_id: String,
    pub kind: String,
    pub cores: i32,
    pub region: String,
    pub zone: String,
    pub cpu: f64,
    pub mem: f64,
    pub free: f64,
    pub network: f64,
    pub disk: f64,
    pub status: String,
    pub last_update: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    pub machine_id: String,
    pub job_id: String,
    pub name: String,
    pub units: i32,
    pub start_time: String,
    pub end_time: String,
}

pub fn machines(count: usize, seed: Option<u64>) -> Vec<Machine> {
    let mut rng = create_rng(seed);
    let mut result = Vec::with_capacity(count);

    for _ in 0..count {
        let rand_val: f64 = rng.gen();
        let (kind, cores) = if rand_val < 0.2 {
            let cores = *[8, 16, 32].choose(&mut rng).unwrap();
            ("core", cores)
        } else if rand_val < 0.7 {
            let cores = *[4, 8].choose(&mut rng).unwrap();
            ("edge", cores)
        } else {
            let cores = *[32, 64].choose(&mut rng).unwrap();
            ("worker", cores)
        };

        let machine = Machine {
            machine_id: if seed.is_some() {
                generate_id_seeded(&mut rng)
            } else {
                generate_id()
            },
            kind: kind.to_string(),
            cores,
            region: REGIONS.choose(&mut rng).unwrap().to_string(),
            zone: ZONES.choose(&mut rng).unwrap().to_string(),
        };
        result.push(machine);
    }

    result
}

pub fn usage(machine: &Machine, prev_usage: Option<&Usage>, seed: Option<u64>) -> Usage {
    let mut rng = create_rng(seed);

    // 10% chance to reset to zero
    let should_reset = prev_usage.is_none() || rng.gen::<f64>() < 0.1;

    if should_reset {
        return Usage {
            machine_id: machine.machine_id.clone(),
            kind: machine.kind.clone(),
            cores: machine.cores,
            region: machine.region.clone(),
            zone: machine.zone.clone(),
            cpu: 0.0,
            mem: 0.0,
            free: 100.0,
            network: 0.0,
            disk: 0.0,
        };
    }

    let prev = prev_usage.unwrap();

    let (cpu, mem, network, disk) = match machine.kind.as_str() {
        "core" => {
            // bursty cpu/mem/network/disk
            let cpu = randrange(
                &mut rng,
                if prev.cpu > 0.0 { prev.cpu - 15.0 } else { 0.0 },
                if prev.cpu > 0.0 {
                    prev.cpu + 15.0
                } else {
                    50.0
                },
            );
            let mem = randrange(
                &mut rng,
                if prev.mem > 0.0 {
                    prev.mem - 15.0
                } else {
                    20.0
                },
                if prev.mem > 0.0 {
                    prev.mem + 15.0
                } else {
                    70.0
                },
            );
            let network = randrange(
                &mut rng,
                if prev.network > 0.0 {
                    prev.network - 15.0
                } else {
                    20.0
                },
                if prev.network > 0.0 {
                    prev.network + 15.0
                } else {
                    70.0
                },
            );
            let disk = randrange(
                &mut rng,
                if prev.disk > 0.0 {
                    prev.disk - 15.0
                } else {
                    20.0
                },
                if prev.disk > 0.0 {
                    prev.disk + 15.0
                } else {
                    70.0
                },
            );
            (cpu, mem, network, disk)
        }
        "edge" => {
            // low cpu, medium mem, high network/disk
            let cpu = randrange(
                &mut rng,
                if prev.cpu > 0.0 {
                    prev.cpu - 5.0
                } else {
                    15.0 - 5.0
                },
                if prev.cpu > 0.0 {
                    prev.cpu + 5.0
                } else {
                    35.0 + 5.0
                },
            );
            let mem = randrange(
                &mut rng,
                if prev.mem > 0.0 {
                    prev.mem - 5.0
                } else {
                    35.0 - 5.0
                },
                if prev.mem > 0.0 {
                    prev.mem + 5.0
                } else {
                    55.0 + 5.0
                },
            );
            let network = randrange(
                &mut rng,
                if prev.network > 0.0 {
                    prev.network - 5.0
                } else {
                    65.0 - 5.0
                },
                if prev.network > 0.0 {
                    prev.network + 5.0
                } else {
                    75.0 + 5.0
                },
            );
            let disk = randrange(
                &mut rng,
                if prev.disk > 0.0 {
                    prev.disk - 5.0
                } else {
                    65.0 - 5.0
                },
                if prev.disk > 0.0 {
                    prev.disk + 5.0
                } else {
                    75.0 + 5.0
                },
            );
            (cpu, mem, network, disk)
        }
        _ => {
            // worker: high cpu, high mem, high network/disk
            let cpu = randrange(
                &mut rng,
                if prev.cpu > 0.0 {
                    prev.cpu - 5.0
                } else {
                    75.0 - 5.0
                },
                if prev.cpu > 0.0 {
                    prev.cpu + 5.0
                } else {
                    85.0 + 5.0
                },
            );
            let mem = randrange(
                &mut rng,
                if prev.mem > 0.0 {
                    prev.mem - 5.0
                } else {
                    75.0 - 5.0
                },
                if prev.mem > 0.0 {
                    prev.mem + 5.0
                } else {
                    85.0 + 5.0
                },
            );
            let network = randrange(
                &mut rng,
                if prev.network > 0.0 {
                    prev.network - 5.0
                } else {
                    75.0 - 5.0
                },
                if prev.network > 0.0 {
                    prev.network + 5.0
                } else {
                    85.0 + 5.0
                },
            );
            let disk = randrange(
                &mut rng,
                if prev.disk > 0.0 {
                    prev.disk - 5.0
                } else {
                    75.0 - 5.0
                },
                if prev.disk > 0.0 {
                    prev.disk + 5.0
                } else {
                    85.0 + 5.0
                },
            );
            (cpu, mem, network, disk)
        }
    };

    let cpu = clip(cpu, 0.0, 100.0);
    let mem = clip(mem, 0.0, 100.0);
    let free = 100.0 - mem;
    let network = clip(network, 0.0, f64::INFINITY);
    let disk = clip(disk, 0.0, f64::INFINITY);

    Usage {
        machine_id: machine.machine_id.clone(),
        kind: machine.kind.clone(),
        cores: machine.cores,
        region: machine.region.clone(),
        zone: machine.zone.clone(),
        cpu,
        mem,
        free,
        network,
        disk,
    }
}

pub fn status(usage_data: &Usage, _json: bool) -> Status {
    let now = Utc::now().naive_utc();
    let last_update = now.format("%Y-%m-%dT%H:%M:%S%.6f").to_string();

    // Status is determined by cpu level
    let status_str = if usage_data.cpu < 20.0 {
        "idle"
    } else if usage_data.cpu > 80.0 {
        "capacity"
    } else {
        "active"
    };

    Status {
        machine_id: usage_data.machine_id.clone(),
        kind: usage_data.kind.clone(),
        cores: usage_data.cores,
        region: usage_data.region.clone(),
        zone: usage_data.zone.clone(),
        cpu: usage_data.cpu,
        mem: usage_data.mem,
        free: usage_data.free,
        network: usage_data.network,
        disk: usage_data.disk,
        status: status_str.to_string(),
        last_update,
    }
}

pub fn job(machine: &Machine, _json: bool, seed: Option<u64>) -> Option<Job> {
    if machine.kind != "worker" {
        return None;
    }

    let mut rng = create_rng(seed);
    if rng.gen::<f64>() < 0.5 {
        return None;
    }

    let now = Utc::now().naive_utc();
    let end = now + ChronoDuration::seconds(rng.gen_range(0..400));

    let start_time = now.format("%Y-%m-%dT%H:%M:%S%.6f").to_string();
    let end_time = end.format("%Y-%m-%dT%H:%M:%S%.6f").to_string();

    Some(Job {
        machine_id: machine.machine_id.clone(),
        job_id: if seed.is_some() {
            generate_id_seeded(&mut rng)
        } else {
            generate_id()
        },
        name: generate_name(&mut rng),
        units: *[1, 2, 4, 8].choose(&mut rng).unwrap(),
        start_time,
        end_time,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_id() {
        let id = generate_id();
        assert_eq!(id.len(), 12);
    }

    #[test]
    fn test_clip() {
        assert_eq!(clip(5.0, 10.0, 20.0), 10.0);
        assert_eq!(clip(25.0, 10.0, 20.0), 20.0);
        assert_eq!(clip(15.0, 10.0, 20.0), 15.0);
    }

    #[test]
    fn test_randrange() {
        let mut rng = create_rng(None);
        for _ in 0..100 {
            let val = randrange(&mut rng, 10.0, 20.0);
            assert!(val >= 10.0 && val < 20.0);
        }
    }

    #[test]
    fn test_machines() {
        let ms = machines(100, None);
        assert_eq!(ms.len(), 100);
        for m in &ms {
            assert_eq!(m.machine_id.len(), 12);
            assert!(["core", "edge", "worker"].contains(&m.kind.as_str()));
            assert!(REGIONS.contains(&m.region.as_str()));
            assert!(ZONES.contains(&m.zone.as_str()));
            assert!([4, 8, 16, 32, 64].contains(&m.cores));
        }
    }

    #[test]
    fn test_machines_seeded() {
        let ms1 = machines(10, Some(42));
        let ms2 = machines(10, Some(42));
        // Same seed should produce same machines
        for (m1, m2) in ms1.iter().zip(ms2.iter()) {
            assert_eq!(m1.machine_id, m2.machine_id);
            assert_eq!(m1.kind, m2.kind);
            assert_eq!(m1.cores, m2.cores);
            assert_eq!(m1.region, m2.region);
            assert_eq!(m1.zone, m2.zone);
        }
    }

    #[test]
    fn test_usage() {
        let m = machines(1, None).pop().unwrap();
        let u = usage(&m, None, None);
        assert_eq!(u.machine_id, m.machine_id);
        assert_eq!(u.cpu, 0.0);
        assert_eq!(u.mem, 0.0);
        assert_eq!(u.free, 100.0);
    }

    #[test]
    fn test_status() {
        let m = machines(1, None).pop().unwrap();
        let u = usage(&m, None, None);
        let s = status(&u, false);
        assert_eq!(s.machine_id, m.machine_id);
        // cpu=0 means status is "idle" (cpu < 20)
        assert_eq!(s.status, "idle");
    }

    #[test]
    fn test_job() {
        let mut found_job = false;
        for _ in 0..100 {
            let m = Machine {
                machine_id: generate_id(),
                kind: "worker".to_string(),
                cores: 32,
                region: "na".to_string(),
                zone: "A".to_string(),
            };
            if let Some(j) = job(&m, false, None) {
                assert_eq!(j.machine_id, m.machine_id);
                assert_eq!(j.job_id.len(), 12);
                assert!(j.name.contains('-'));
                found_job = true;
            }
        }
        assert!(found_job, "Should find at least one job in 100 iterations");
    }

    #[test]
    fn test_job_non_worker() {
        let m = Machine {
            machine_id: generate_id(),
            kind: "edge".to_string(),
            cores: 8,
            region: "na".to_string(),
            zone: "A".to_string(),
        };
        assert!(job(&m, false, None).is_none());
    }

    // =========================================================================
    // Telemetry Tests
    // =========================================================================

    #[test]
    fn test_telemetry_default() {
        let config = TelemetryConfig::default();
        let readings = generate_telemetry(&config);
        assert_eq!(
            readings.len(),
            config.machine_count * config.readings_per_machine
        );

        for reading in &readings {
            assert!(!reading.timestamp.is_empty());
            assert!(reading.cpu >= 0.0 && reading.cpu <= 100.0);
            assert!(reading.mem >= 0.0 && reading.mem <= 100.0);
            assert!(reading.health_score >= 0.0 && reading.health_score <= 1.0);
        }
    }

    #[test]
    fn test_telemetry_seeded() {
        let mut config = TelemetryConfig::default();
        config.seed = Some(42);
        config.machine_count = 5;
        config.readings_per_machine = 20;

        let readings1 = generate_telemetry(&config);
        let readings2 = generate_telemetry(&config);

        for (r1, r2) in readings1.iter().zip(readings2.iter()) {
            assert_eq!(r1.timestamp, r2.timestamp);
            assert_eq!(r1.machine_id, r2.machine_id);
            assert_eq!(r1.cpu, r2.cpu);
            assert_eq!(r1.mem, r2.mem);
        }
    }

    #[test]
    fn test_telemetry_cpu_spikes_scenario() {
        let mut config = TelemetryConfig::cpu_spikes();
        config.seed = Some(123);
        config.machine_count = 5;
        config.readings_per_machine = 50;

        let readings = generate_telemetry(&config);

        // Should have some CPU spike anomalies
        let spike_count = readings
            .iter()
            .filter(|r| r.anomaly_type == "cpuspike")
            .count();
        assert!(
            spike_count > 0,
            "Expected CPU spikes in cpu_spikes scenario"
        );
    }

    #[test]
    fn test_telemetry_memory_leak_scenario() {
        let mut config = TelemetryConfig::memory_leak();
        config.seed = Some(456);
        config.machine_count = 3;
        config.readings_per_machine = 100;

        let readings = generate_telemetry(&config);

        // Should have memory leak anomalies
        let leak_count = readings
            .iter()
            .filter(|r| r.anomaly_type == "memoryleak")
            .count();
        assert!(
            leak_count > 0,
            "Expected memory leaks in memory_leak scenario"
        );
    }

    #[test]
    fn test_telemetry_production_scenario() {
        let mut config = TelemetryConfig::production();
        config.seed = Some(789);
        config.machine_count = 10;
        config.readings_per_machine = 50;

        let readings = generate_telemetry(&config);
        assert_eq!(readings.len(), 500);

        // Should have variety of states
        let states: std::collections::HashSet<_> =
            readings.iter().map(|r| r.state.clone()).collect();
        assert!(
            states.len() > 1,
            "Production scenario should have multiple states"
        );
    }

    #[test]
    fn test_telemetry_chaos_scenario() {
        let mut config = TelemetryConfig::chaos();
        config.seed = Some(999);
        config.machine_count = 5;
        config.readings_per_machine = 100;

        let readings = generate_telemetry(&config);

        // Should have many anomalies in chaos mode
        let anomaly_count = readings.iter().filter(|r| r.anomaly_type != "none").count();
        let anomaly_rate = anomaly_count as f64 / readings.len() as f64;
        assert!(
            anomaly_rate > 0.1,
            "Chaos scenario should have high anomaly rate (got {})",
            anomaly_rate
        );
    }

    #[test]
    fn test_telemetry_maintenance_scenario() {
        let mut config = TelemetryConfig::maintenance_window();
        config.seed = Some(111);
        config.machine_count = 5;
        config.readings_per_machine = 100;
        config.start_time = Some("2024-01-15 02:00:00".to_string());

        let readings = generate_telemetry(&config);

        // Should have some maintenance states
        let maintenance_count = readings.iter().filter(|r| r.state == "maintenance").count();
        assert!(
            maintenance_count > 0,
            "Maintenance scenario should have maintenance states"
        );
    }

    #[test]
    fn test_preset_scenarios_compile() {
        // Ensure all preset scenarios compile and produce data
        let scenarios: Vec<(&str, TelemetryConfig)> = vec![
            ("normal", TelemetryConfig::normal()),
            ("cpu_spikes", TelemetryConfig::cpu_spikes()),
            ("memory_leak", TelemetryConfig::memory_leak()),
            ("network_congestion", TelemetryConfig::network_congestion()),
            ("disk_pressure", TelemetryConfig::disk_pressure()),
            ("cascade_failure", TelemetryConfig::cascade_failure()),
            ("maintenance_window", TelemetryConfig::maintenance_window()),
            ("sensor_drift", TelemetryConfig::sensor_drift_scenario()),
            ("degradation_cycle", TelemetryConfig::degradation_cycle()),
            ("production", TelemetryConfig::production()),
            ("chaos", TelemetryConfig::chaos()),
        ];

        for (name, mut config) in scenarios {
            config.machine_count = 3;
            config.readings_per_machine = 10;
            config.seed = Some(42);

            let readings = generate_telemetry(&config);
            assert_eq!(
                readings.len(),
                30,
                "Scenario '{}' should produce 30 readings",
                name
            );
        }
    }
}
