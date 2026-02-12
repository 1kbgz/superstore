use chrono::{Duration as ChronoDuration, Utc};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
const REGIONS: [&str; 3] = ["na", "eu", "ap"];
const ZONES: [&str; 4] = ["A", "B", "C", "D"];

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
}
