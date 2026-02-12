//! Temporal dependencies for generating time-dependent data.
//!
//! This module provides models for generating data with temporal dependencies,
//! including autoregressive (AR) processes and Markov chains.

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Error type for temporal model operations.
#[derive(Debug, Clone)]
pub enum TemporalError {
    InvalidCoefficients(String),
    InvalidProbabilities(String),
    InsufficientHistory(String),
}

impl std::fmt::Display for TemporalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemporalError::InvalidCoefficients(msg) => {
                write!(f, "Invalid AR coefficients: {}", msg)
            }
            TemporalError::InvalidProbabilities(msg) => {
                write!(f, "Invalid transition probabilities: {}", msg)
            }
            TemporalError::InsufficientHistory(msg) => {
                write!(f, "Insufficient history: {}", msg)
            }
        }
    }
}

impl std::error::Error for TemporalError {}

/// AR(1) autoregressive model.
///
/// Generates values according to: x_t = phi * x_{t-1} + epsilon_t
/// where epsilon_t ~ N(0, sigma^2)
#[derive(Debug, Clone)]
pub struct AR1 {
    /// Autoregressive coefficient (-1 < phi < 1 for stationarity)
    pub phi: f64,
    /// Innovation (noise) standard deviation
    pub sigma: f64,
    /// Mean of the process
    pub mean: f64,
    /// Current state
    state: f64,
}

impl AR1 {
    /// Create a new AR(1) model.
    ///
    /// # Arguments
    /// * `phi` - Autoregressive coefficient (should be between -1 and 1 for stationarity)
    /// * `sigma` - Standard deviation of innovations
    /// * `mean` - Mean of the process (default: 0.0)
    pub fn new(phi: f64, sigma: f64, mean: f64) -> Result<Self, TemporalError> {
        if phi.abs() >= 1.0 {
            return Err(TemporalError::InvalidCoefficients(format!(
                "phi={} is not stationary (must be between -1 and 1)",
                phi
            )));
        }
        if sigma <= 0.0 {
            return Err(TemporalError::InvalidCoefficients(format!(
                "sigma={} must be positive",
                sigma
            )));
        }
        Ok(Self {
            phi,
            sigma,
            mean,
            state: mean,
        })
    }

    /// Reset the state to the mean.
    pub fn reset(&mut self) {
        self.state = self.mean;
    }

    /// Set the current state.
    pub fn set_state(&mut self, state: f64) {
        self.state = state;
    }

    /// Get the current state.
    pub fn state(&self) -> f64 {
        self.state
    }

    /// Generate the next value.
    pub fn next<R: Rng>(&mut self, rng: &mut R) -> f64 {
        let noise = Normal::new(0.0, self.sigma).unwrap();
        let innovation = noise.sample(rng);

        // x_t = mean + phi * (x_{t-1} - mean) + epsilon_t
        self.state = self.mean + self.phi * (self.state - self.mean) + innovation;
        self.state
    }

    /// Generate n values.
    pub fn sample_n<R: Rng>(&mut self, rng: &mut R, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next(rng)).collect()
    }

    /// Stationary variance of the process.
    pub fn stationary_variance(&self) -> f64 {
        self.sigma.powi(2) / (1.0 - self.phi.powi(2))
    }
}

/// AR(p) autoregressive model of order p.
///
/// Generates values according to: x_t = sum_i(phi_i * x_{t-i}) + epsilon_t
#[derive(Debug, Clone)]
pub struct ARp {
    /// Autoregressive coefficients [phi_1, phi_2, ..., phi_p]
    pub coefficients: Vec<f64>,
    /// Innovation (noise) standard deviation
    pub sigma: f64,
    /// Mean of the process
    pub mean: f64,
    /// History buffer (most recent first)
    history: Vec<f64>,
}

impl ARp {
    /// Create a new AR(p) model.
    ///
    /// # Arguments
    /// * `coefficients` - Vector of AR coefficients [phi_1, phi_2, ..., phi_p]
    /// * `sigma` - Standard deviation of innovations
    /// * `mean` - Mean of the process
    pub fn new(coefficients: Vec<f64>, sigma: f64, mean: f64) -> Result<Self, TemporalError> {
        if coefficients.is_empty() {
            return Err(TemporalError::InvalidCoefficients(
                "At least one coefficient required".to_string(),
            ));
        }
        if sigma <= 0.0 {
            return Err(TemporalError::InvalidCoefficients(format!(
                "sigma={} must be positive",
                sigma
            )));
        }

        let p = coefficients.len();
        let history = vec![mean; p];

        Ok(Self {
            coefficients,
            sigma,
            mean,
            history,
        })
    }

    /// Create an AR(1) model using this structure.
    pub fn ar1(phi: f64, sigma: f64, mean: f64) -> Result<Self, TemporalError> {
        Self::new(vec![phi], sigma, mean)
    }

    /// Create an AR(2) model.
    pub fn ar2(phi1: f64, phi2: f64, sigma: f64, mean: f64) -> Result<Self, TemporalError> {
        Self::new(vec![phi1, phi2], sigma, mean)
    }

    /// Order of the AR model.
    pub fn order(&self) -> usize {
        self.coefficients.len()
    }

    /// Reset the state to the mean.
    pub fn reset(&mut self) {
        self.history = vec![self.mean; self.coefficients.len()];
    }

    /// Set the history (most recent first).
    pub fn set_history(&mut self, history: Vec<f64>) -> Result<(), TemporalError> {
        if history.len() != self.coefficients.len() {
            return Err(TemporalError::InsufficientHistory(format!(
                "Need {} values for AR({}) model",
                self.coefficients.len(),
                self.coefficients.len()
            )));
        }
        self.history = history;
        Ok(())
    }

    /// Generate the next value.
    pub fn next<R: Rng>(&mut self, rng: &mut R) -> f64 {
        let noise = Normal::new(0.0, self.sigma).unwrap();
        let innovation = noise.sample(rng);

        // x_t = mean + sum_i(phi_i * (x_{t-i} - mean)) + epsilon_t
        let ar_component: f64 = self
            .coefficients
            .iter()
            .zip(self.history.iter())
            .map(|(phi, x)| phi * (x - self.mean))
            .sum();

        let value = self.mean + ar_component + innovation;

        // Update history (shift in the new value)
        self.history.pop();
        self.history.insert(0, value);

        value
    }

    /// Generate n values.
    pub fn sample_n<R: Rng>(&mut self, rng: &mut R, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next(rng)).collect()
    }
}

/// Markov chain for categorical state transitions.
#[derive(Debug, Clone)]
pub struct MarkovChain {
    /// Transition probability matrix (row = from state, col = to state)
    transition_matrix: Vec<Vec<f64>>,
    /// State labels
    states: Vec<String>,
    /// Current state index
    current_state: usize,
}

impl MarkovChain {
    /// Create a new Markov chain.
    ///
    /// # Arguments
    /// * `transition_matrix` - Row-stochastic transition matrix (rows must sum to 1)
    /// * `states` - State labels
    pub fn new(
        transition_matrix: Vec<Vec<f64>>,
        states: Vec<String>,
    ) -> Result<Self, TemporalError> {
        let n = states.len();

        if transition_matrix.len() != n {
            return Err(TemporalError::InvalidProbabilities(
                "Transition matrix rows don't match number of states".to_string(),
            ));
        }

        for (i, row) in transition_matrix.iter().enumerate() {
            if row.len() != n {
                return Err(TemporalError::InvalidProbabilities(format!(
                    "Row {} has wrong number of columns",
                    i
                )));
            }

            let row_sum: f64 = row.iter().sum();
            if (row_sum - 1.0).abs() > 1e-6 {
                return Err(TemporalError::InvalidProbabilities(format!(
                    "Row {} sums to {} (should be 1.0)",
                    i, row_sum
                )));
            }

            for (j, &prob) in row.iter().enumerate() {
                if prob < 0.0 || prob > 1.0 {
                    return Err(TemporalError::InvalidProbabilities(format!(
                        "Invalid probability {} at [{}, {}]",
                        prob, i, j
                    )));
                }
            }
        }

        Ok(Self {
            transition_matrix,
            states,
            current_state: 0,
        })
    }

    /// Create a simple two-state Markov chain.
    pub fn two_state(
        state_a: &str,
        state_b: &str,
        prob_a_to_b: f64,
        prob_b_to_a: f64,
    ) -> Result<Self, TemporalError> {
        if prob_a_to_b < 0.0 || prob_a_to_b > 1.0 || prob_b_to_a < 0.0 || prob_b_to_a > 1.0 {
            return Err(TemporalError::InvalidProbabilities(
                "Probabilities must be between 0 and 1".to_string(),
            ));
        }

        Self::new(
            vec![
                vec![1.0 - prob_a_to_b, prob_a_to_b],
                vec![prob_b_to_a, 1.0 - prob_b_to_a],
            ],
            vec![state_a.to_string(), state_b.to_string()],
        )
    }

    /// Get all states.
    pub fn states(&self) -> &[String] {
        &self.states
    }

    /// Get current state.
    pub fn current_state(&self) -> &str {
        &self.states[self.current_state]
    }

    /// Get current state index.
    pub fn current_state_index(&self) -> usize {
        self.current_state
    }

    /// Set current state by index.
    pub fn set_state(&mut self, index: usize) -> Result<(), TemporalError> {
        if index >= self.states.len() {
            return Err(TemporalError::InvalidProbabilities(format!(
                "State index {} out of range",
                index
            )));
        }
        self.current_state = index;
        Ok(())
    }

    /// Set current state by name.
    pub fn set_state_by_name(&mut self, state: &str) -> Result<(), TemporalError> {
        match self.states.iter().position(|s| s == state) {
            Some(idx) => {
                self.current_state = idx;
                Ok(())
            }
            None => Err(TemporalError::InvalidProbabilities(format!(
                "Unknown state: {}",
                state
            ))),
        }
    }

    /// Generate the next state.
    pub fn next<R: Rng>(&mut self, rng: &mut R) -> &str {
        let probs = &self.transition_matrix[self.current_state];
        let r: f64 = rng.gen();

        let mut cumsum = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if r < cumsum {
                self.current_state = i;
                break;
            }
        }

        &self.states[self.current_state]
    }

    /// Generate the next state and return its index.
    pub fn next_index<R: Rng>(&mut self, rng: &mut R) -> usize {
        self.next(rng);
        self.current_state
    }

    /// Generate n state transitions.
    pub fn sample_n<R: Rng>(&mut self, rng: &mut R, n: usize) -> Vec<String> {
        (0..n).map(|_| self.next(rng).to_string()).collect()
    }

    /// Generate n state transitions as indices.
    pub fn sample_n_indices<R: Rng>(&mut self, rng: &mut R, n: usize) -> Vec<usize> {
        (0..n).map(|_| self.next_index(rng)).collect()
    }

    /// Get stationary distribution (long-run probabilities).
    pub fn stationary_distribution(&self) -> Vec<f64> {
        // Use power iteration to find stationary distribution
        let n = self.states.len();
        let mut pi = vec![1.0 / n as f64; n];

        for _ in 0..1000 {
            let mut new_pi = vec![0.0; n];
            for j in 0..n {
                for i in 0..n {
                    new_pi[j] += pi[i] * self.transition_matrix[i][j];
                }
            }
            pi = new_pi;
        }

        pi
    }
}

/// Random walk model.
#[derive(Debug, Clone)]
pub struct RandomWalk {
    /// Step standard deviation
    pub sigma: f64,
    /// Current position
    position: f64,
    /// Optional drift (mean step size)
    pub drift: f64,
}

impl RandomWalk {
    /// Create a new random walk.
    pub fn new(sigma: f64, start: f64, drift: f64) -> Result<Self, TemporalError> {
        if sigma <= 0.0 {
            return Err(TemporalError::InvalidCoefficients(format!(
                "sigma={} must be positive",
                sigma
            )));
        }
        Ok(Self {
            sigma,
            position: start,
            drift,
        })
    }

    /// Create a simple random walk without drift.
    pub fn simple(sigma: f64, start: f64) -> Result<Self, TemporalError> {
        Self::new(sigma, start, 0.0)
    }

    /// Get current position.
    pub fn position(&self) -> f64 {
        self.position
    }

    /// Set current position.
    pub fn set_position(&mut self, position: f64) {
        self.position = position;
    }

    /// Generate the next position.
    pub fn next<R: Rng>(&mut self, rng: &mut R) -> f64 {
        let noise = Normal::new(self.drift, self.sigma).unwrap();
        self.position += noise.sample(rng);
        self.position
    }

    /// Generate n positions.
    pub fn sample_n<R: Rng>(&mut self, rng: &mut R, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next(rng)).collect()
    }
}

/// Exponential smoothing generator.
///
/// Useful for generating data with smoothly changing trends.
#[derive(Debug, Clone)]
pub struct ExponentialSmoothing {
    /// Smoothing factor (0 < alpha <= 1)
    pub alpha: f64,
    /// Innovation standard deviation
    pub sigma: f64,
    /// Current smoothed value
    smoothed: f64,
}

impl ExponentialSmoothing {
    /// Create a new exponential smoothing model.
    ///
    /// # Arguments
    /// * `alpha` - Smoothing factor (0 < alpha <= 1). Smaller = smoother.
    /// * `sigma` - Standard deviation of random innovations
    /// * `initial` - Initial smoothed value
    pub fn new(alpha: f64, sigma: f64, initial: f64) -> Result<Self, TemporalError> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(TemporalError::InvalidCoefficients(format!(
                "alpha={} must be in (0, 1]",
                alpha
            )));
        }
        if sigma <= 0.0 {
            return Err(TemporalError::InvalidCoefficients(format!(
                "sigma={} must be positive",
                sigma
            )));
        }
        Ok(Self {
            alpha,
            sigma,
            smoothed: initial,
        })
    }

    /// Generate the next value.
    pub fn next<R: Rng>(&mut self, rng: &mut R) -> f64 {
        let noise = Normal::new(0.0, self.sigma).unwrap();
        let observation = self.smoothed + noise.sample(rng);
        self.smoothed = self.alpha * observation + (1.0 - self.alpha) * self.smoothed;
        observation
    }

    /// Generate n values.
    pub fn sample_n<R: Rng>(&mut self, rng: &mut R, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next(rng)).collect()
    }

    /// Get current smoothed value.
    pub fn smoothed(&self) -> f64 {
        self.smoothed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_ar1_stationarity() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut ar = AR1::new(0.7, 1.0, 0.0).unwrap();

        let samples = ar.sample_n(&mut rng, 1000);
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;

        // Mean should be close to 0
        assert!(mean.abs() < 1.0);
    }

    #[test]
    fn test_ar1_autocorrelation() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut ar = AR1::new(0.9, 0.5, 0.0).unwrap();

        let samples = ar.sample_n(&mut rng, 1000);

        // Compute lag-1 autocorrelation
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let var: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;

        let cov: f64 = samples
            .iter()
            .skip(1)
            .zip(samples.iter())
            .map(|(x, y)| (x - mean) * (y - mean))
            .sum::<f64>()
            / (samples.len() - 1) as f64;

        let autocorr = cov / var;

        // Should be close to phi=0.9
        assert!((autocorr - 0.9).abs() < 0.15);
    }

    #[test]
    fn test_arp_ar2() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut ar = ARp::ar2(0.5, 0.3, 1.0, 0.0).unwrap();

        let samples = ar.sample_n(&mut rng, 100);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_markov_chain() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut mc = MarkovChain::two_state("sunny", "rainy", 0.3, 0.5).unwrap();

        let states = mc.sample_n(&mut rng, 100);
        assert_eq!(states.len(), 100);

        // Check that both states appear
        let sunny_count = states.iter().filter(|s| *s == "sunny").count();
        let rainy_count = states.iter().filter(|s| *s == "rainy").count();
        assert!(sunny_count > 0 && rainy_count > 0);
    }

    #[test]
    fn test_markov_stationary() {
        let mc = MarkovChain::two_state("A", "B", 0.3, 0.5).unwrap();
        let stationary = mc.stationary_distribution();

        // Stationary distribution: pi_A = 0.5/(0.3+0.5), pi_B = 0.3/(0.3+0.5)
        assert!((stationary[0] - 0.625).abs() < 0.05);
        assert!((stationary[1] - 0.375).abs() < 0.05);
    }

    #[test]
    fn test_random_walk() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut rw = RandomWalk::simple(1.0, 0.0).unwrap();

        let samples = rw.sample_n(&mut rng, 100);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_random_walk_with_drift() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut rw = RandomWalk::new(0.5, 0.0, 1.0).unwrap();

        let samples = rw.sample_n(&mut rng, 100);

        // With positive drift, should generally increase
        assert!(samples.last().unwrap() > &50.0);
    }

    #[test]
    fn test_exponential_smoothing() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut es = ExponentialSmoothing::new(0.3, 2.0, 100.0).unwrap();

        let samples = es.sample_n(&mut rng, 100);
        assert_eq!(samples.len(), 100);

        // Values should be relatively smooth (not jumping wildly)
        let max_diff = samples
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0_f64, f64::max);

        assert!(max_diff < 20.0);
    }
}
