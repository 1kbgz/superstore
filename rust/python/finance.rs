use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use superstore::finance::{
    generate_finance_data, generate_multi_asset_prices, generate_options_chain,
    generate_stock_prices, FinanceConfig, OhlcvBar, OhlcvConfig, OptionQuote, OptionsConfig,
    StockConfig,
};

// =============================================================================
// Helper Functions for creating DataFrames
// =============================================================================

/// Create pandas DataFrame from OhlcvBar rows
fn create_ohlcv_pandas(py: Python<'_>, rows: &[OhlcvBar]) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let data = PyDict::new(py);

    let dates: Vec<&str> = rows.iter().map(|r| r.date.as_str()).collect();
    let tickers: Vec<&str> = rows.iter().map(|r| r.ticker.as_str()).collect();
    let opens: Vec<f64> = rows.iter().map(|r| r.open).collect();
    let highs: Vec<f64> = rows.iter().map(|r| r.high).collect();
    let lows: Vec<f64> = rows.iter().map(|r| r.low).collect();
    let closes: Vec<f64> = rows.iter().map(|r| r.close).collect();
    let volumes: Vec<u64> = rows.iter().map(|r| r.volume).collect();
    let vwaps: Vec<f64> = rows.iter().map(|r| r.vwap).collect();
    let returns: Vec<f64> = rows.iter().map(|r| r.returns).collect();

    data.set_item("date", PyList::new(py, &dates)?)?;
    data.set_item("ticker", PyList::new(py, &tickers)?)?;
    data.set_item("open", PyList::new(py, &opens)?)?;
    data.set_item("high", PyList::new(py, &highs)?)?;
    data.set_item("low", PyList::new(py, &lows)?)?;
    data.set_item("close", PyList::new(py, &closes)?)?;
    data.set_item("volume", PyList::new(py, &volumes)?)?;
    data.set_item("vwap", PyList::new(py, &vwaps)?)?;
    data.set_item("returns", PyList::new(py, &returns)?)?;

    let df = pandas.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create polars DataFrame from OhlcvBar rows
fn create_ohlcv_polars(py: Python<'_>, rows: &[OhlcvBar]) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let data = PyDict::new(py);

    let dates: Vec<&str> = rows.iter().map(|r| r.date.as_str()).collect();
    let tickers: Vec<&str> = rows.iter().map(|r| r.ticker.as_str()).collect();
    let opens: Vec<f64> = rows.iter().map(|r| r.open).collect();
    let highs: Vec<f64> = rows.iter().map(|r| r.high).collect();
    let lows: Vec<f64> = rows.iter().map(|r| r.low).collect();
    let closes: Vec<f64> = rows.iter().map(|r| r.close).collect();
    let volumes: Vec<u64> = rows.iter().map(|r| r.volume).collect();
    let vwaps: Vec<f64> = rows.iter().map(|r| r.vwap).collect();
    let returns: Vec<f64> = rows.iter().map(|r| r.returns).collect();

    data.set_item("date", PyList::new(py, &dates)?)?;
    data.set_item("ticker", PyList::new(py, &tickers)?)?;
    data.set_item("open", PyList::new(py, &opens)?)?;
    data.set_item("high", PyList::new(py, &highs)?)?;
    data.set_item("low", PyList::new(py, &lows)?)?;
    data.set_item("close", PyList::new(py, &closes)?)?;
    data.set_item("volume", PyList::new(py, &volumes)?)?;
    data.set_item("vwap", PyList::new(py, &vwaps)?)?;
    data.set_item("returns", PyList::new(py, &returns)?)?;

    let df = polars.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create dict from OhlcvBar rows
fn create_ohlcv_dict(py: Python<'_>, rows: &[OhlcvBar]) -> PyResult<Py<PyAny>> {
    let data = PyDict::new(py);

    let dates: Vec<&str> = rows.iter().map(|r| r.date.as_str()).collect();
    let tickers: Vec<&str> = rows.iter().map(|r| r.ticker.as_str()).collect();
    let opens: Vec<f64> = rows.iter().map(|r| r.open).collect();
    let highs: Vec<f64> = rows.iter().map(|r| r.high).collect();
    let lows: Vec<f64> = rows.iter().map(|r| r.low).collect();
    let closes: Vec<f64> = rows.iter().map(|r| r.close).collect();
    let volumes: Vec<u64> = rows.iter().map(|r| r.volume).collect();
    let vwaps: Vec<f64> = rows.iter().map(|r| r.vwap).collect();
    let returns: Vec<f64> = rows.iter().map(|r| r.returns).collect();

    data.set_item("date", PyList::new(py, &dates)?)?;
    data.set_item("ticker", PyList::new(py, &tickers)?)?;
    data.set_item("open", PyList::new(py, &opens)?)?;
    data.set_item("high", PyList::new(py, &highs)?)?;
    data.set_item("low", PyList::new(py, &lows)?)?;
    data.set_item("close", PyList::new(py, &closes)?)?;
    data.set_item("volume", PyList::new(py, &volumes)?)?;
    data.set_item("vwap", PyList::new(py, &vwaps)?)?;
    data.set_item("returns", PyList::new(py, &returns)?)?;

    Ok(data.into())
}

/// Create pandas DataFrame from OptionQuote rows
fn create_options_pandas(py: Python<'_>, rows: &[OptionQuote]) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let data = PyDict::new(py);

    let dates: Vec<&str> = rows.iter().map(|r| r.date.as_str()).collect();
    let tickers: Vec<&str> = rows.iter().map(|r| r.ticker.as_str()).collect();
    let option_types: Vec<&str> = rows.iter().map(|r| r.option_type.as_str()).collect();
    let underlying_prices: Vec<f64> = rows.iter().map(|r| r.underlying_price).collect();
    let strikes: Vec<f64> = rows.iter().map(|r| r.strike).collect();
    let expirations: Vec<&str> = rows.iter().map(|r| r.expiration.as_str()).collect();
    let days_to_expiry: Vec<u32> = rows.iter().map(|r| r.days_to_expiry).collect();
    let prices: Vec<f64> = rows.iter().map(|r| r.price).collect();
    let deltas: Vec<f64> = rows.iter().map(|r| r.delta).collect();
    let gammas: Vec<f64> = rows.iter().map(|r| r.gamma).collect();
    let thetas: Vec<f64> = rows.iter().map(|r| r.theta).collect();
    let vegas: Vec<f64> = rows.iter().map(|r| r.vega).collect();
    let implied_vols: Vec<f64> = rows.iter().map(|r| r.implied_vol).collect();

    data.set_item("date", PyList::new(py, &dates)?)?;
    data.set_item("ticker", PyList::new(py, &tickers)?)?;
    data.set_item("option_type", PyList::new(py, &option_types)?)?;
    data.set_item("underlying_price", PyList::new(py, &underlying_prices)?)?;
    data.set_item("strike", PyList::new(py, &strikes)?)?;
    data.set_item("expiration", PyList::new(py, &expirations)?)?;
    data.set_item("days_to_expiry", PyList::new(py, &days_to_expiry)?)?;
    data.set_item("price", PyList::new(py, &prices)?)?;
    data.set_item("delta", PyList::new(py, &deltas)?)?;
    data.set_item("gamma", PyList::new(py, &gammas)?)?;
    data.set_item("theta", PyList::new(py, &thetas)?)?;
    data.set_item("vega", PyList::new(py, &vegas)?)?;
    data.set_item("implied_vol", PyList::new(py, &implied_vols)?)?;

    let df = pandas.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

/// Create polars DataFrame from OptionQuote rows
fn create_options_polars(py: Python<'_>, rows: &[OptionQuote]) -> PyResult<Py<PyAny>> {
    let polars = py.import("polars")?;
    let data = PyDict::new(py);

    let dates: Vec<&str> = rows.iter().map(|r| r.date.as_str()).collect();
    let tickers: Vec<&str> = rows.iter().map(|r| r.ticker.as_str()).collect();
    let option_types: Vec<&str> = rows.iter().map(|r| r.option_type.as_str()).collect();
    let underlying_prices: Vec<f64> = rows.iter().map(|r| r.underlying_price).collect();
    let strikes: Vec<f64> = rows.iter().map(|r| r.strike).collect();
    let expirations: Vec<&str> = rows.iter().map(|r| r.expiration.as_str()).collect();
    let days_to_expiry: Vec<u32> = rows.iter().map(|r| r.days_to_expiry).collect();
    let prices: Vec<f64> = rows.iter().map(|r| r.price).collect();
    let deltas: Vec<f64> = rows.iter().map(|r| r.delta).collect();
    let gammas: Vec<f64> = rows.iter().map(|r| r.gamma).collect();
    let thetas: Vec<f64> = rows.iter().map(|r| r.theta).collect();
    let vegas: Vec<f64> = rows.iter().map(|r| r.vega).collect();
    let implied_vols: Vec<f64> = rows.iter().map(|r| r.implied_vol).collect();

    data.set_item("date", PyList::new(py, &dates)?)?;
    data.set_item("ticker", PyList::new(py, &tickers)?)?;
    data.set_item("option_type", PyList::new(py, &option_types)?)?;
    data.set_item("underlying_price", PyList::new(py, &underlying_prices)?)?;
    data.set_item("strike", PyList::new(py, &strikes)?)?;
    data.set_item("expiration", PyList::new(py, &expirations)?)?;
    data.set_item("days_to_expiry", PyList::new(py, &days_to_expiry)?)?;
    data.set_item("price", PyList::new(py, &prices)?)?;
    data.set_item("delta", PyList::new(py, &deltas)?)?;
    data.set_item("gamma", PyList::new(py, &gammas)?)?;
    data.set_item("theta", PyList::new(py, &thetas)?)?;
    data.set_item("vega", PyList::new(py, &vegas)?)?;
    data.set_item("implied_vol", PyList::new(py, &implied_vols)?)?;

    let df = polars.call_method1("DataFrame", (data,))?;
    Ok(df.into())
}

// =============================================================================
// Config Parsing
// =============================================================================

/// Parse FinanceConfig from Python dict
fn parse_finance_config(dict: &Bound<'_, PyDict>) -> PyResult<(FinanceConfig, String)> {
    let ndays: usize = dict
        .get_item("ndays")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(252);

    let n_assets: usize = dict
        .get_item("n_assets")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(1);

    let output: String = dict
        .get_item("output")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_else(|| "pandas".to_string());

    let seed: Option<u64> = dict.get_item("seed")?.and_then(|v| v.extract().ok());

    let start_date: Option<String> = dict.get_item("start_date")?.and_then(|v| v.extract().ok());

    let asset_correlation: f64 = dict
        .get_item("asset_correlation")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(0.5);

    let tickers: Vec<String> = dict
        .get_item("tickers")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_else(|| vec!["AAPL".to_string()]);

    // Parse nested StockConfig
    let stock = if let Some(stock_val) = dict.get_item("stock")? {
        if let Ok(stock_dict) = stock_val.downcast::<PyDict>() {
            let annual_drift: f64 = stock_dict
                .get_item("annual_drift")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.08);
            let annual_volatility: f64 = stock_dict
                .get_item("annual_volatility")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.20);
            let initial_price: f64 = stock_dict
                .get_item("initial_price")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(100.0);
            let enable_jumps: bool = stock_dict
                .get_item("enable_jumps")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(false);
            let jump_probability: f64 = stock_dict
                .get_item("jump_probability")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.02);
            let jump_mean: f64 = stock_dict
                .get_item("jump_mean")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.0);
            let jump_stddev: f64 = stock_dict
                .get_item("jump_stddev")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.05);
            StockConfig {
                annual_drift,
                annual_volatility,
                initial_price,
                enable_jumps,
                jump_probability,
                jump_mean,
                jump_stddev,
            }
        } else {
            StockConfig::default()
        }
    } else {
        StockConfig::default()
    };

    // Parse nested OhlcvConfig
    let ohlcv = if let Some(ohlcv_val) = dict.get_item("ohlcv")? {
        if let Ok(ohlcv_dict) = ohlcv_val.downcast::<PyDict>() {
            let avg_volume: u64 = ohlcv_dict
                .get_item("avg_volume")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(1_000_000);
            let volume_volatility: f64 = ohlcv_dict
                .get_item("volume_volatility")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.5);
            let intraday_volatility: f64 = ohlcv_dict
                .get_item("intraday_volatility")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.02);
            let volume_price_correlation: f64 = ohlcv_dict
                .get_item("volume_price_correlation")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.3);
            OhlcvConfig {
                avg_volume,
                volume_volatility,
                intraday_volatility,
                volume_price_correlation,
            }
        } else {
            OhlcvConfig::default()
        }
    } else {
        OhlcvConfig::default()
    };

    // Parse nested OptionsConfig
    let options = if let Some(options_val) = dict.get_item("options")? {
        if let Ok(options_dict) = options_val.downcast::<PyDict>() {
            let risk_free_rate: f64 = options_dict
                .get_item("risk_free_rate")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.05);
            let dividend_yield: f64 = options_dict
                .get_item("dividend_yield")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.02);
            let expirations: Vec<u32> = options_dict
                .get_item("expirations")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_else(|| vec![7, 14, 30, 60, 90]);
            let strike_offsets: Vec<f64> = options_dict
                .get_item("strike_offsets")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_else(|| vec![0.90, 0.95, 0.97, 1.0, 1.03, 1.05, 1.10]);
            OptionsConfig {
                risk_free_rate,
                dividend_yield,
                expirations,
                strike_offsets,
            }
        } else {
            OptionsConfig::default()
        }
    } else {
        OptionsConfig::default()
    };

    let config = FinanceConfig {
        ndays,
        n_assets,
        seed,
        start_date,
        asset_correlation,
        stock,
        ohlcv,
        options,
        tickers,
    };

    Ok((config, output))
}

// =============================================================================
// Python Functions
// =============================================================================

/// Generate stock price data (OHLCV bars).
///
/// Returns realistic stock price data using Geometric Brownian Motion
/// with optional jump diffusion. Includes OHLCV bars with realistic
/// intraday relationships and volume patterns.
///
/// # Arguments
/// * `config` - Optional FinanceConfig or dict with generation parameters
///
/// # Returns
/// * DataFrame (pandas/polars) or dict of OHLCV bars
#[pyfunction]
#[pyo3(signature = (config=None))]
pub fn stock_prices(py: Python<'_>, config: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    let (finance_config, output_format) = if let Some(dict) = config {
        parse_finance_config(dict)?
    } else {
        (FinanceConfig::default(), "pandas".to_string())
    };

    let rows = if finance_config.n_assets > 1 {
        generate_multi_asset_prices(&finance_config)
    } else {
        generate_stock_prices(&finance_config)
    };

    match output_format.to_lowercase().as_str() {
        "polars" => create_ohlcv_polars(py, &rows),
        "dict" => create_ohlcv_dict(py, &rows),
        _ => create_ohlcv_pandas(py, &rows),
    }
}

/// Generate options chain with Greeks.
///
/// Returns options data including Black-Scholes pricing and Greeks
/// (delta, gamma, theta, vega) for various strikes and expirations.
///
/// # Arguments
/// * `config` - Optional FinanceConfig or dict with generation parameters
/// * `spot_price` - Current underlying price (default: 100.0)
/// * `date` - Pricing date (default: "2024-01-15")
///
/// # Returns
/// * DataFrame (pandas/polars) of options chain
#[pyfunction]
#[pyo3(signature = (config=None, spot_price=None, date=None))]
pub fn options_chain(
    py: Python<'_>,
    config: Option<&Bound<'_, PyDict>>,
    spot_price: Option<f64>,
    date: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let (finance_config, output_format) = if let Some(dict) = config {
        parse_finance_config(dict)?
    } else {
        (FinanceConfig::default(), "pandas".to_string())
    };

    let spot = spot_price.unwrap_or(100.0);
    let pricing_date = date.unwrap_or("2024-01-15");

    let rows = generate_options_chain(&finance_config, spot, pricing_date);

    match output_format.to_lowercase().as_str() {
        "polars" => create_options_polars(py, &rows),
        _ => create_options_pandas(py, &rows),
    }
}

/// Generate complete finance dataset: stock prices + options.
///
/// Returns both OHLCV price data and an options chain for the last
/// trading day.
///
/// # Arguments
/// * `config` - Optional FinanceConfig or dict with generation parameters
///
/// # Returns
/// * Tuple of (prices_df, options_df)
#[pyfunction]
#[pyo3(signature = (config=None))]
pub fn finance(
    py: Python<'_>,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let (finance_config, output_format) = if let Some(dict) = config {
        parse_finance_config(dict)?
    } else {
        (FinanceConfig::default(), "pandas".to_string())
    };

    let (prices, options) = generate_finance_data(&finance_config);

    let prices_df = match output_format.to_lowercase().as_str() {
        "polars" => create_ohlcv_polars(py, &prices)?,
        _ => create_ohlcv_pandas(py, &prices)?,
    };

    let options_df = match output_format.to_lowercase().as_str() {
        "polars" => create_options_polars(py, &options)?,
        _ => create_options_pandas(py, &options)?,
    };

    Ok((prices_df, options_df))
}
