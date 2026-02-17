# Retail Data Generation

Generate synthetic retail transaction and employee data with realistic business patterns.

## Overview

The retail generators create data suitable for:

- Sales analytics dashboards
- Business intelligence demos
- Retail forecasting models
- Customer segmentation analysis
- HR analytics and demographics

## Superstore Transactions

The `superstore()` function generates retail transaction records with realistic correlations between sales, quantity, discount, and profit.

### Basic Usage

```python
from superstore import superstore

# Generate 1000 transactions as a pandas DataFrame
df = superstore(count=1000)

# Generate as polars or dict
df = superstore(count=1000, output="polars")
data = superstore(count=1000, output="dict")
```

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `row_id` | int | Unique row identifier |
| `order_id` | str | Order identifier (format: CA-2024-XXXXXX) |
| `order_date` | date | Date of order |
| `ship_date` | date | Date of shipment |
| `ship_mode` | str | Shipping method (Standard, Express, etc.) |
| `customer_id` | str | Customer identifier |
| `customer_name` | str | Customer full name |
| `segment` | str | Customer segment (Consumer, Corporate, Home Office) |
| `city` | str | Delivery city |
| `state` | str | Delivery state |
| `postal_code` | str | Postal code |
| `region` | str | Region (East, West, Central, South) |
| `product_id` | str | Product identifier |
| `category` | str | Product category |
| `sub_category` | str | Product sub-category |
| `product_name` | str | Product name |
| `sales` | float | Transaction sales amount |
| `quantity` | int | Quantity ordered |
| `discount` | float | Discount applied (0.0 - 0.5) |
| `profit` | float | Transaction profit |

### Configuration

Use `SuperstoreConfig` for fine-grained control over data generation:

```python
from superstore import superstore, SuperstoreConfig

config = SuperstoreConfig(
    count=10000,
    seed=42,  # Reproducible output

    # Correlation settings
    sales_quantity_correlation=0.8,   # Higher quantities = higher sales
    sales_profit_correlation=0.9,     # Higher sales = higher profit
    discount_profit_correlation=-0.6, # Higher discounts = lower profit

    # Price formatting
    enable_price_points=True,  # Round to $X.99 values
)
df = superstore(config=config)
```

#### Seasonality Configuration

Model seasonal sales patterns:

```python
config = SuperstoreConfig(
    count=10000,
    seasonality={
        "enable": True,
        "q4_multiplier": 1.8,         # 80% more sales in Q4 (holidays)
        "summer_multiplier": 0.85,     # 15% fewer sales in summer
        "back_to_school_multiplier": 1.3,  # 30% more in Aug/Sep
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable` | `True` | Enable seasonal effects |
| `q4_multiplier` | `1.5` | Q4 (holiday) sales multiplier |
| `summer_multiplier` | `0.9` | Summer sales multiplier |
| `back_to_school_multiplier` | `1.2` | August/September multiplier |

#### Promotional Configuration

Model promotion and discount effects:

```python
config = SuperstoreConfig(
    count=10000,
    promotions={
        "enable": True,
        "discount_quantity_correlation": 0.6,  # Discounts increase quantity
        "price_elasticity": -1.0,  # Price sensitivity
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable` | `True` | Enable promotional patterns |
| `discount_quantity_correlation` | `0.5` | How much discounts increase quantity |
| `price_elasticity` | `-0.8` | Price elasticity of demand (-2 to 0) |

#### Customer Configuration

Model customer behavior and segmentation:

```python
config = SuperstoreConfig(
    count=10000,
    customers={
        "enable_cohorts": True,
        "repeat_customer_rate": 0.75,  # 75% repeat customers
        "vip_segment_rate": 0.15,      # 15% VIP customers
        "vip_order_multiplier": 2.5,   # VIPs spend 2.5x more
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_cohorts` | `True` | Enable customer cohort modeling |
| `repeat_customer_rate` | `0.7` | Fraction of repeat customers |
| `vip_segment_rate` | `0.1` | Fraction of VIP customers |
| `vip_order_multiplier` | `2.0` | VIP order value multiplier |

### Large Dataset Generation

For datasets larger than memory, use streaming or parallel generation:

```python
from superstore import superstoreStream, superstoreParallel

# Streaming: process chunks one at a time
for chunk in superstoreStream(count=10_000_000, chunk_size=100_000):
    process_and_save(chunk)

# Parallel: use all CPU cores for faster generation
df = superstoreParallel(count=1_000_000)
```

### Direct File Export

Export directly to files without loading into memory:

```python
from superstore import superstoreToCsv, superstoreToParquet, superstoreArrowIpc

# Export to different formats
superstoreToCsv("sales.csv", count=1_000_000)
superstoreToParquet("sales.parquet", count=1_000_000)
superstoreArrowIpc("sales.arrow", count=1_000_000)
```

---

## Employee Records

The `employees()` function generates realistic employee records with personal information.

### Basic Usage

```python
from superstore import employees

# Generate 500 employees
df = employees(count=500)

# Different output formats
df = employees(count=500, output="polars")
data = employees(count=500, output="dict")
```

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `employee_id` | int | Unique employee identifier |
| `first_name` | str | First name |
| `last_name` | str | Last name |
| `email` | str | Email address |
| `phone` | str | Phone number |
| `hire_date` | date | Date of hire |
| `department` | str | Department name |
| `job_title` | str | Job title |
| `salary` | float | Annual salary |
| `manager_id` | int | Manager's employee ID |
| `location` | str | Office location |

### Large Dataset Generation

```python
from superstore import employeesStream, employeesParallel

# Streaming generation
for chunk in employeesStream(count=100_000, chunk_size=10_000):
    process(chunk)

# Parallel generation
df = employeesParallel(count=50_000)
```

### Direct File Export

```python
from superstore import employeesToCsv, employeesToParquet, employeesArrowIpc

employeesToCsv("employees.csv", count=10_000)
employeesToParquet("employees.parquet", count=10_000)
employeesArrowIpc("employees.arrow", count=10_000)
```

---

## API Reference

See the full API documentation:

- [superstore()](api.md)
- [employees()](api.md)
- [SuperstoreConfig](api.md)
