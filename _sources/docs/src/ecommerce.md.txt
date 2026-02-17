````markdown
# E-commerce Data Generation

Generate synthetic e-commerce data with realistic user sessions, shopping carts, orders, and customer RFM metrics.

## Overview

The e-commerce generators create data suitable for:

- Conversion funnel analysis
- Customer segmentation (RFM)
- Cart abandonment studies
- Session behavior analytics
- A/B testing simulations

## Quick Start

```python
from superstore import ecommerce_data, EcommerceConfig

# Generate complete e-commerce dataset
data = ecommerce_data()

# Access individual tables
sessions_df = data["sessions"]
products_df = data["products"]
cart_events_df = data["cart_events"]
orders_df = data["orders"]
customers_df = data["customers"]
```

---

## Session Data

The `ecommerce_sessions()` function generates user session records with MarkovChain-based page navigation.

### Basic Usage

```python
from superstore import ecommerce_sessions

# Generate 1000 sessions
df = ecommerce_sessions(count=1000, seed=42)
```

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | str | Unique session identifier |
| `user_id` | str | User identifier |
| `start_time` | datetime | Session start timestamp |
| `end_time` | datetime | Session end timestamp |
| `duration_seconds` | int | Total session duration |
| `device_type` | str | Device type (desktop, mobile, tablet) |
| `browser` | str | Browser name |
| `traffic_source` | str | Traffic source (organic, paid_search, direct, social, email, referral, affiliate) |
| `landing_page` | str | First page viewed |
| `pages_viewed` | int | Number of pages viewed |
| `bounced` | bool | Whether session was a bounce (single page) |
| `converted` | bool | Whether session resulted in purchase |
| `total_value` | float | Total purchase value (0 if not converted) |

### MarkovChain Session States

Sessions navigate through states using a configurable transition matrix:

```
landing → browse → view_product → add_to_cart → view_cart → checkout_start → checkout_payment → purchase
                 ↘              ↘              ↘           ↘                ↘
                  exit           exit           exit         exit             exit
```

---

## Product Catalog

The `ecommerce_products()` function generates a product catalog with realistic pricing.

### Basic Usage

```python
from superstore import ecommerce_products

# Generate 500 products
df = ecommerce_products(count=500, seed=42)
```

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `product_id` | str | Unique product identifier |
| `name` | str | Product name |
| `category` | str | Product category |
| `subcategory` | str | Product subcategory |
| `price` | float | Product price (log-normal distribution) |
| `rating` | float | Average rating (1.0-5.0) |
| `review_count` | int | Number of reviews |
| `in_stock` | bool | Stock availability |

---

## Cart Events

Cart events track user interactions with shopping carts.

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | str | Unique event identifier |
| `session_id` | str | Associated session |
| `user_id` | str | User identifier |
| `timestamp` | datetime | Event timestamp |
| `event_type` | str | Event type (add, remove, update_quantity) |
| `product_id` | str | Product identifier |
| `quantity` | int | Item quantity |
| `unit_price` | float | Price per unit |
| `total_price` | float | Total line price |

---

## Orders

Completed purchase orders.

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `order_id` | str | Unique order identifier |
| `user_id` | str | Customer identifier |
| `session_id` | str | Originating session |
| `order_time` | datetime | Order timestamp |
| `total_items` | int | Number of items |
| `subtotal` | float | Subtotal before tax/shipping |
| `discount` | float | Discount amount |
| `tax` | float | Tax amount |
| `shipping` | float | Shipping cost |
| `total` | float | Final order total |
| `payment_method` | str | Payment method (credit_card, paypal, apple_pay, etc.) |
| `status` | str | Order status (completed, processing, shipped) |

---

## Customers with RFM Metrics

Customer records include RFM (Recency, Frequency, Monetary) segmentation.

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | str | Unique customer identifier |
| `email` | str | Customer email |
| `first_order_date` | date | First purchase date |
| `last_order_date` | date | Most recent purchase date |
| `total_orders` | int | Lifetime order count |
| `total_spent` | float | Lifetime spend |
| `avg_order_value` | float | Average order value |
| `rfm_recency` | int | Recency score (1-5) |
| `rfm_frequency` | int | Frequency score (1-5) |
| `rfm_monetary` | float | Monetary value |
| `rfm_score` | str | Combined RFM score (e.g., "544") |
| `rfm_segment` | str | Customer segment label |

### RFM Segments

| Segment | Description |
|---------|-------------|
| Champions | High recency, frequency, and monetary |
| Loyal Customers | High frequency and monetary |
| Potential Loyalists | Recent customers with medium frequency |
| New Customers | Very recent, low frequency |
| At Risk | Previously good customers, declining |
| Need Attention | Below average across metrics |
| Hibernating | Low activity, long time since purchase |
| Lost | No recent activity, low value |

---

## Configuration

Use `EcommerceConfig` for detailed control:

```python
from superstore import ecommerce_data, EcommerceConfig

config = EcommerceConfig(
    sessions=10000,    # Number of sessions
    customers=2000,    # Number of unique customers
    days=30,           # Time span in days
    seed=42,           # Reproducibility
)
data = ecommerce_data(config=config.model_dump())
```

### Session Configuration

Control user session behavior:

```python
config = EcommerceConfig(
    sessions=5000,
    session={
        "avg_pages_per_session": 5.0,
        "cart_add_probability": 0.15,
        "checkout_start_probability": 0.40,
        "purchase_completion_probability": 0.65,
        "avg_session_duration_seconds": 300,
        "enable_bounces": True,
        "bounce_rate": 0.35,
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `avg_pages_per_session` | `5.0` | Average pages viewed |
| `cart_add_probability` | `0.15` | P(add to cart \| view product) |
| `checkout_start_probability` | `0.40` | P(start checkout \| view cart) |
| `purchase_completion_probability` | `0.65` | P(purchase \| checkout start) |
| `avg_session_duration_seconds` | `300` | Average session length |
| `enable_bounces` | `True` | Enable single-page bounces |
| `bounce_rate` | `0.35` | Bounce probability |

### Cart Configuration

Configure cart behavior and abandonment:

```python
config = EcommerceConfig(
    cart={
        "avg_items_per_cart": 2.5,
        "remove_probability": 0.10,
        "quantity_update_probability": 0.05,
        "max_items": 20,
        "enable_abandonment": True,
        "abandonment_rate": 0.70,
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `avg_items_per_cart` | `2.5` | Average items added |
| `remove_probability` | `0.10` | P(remove item) |
| `quantity_update_probability` | `0.05` | P(update quantity) |
| `max_items` | `20` | Maximum cart size |
| `enable_abandonment` | `True` | Enable cart abandonment |
| `abandonment_rate` | `0.70` | Cart abandonment rate |

### Catalog Configuration

Configure the product catalog:

```python
config = EcommerceConfig(
    catalog={
        "num_products": 500,
        "min_price": 5.0,
        "max_price": 1000.0,
        "lognormal_prices": True,
        "categories": ["Electronics", "Clothing", "Home", "Sports"],
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_products` | `500` | Number of products |
| `min_price` | `5.0` | Minimum product price |
| `max_price` | `1000.0` | Maximum product price |
| `lognormal_prices` | `True` | Use log-normal price distribution |
| `categories` | `[...]` | Product categories |

### RFM Configuration

Configure RFM analysis parameters:

```python
config = EcommerceConfig(
    rfm={
        "enable": True,
        "recency_window_days": 365,
        "num_buckets": 5,
        "pareto_shape": 1.5,
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable` | `True` | Calculate RFM metrics |
| `recency_window_days` | `365` | Recency lookback period |
| `num_buckets` | `5` | Number of RFM score buckets |
| `pareto_shape` | `1.5` | Shape parameter for 80/20 distribution |

### Funnel Configuration

Configure conversion funnel tracking:

```python
config = EcommerceConfig(
    funnel={
        "enable": True,
        "stages": ["visit", "view_product", "add_to_cart", "checkout", "purchase"],
        "time_of_day_effects": True,
        "day_of_week_effects": True,
    }
)
```

---

## Complete Example

```python
from superstore import ecommerce_data, EcommerceConfig

config = EcommerceConfig(
    sessions=20000,
    customers=5000,
    days=90,
    seed=42,
    session={
        "avg_pages_per_session": 6.0,
        "bounce_rate": 0.30,
        "purchase_completion_probability": 0.70,
    },
    cart={
        "abandonment_rate": 0.65,
        "avg_items_per_cart": 3.0,
    },
    catalog={
        "num_products": 1000,
        "categories": ["Electronics", "Fashion", "Home", "Beauty", "Sports"],
    },
    rfm={
        "num_buckets": 5,
        "pareto_shape": 1.8,
    },
)

data = ecommerce_data(config=config.model_dump())

# Analyze conversion rates
sessions = data["sessions"]
conversion_rate = sessions["converted"].mean()
print(f"Conversion rate: {conversion_rate:.2%}")

# Segment customers by RFM
customers = data["customers"]
segment_counts = customers["rfm_segment"].value_counts()
print(segment_counts)
```

---

## API Reference

See the full API documentation:

- [ecommerce_sessions()](api.md)
- [ecommerce_products()](api.md)
- [ecommerce_data()](api.md)
- [EcommerceConfig](api.md)
````
