"""Tests for e-commerce data generators."""

import pytest

SESSIONS_COLUMNS = [
    "session_id",
    "user_id",
    "start_time",
    "end_time",
    "duration_seconds",
    "device_type",
    "browser",
    "traffic_source",
    "landing_page",
    "pages_viewed",
    "bounced",
    "converted",
    "total_value",
]

PRODUCTS_COLUMNS = [
    "product_id",
    "name",
    "category",
    "subcategory",
    "price",
    "rating",
    "review_count",
    "in_stock",
]

CART_EVENTS_COLUMNS = [
    "event_id",
    "session_id",
    "user_id",
    "timestamp",
    "event_type",
    "product_id",
    "quantity",
    "unit_price",
    "total_price",
]

ORDERS_COLUMNS = [
    "order_id",
    "user_id",
    "session_id",
    "order_time",
    "total_items",
    "subtotal",
    "discount",
    "tax",
    "shipping",
    "total",
    "payment_method",
    "status",
]

CUSTOMERS_COLUMNS = [
    "customer_id",
    "email",
    "first_order_date",
    "last_order_date",
    "total_orders",
    "total_spent",
    "avg_order_value",
    "rfm_recency",
    "rfm_frequency",
    "rfm_monetary",
    "rfm_score",
    "rfm_segment",
]


class TestEcommerceSessions:
    def test_sessions_pandas(self):
        import pandas as pd

        from superstore import ecommerce_sessions

        df = ecommerce_sessions(100, seed=42)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == SESSIONS_COLUMNS
        assert df.shape[0] == 100

    def test_sessions_polars(self):
        import polars as pl

        from superstore import ecommerce_sessions

        df = ecommerce_sessions(100, seed=42, output="polars")
        assert isinstance(df, pl.DataFrame)
        assert df.columns == SESSIONS_COLUMNS
        assert df.shape[0] == 100

    def test_sessions_dict(self):
        from superstore import ecommerce_sessions

        data = ecommerce_sessions(100, seed=42, output="dict")
        assert isinstance(data, dict)
        assert set(data.keys()) == set(SESSIONS_COLUMNS)
        assert len(data["session_id"]) == 100

    def test_sessions_reproducibility(self):
        from superstore import ecommerce_sessions

        df1 = ecommerce_sessions(50, seed=123)
        df2 = ecommerce_sessions(50, seed=123)
        assert df1["session_id"].tolist() == df2["session_id"].tolist()

    def test_sessions_content(self):
        from superstore import ecommerce_sessions

        df = ecommerce_sessions(200, seed=42)

        # Check device types are valid
        valid_devices = {"desktop", "mobile", "tablet"}
        assert set(df["device_type"].unique()).issubset(valid_devices)

        # Check traffic sources are valid
        valid_sources = {"organic", "paid_search", "direct", "social", "email", "referral", "affiliate"}
        assert set(df["traffic_source"].unique()).issubset(valid_sources)

        # Check bounced and converted are boolean
        assert df["bounced"].dtype == bool
        assert df["converted"].dtype == bool

        # Check duration is positive
        assert (df["duration_seconds"] >= 0).all()


class TestEcommerceProducts:
    def test_products_pandas(self):
        import pandas as pd

        from superstore import ecommerce_products

        df = ecommerce_products(50, seed=42)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == PRODUCTS_COLUMNS
        assert df.shape[0] == 50

    def test_products_polars(self):
        import polars as pl

        from superstore import ecommerce_products

        df = ecommerce_products(50, seed=42, output="polars")
        assert isinstance(df, pl.DataFrame)
        assert df.columns == PRODUCTS_COLUMNS
        assert df.shape[0] == 50

    def test_products_dict(self):
        from superstore import ecommerce_products

        data = ecommerce_products(50, seed=42, output="dict")
        assert isinstance(data, dict)
        assert set(data.keys()) == set(PRODUCTS_COLUMNS)
        assert len(data["product_id"]) == 50

    def test_products_content(self):
        from superstore import ecommerce_products

        df = ecommerce_products(100, seed=42)

        # Check price is positive
        assert (df["price"] > 0).all()

        # Check rating is in valid range
        assert (df["rating"] >= 1.0).all()
        assert (df["rating"] <= 5.0).all()

        # Check in_stock is boolean
        assert df["in_stock"].dtype == bool


class TestEcommerceData:
    def test_ecommerce_data_pandas(self):
        import pandas as pd

        from superstore import ecommerce_data

        data = ecommerce_data()
        assert isinstance(data, dict)
        assert "products" in data
        assert "sessions" in data
        assert "cart_events" in data
        assert "orders" in data
        assert "customers" in data

        assert isinstance(data["products"], pd.DataFrame)
        assert isinstance(data["sessions"], pd.DataFrame)

    def test_ecommerce_data_polars(self):
        import polars as pl

        from superstore import ecommerce_data

        data = ecommerce_data(output="polars")
        assert isinstance(data["products"], pl.DataFrame)
        assert isinstance(data["sessions"], pl.DataFrame)

    def test_ecommerce_data_dict(self):
        from superstore import ecommerce_data

        data = ecommerce_data(output="dict")
        assert isinstance(data["products"], dict)
        assert isinstance(data["sessions"], dict)

    def test_ecommerce_data_with_config(self):
        from superstore import EcommerceConfig, ecommerce_data

        config = EcommerceConfig(
            sessions=100,
            customers=50,
            seed=42,
        )
        data = ecommerce_data(config=config.model_dump())

        assert data["sessions"].shape[0] == 100
        assert list(data["products"].columns) == PRODUCTS_COLUMNS
        assert list(data["sessions"].columns) == SESSIONS_COLUMNS

    def test_ecommerce_data_columns(self):
        from superstore import ecommerce_data

        data = ecommerce_data(config={"sessions": 50, "seed": 42})

        assert list(data["products"].columns) == PRODUCTS_COLUMNS
        assert list(data["sessions"].columns) == SESSIONS_COLUMNS
        assert list(data["cart_events"].columns) == CART_EVENTS_COLUMNS
        assert list(data["orders"].columns) == ORDERS_COLUMNS
        assert list(data["customers"].columns) == CUSTOMERS_COLUMNS

    def test_ecommerce_data_reproducibility(self):
        from superstore import ecommerce_data

        data1 = ecommerce_data(config={"sessions": 50, "seed": 999})
        data2 = ecommerce_data(config={"sessions": 50, "seed": 999})

        assert data1["sessions"]["session_id"].tolist() == data2["sessions"]["session_id"].tolist()
        assert data1["products"]["product_id"].tolist() == data2["products"]["product_id"].tolist()


class TestEcommerceConfig:
    def test_config_defaults(self):
        from superstore import EcommerceConfig

        config = EcommerceConfig()
        assert config.sessions == 10000
        assert config.customers == 2000
        assert config.days == 30

    def test_config_nested(self):
        from superstore import CartConfig, EcommerceConfig, SessionConfig

        config = EcommerceConfig(
            sessions=5000,
            session=SessionConfig(
                bounce_rate=0.25,
                avg_pages_per_session=8.0,
            ),
            cart=CartConfig(
                abandonment_rate=0.60,
            ),
        )
        assert config.session.bounce_rate == 0.25
        assert config.session.avg_pages_per_session == 8.0
        assert config.cart.abandonment_rate == 0.60

    def test_config_validation(self):
        from pydantic import ValidationError

        from superstore import SessionConfig

        with pytest.raises(ValidationError):
            SessionConfig(bounce_rate=1.5)  # Must be <= 1.0

        with pytest.raises(ValidationError):
            SessionConfig(cart_add_probability=-0.1)  # Must be >= 0.0

    def test_config_factory(self):
        from superstore import ecommerce_config

        config = ecommerce_config(sessions=1000, seed=42)
        assert config.sessions == 1000
        assert config.seed == 42
