SUPERSTORE_COLUMNS = [
    "Row ID",
    "Order ID",
    "Order Date",
    "Ship Date",
    "Ship Mode",
    "Customer ID",
    "Segment",
    "Country",
    "City",
    "State",
    "Postal Code",
    "Region",
    "Product ID",
    "Category",
    "Sub-Category",
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
]

EMPLOYEES_COLUMNS = [
    "Row ID",
    "Employee ID",
    "First Name",
    "Surname",
    "Prefix",
    "Suffix",
    "Phone Number",
    "Email",
    "SSN",
    "Street",
    "City",
    "Postal Code",
    "Region",
    "State",
    "Country",
    "Start Date",
    "Date of Birth",
]


class TestSuperstore:
    def test_superstore(self):
        import pandas as pd

        from superstore import employees, superstore

        df = superstore()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == SUPERSTORE_COLUMNS
        assert df.shape[0] == 1000

        df = employees()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == EMPLOYEES_COLUMNS
        assert df.shape[0] == 1000

    def test_superstore_polars(self):
        import polars as pl

        from superstore import employees, superstore

        df = superstore(output="polars")
        assert isinstance(df, pl.DataFrame)
        assert df.columns == SUPERSTORE_COLUMNS
        assert df.shape[0] == 1000

        df = employees(output="polars")
        assert isinstance(df, pl.DataFrame)
        assert df.columns == EMPLOYEES_COLUMNS
        assert df.shape[0] == 1000

    def test_superstore_dict(self):
        from superstore import employees, superstore

        data = superstore(output="dict")
        assert isinstance(data, list)
        assert len(data) == 1000
        assert isinstance(data[0], dict)
        assert set(data[0].keys()) == set(SUPERSTORE_COLUMNS)

        data = employees(output="dict")
        assert isinstance(data, list)
        assert len(data) == 1000
        assert isinstance(data[0], dict)
        assert set(data[0].keys()) == set(EMPLOYEES_COLUMNS)

    def test_superstore_count(self):
        import pandas as pd

        from superstore import employees, superstore

        df = superstore(count=100)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 100

        df = employees(count=50)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 50

    def test_superstore_seed_reproducibility(self):
        """Test that same seed produces identical results."""
        from superstore import superstore

        df1 = superstore(count=100, seed=42)
        df2 = superstore(count=100, seed=42)

        # DataFrames should be identical
        assert df1.equals(df2)

        # Specific values should match
        assert df1["Order ID"].tolist() == df2["Order ID"].tolist()
        assert df1["City"].tolist() == df2["City"].tolist()
        assert df1["Sales"].tolist() == df2["Sales"].tolist()

    def test_superstore_seed_different_seeds(self):
        """Test that different seeds produce different results."""
        from superstore import superstore

        df1 = superstore(count=100, seed=42)
        df2 = superstore(count=100, seed=123)

        # DataFrames should be different
        assert not df1.equals(df2)

    def test_superstore_seed_no_seed_varies(self):
        """Test that no seed produces different results each call."""
        from superstore import superstore

        df1 = superstore(count=100)
        df2 = superstore(count=100)

        # DataFrames should be different (extremely unlikely to match)
        assert not df1["Order ID"].tolist() == df2["Order ID"].tolist()

    def test_employees_seed_reproducibility(self):
        """Test that same seed produces identical employee results."""
        from superstore import employees

        df1 = employees(count=100, seed=42)
        df2 = employees(count=100, seed=42)

        # DataFrames should be identical
        assert df1.equals(df2)

        # Specific values should match
        assert df1["Employee ID"].tolist() == df2["Employee ID"].tolist()
        assert df1["First Name"].tolist() == df2["First Name"].tolist()
        assert df1["Email"].tolist() == df2["Email"].tolist()

    def test_employees_seed_different_seeds(self):
        """Test that different seeds produce different employee results."""
        from superstore import employees

        df1 = employees(count=100, seed=42)
        df2 = employees(count=100, seed=123)

        # DataFrames should be different
        assert not df1.equals(df2)

    def test_seed_with_polars_output(self):
        """Test seed reproducibility with polars output."""
        from superstore import employees, superstore

        df1 = superstore(count=100, output="polars", seed=42)
        df2 = superstore(count=100, output="polars", seed=42)
        assert df1.equals(df2)

        df1 = employees(count=100, output="polars", seed=42)
        df2 = employees(count=100, output="polars", seed=42)
        assert df1.equals(df2)

    def test_seed_with_dict_output(self):
        """Test seed reproducibility with dict output."""
        from superstore import employees, superstore

        data1 = superstore(count=100, output="dict", seed=42)
        data2 = superstore(count=100, output="dict", seed=42)
        assert data1 == data2

        data1 = employees(count=100, output="dict", seed=42)
        data2 = employees(count=100, output="dict", seed=42)
        assert data1 == data2


class TestSuperstoreConfig:
    """Tests for SuperstoreConfig-based API."""

    def test_superstore_with_config(self):
        """Test superstore() with SuperstoreConfig."""
        import pandas as pd

        from superstore import superstore
        from superstore.config import SuperstoreConfig

        config = SuperstoreConfig(count=100, seed=42)
        df = superstore(config)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_superstore_config_output_format(self):
        """Test SuperstoreConfig respects output format."""
        import polars as pl

        from superstore import superstore
        from superstore.config import SuperstoreConfig

        config = SuperstoreConfig(count=50, output="polars", seed=42)
        df = superstore(config)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 50

    def test_superstore_config_override(self):
        """Test explicit params override config values."""
        from superstore import superstore
        from superstore.config import SuperstoreConfig

        config = SuperstoreConfig(count=100, seed=42)
        # Override count with explicit arg
        df = superstore(config, count=50)
        assert len(df) == 50

    def test_superstore_config_seed_reproducibility(self):
        """Test seed in config produces reproducible results."""
        from superstore import superstore
        from superstore.config import SuperstoreConfig

        config = SuperstoreConfig(count=100, seed=42)
        df1 = superstore(config)
        df2 = superstore(config)
        assert df1.equals(df2)

    def test_superstore_config_dict_output(self):
        """Test SuperstoreConfig with dict output."""
        from superstore import superstore
        from superstore.config import SuperstoreConfig

        config = SuperstoreConfig(count=25, output="dict", seed=42)
        data = superstore(config)
        assert isinstance(data, list)
        assert len(data) == 25
        assert all(isinstance(row, dict) for row in data)

    def test_superstore_config_pool_size(self):
        """Test SuperstoreConfig with custom pool_size."""
        from superstore import superstore
        from superstore.config import SuperstoreConfig

        # Test with default pool_size
        config_default = SuperstoreConfig(count=100, seed=42)
        assert config_default.pool_size == 1000

        # Test with custom pool_size - smaller
        config_small = SuperstoreConfig(count=100, seed=42, pool_size=50)
        df_small = superstore(config_small)
        assert len(df_small) == 100

        # Test with custom pool_size - larger
        config_large = SuperstoreConfig(count=500, seed=42, pool_size=5000)
        df_large = superstore(config_large)
        assert len(df_large) == 500

        # Verify pool_size affects variety of generated data
        # With small pool (50), there should be more duplicate cities
        config_tiny = SuperstoreConfig(count=200, seed=42, pool_size=10)
        df_tiny = superstore(config_tiny)
        unique_cities_tiny = df_tiny["City"].nunique()

        config_big = SuperstoreConfig(count=200, seed=42, pool_size=1000)
        df_big = superstore(config_big)
        unique_cities_big = df_big["City"].nunique()

        # With 10 pool vs 1000 pool, the tiny pool should have fewer unique cities
        # (limited by pool size of 10)
        assert unique_cities_tiny <= 10
        assert unique_cities_big > unique_cities_tiny
