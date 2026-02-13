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
