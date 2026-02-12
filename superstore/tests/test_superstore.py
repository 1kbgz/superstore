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
