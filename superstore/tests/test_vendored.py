class TestVendored:
    def test_timeseries(self):
        from superstore import timeseries

        df = timeseries()
        assert len(df) == 30
        assert len(df.columns) == 4
        assert df.columns.tolist() == ["A", "B", "C", "D"]

    def test_timeseries_ncol_nper(self):
        from superstore import timeseries

        df = timeseries(nper=100, ncol=10)
        assert len(df) == 100
        assert len(df.columns) == 10
        assert df.columns.tolist() == list("ABCDEFGHIJ")

    def test_timeseries_polars(self):
        import polars as pl

        from superstore import timeseries

        df = timeseries(output="polars")
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 30
        # Polars includes the index column
        assert "index" in df.columns
        assert "A" in df.columns

    def test_timeseries_dict(self):
        from superstore import timeseries

        data = timeseries(output="dict")
        assert isinstance(data, dict)
        assert "index" in data
        assert "A" in data
        assert len(data["index"]) == 30

    def test_timeseriesdata(self):
        from superstore import timeseriesData

        data = timeseriesData()
        assert isinstance(data, dict)
        assert len(data) == 4
        # Each entry is a pandas Series
        assert "A" in data

    def test_timeseriesdata_polars(self):
        import polars as pl

        from superstore import timeseriesData

        data = timeseriesData(output="polars")
        assert isinstance(data, dict)
        assert "A" in data
        # Each entry is a polars DataFrame with index and value columns
        assert isinstance(data["A"], pl.DataFrame)

    def test_timeseriesdata_dict(self):
        from superstore import timeseriesData

        data = timeseriesData(output="dict")
        assert isinstance(data, dict)
        assert "A" in data
        # Each entry is a dict with 'index' and 'values' keys
        assert isinstance(data["A"], dict)
        assert "index" in data["A"]
        assert "values" in data["A"]

    def test_timeseries_seed_reproducibility(self):
        """Test that same seed produces identical time series."""
        from superstore import timeseries

        df1 = timeseries(nper=50, ncol=3, seed=42)
        df2 = timeseries(nper=50, ncol=3, seed=42)

        # DataFrames should be identical
        assert df1.equals(df2)

        # Specific values should match
        assert df1["A"].tolist() == df2["A"].tolist()
        assert df1["B"].tolist() == df2["B"].tolist()

    def test_timeseries_seed_different_seeds(self):
        """Test that different seeds produce different time series."""
        from superstore import timeseries

        df1 = timeseries(nper=50, ncol=3, seed=42)
        df2 = timeseries(nper=50, ncol=3, seed=123)

        # DataFrames should be different
        assert not df1.equals(df2)

    def test_timeseries_seed_no_seed_varies(self):
        """Test that no seed produces different results each call."""
        from superstore import timeseries

        df1 = timeseries(nper=50, ncol=3)
        df2 = timeseries(nper=50, ncol=3)

        # DataFrames should be different (extremely unlikely to match)
        assert not df1["A"].tolist() == df2["A"].tolist()

    def test_timeseriesdata_seed_reproducibility(self):
        """Test that same seed produces identical time series data."""
        from superstore import timeseriesData

        data1 = timeseriesData(nper=50, ncol=3, seed=42)
        data2 = timeseriesData(nper=50, ncol=3, seed=42)

        # Data should be identical - compare Series values
        assert data1["A"].tolist() == data2["A"].tolist()
        assert data1["B"].tolist() == data2["B"].tolist()

    def test_timeseriesdata_seed_different_seeds(self):
        """Test that different seeds produce different time series data."""
        from superstore import timeseriesData

        data1 = timeseriesData(nper=50, ncol=3, seed=42)
        data2 = timeseriesData(nper=50, ncol=3, seed=123)

        # Data should be different
        assert not data1["A"].tolist() == data2["A"].tolist()

    def test_timeseries_seed_with_polars(self):
        """Test seed reproducibility with polars output."""
        from superstore import timeseries

        df1 = timeseries(nper=50, ncol=3, output="polars", seed=42)
        df2 = timeseries(nper=50, ncol=3, output="polars", seed=42)
        assert df1.equals(df2)

    def test_timeseries_seed_with_dict(self):
        """Test seed reproducibility with dict output."""
        from superstore import timeseries

        data1 = timeseries(nper=50, ncol=3, output="dict", seed=42)
        data2 = timeseries(nper=50, ncol=3, output="dict", seed=42)
        assert data1 == data2
