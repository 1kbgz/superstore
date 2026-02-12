class TestVendored:
    def test_gettimeseries(self):
        from superstore import getTimeSeries

        df = getTimeSeries()
        assert len(df) == 30
        assert len(df.columns) == 4
        assert df.columns.tolist() == ["A", "B", "C", "D"]

    def test_gettimeseries_ncol_nper(self):
        from superstore import getTimeSeries

        df = getTimeSeries(nper=100, ncol=10)
        assert len(df) == 100
        assert len(df.columns) == 10
        assert df.columns.tolist() == list("ABCDEFGHIJ")

    def test_gettimeseries_polars(self):
        import polars as pl

        from superstore import getTimeSeries

        df = getTimeSeries(output="polars")
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 30
        # Polars includes the index column
        assert "index" in df.columns
        assert "A" in df.columns

    def test_gettimeseries_dict(self):
        from superstore import getTimeSeries

        data = getTimeSeries(output="dict")
        assert isinstance(data, dict)
        assert "index" in data
        assert "A" in data
        assert len(data["index"]) == 30

    def test_gettimeseriesdata(self):
        from superstore import getTimeSeriesData

        data = getTimeSeriesData()
        assert isinstance(data, dict)
        assert len(data) == 4
        # Each entry is a pandas Series
        assert "A" in data

    def test_gettimeseriesdata_polars(self):
        import polars as pl

        from superstore import getTimeSeriesData

        data = getTimeSeriesData(output="polars")
        assert isinstance(data, dict)
        assert "A" in data
        # Each entry is a polars DataFrame with index and value columns
        assert isinstance(data["A"], pl.DataFrame)

    def test_gettimeseriesdata_dict(self):
        from superstore import getTimeSeriesData

        data = getTimeSeriesData(output="dict")
        assert isinstance(data, dict)
        assert "A" in data
        # Each entry is a dict with 'index' and 'values' keys
        assert isinstance(data["A"], dict)
        assert "index" in data["A"]
        assert "values" in data["A"]
