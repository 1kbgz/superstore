"""Tests for parallel data generation."""

import pandas as pd


class TestParallel:
    def test_superstore_parallel_basic(self):
        from superstore import superstoreParallel

        df = superstoreParallel(100, seed=42)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_superstore_parallel_reproducible(self):
        from superstore import superstoreParallel

        df1 = superstoreParallel(100, seed=42)
        df2 = superstoreParallel(100, seed=42)

        # With same seed and same thread count, should be identical
        assert df1.equals(df2)

    def test_superstore_parallel_different_seeds(self):
        from superstore import superstoreParallel

        df1 = superstoreParallel(100, seed=42)
        df2 = superstoreParallel(100, seed=123)

        # Different seeds should produce different data
        assert not df1.equals(df2)

    def test_superstore_parallel_polars(self):
        import polars as pl

        from superstore import superstoreParallel

        df = superstoreParallel(100, output="polars", seed=42)
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 100

    def test_superstore_parallel_dict(self):
        from superstore import superstoreParallel

        data = superstoreParallel(100, output="dict", seed=42)
        assert isinstance(data, list)
        assert len(data) == 100
        assert isinstance(data[0], dict)

    def test_employees_parallel_basic(self):
        from superstore import employeesParallel

        df = employeesParallel(100, seed=42)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_employees_parallel_reproducible(self):
        from superstore import employeesParallel

        df1 = employeesParallel(100, seed=42)
        df2 = employeesParallel(100, seed=42)

        assert df1.equals(df2)

    def test_num_threads(self):
        from superstore import numThreads

        threads = numThreads()
        assert threads >= 1

    def test_parallel_large_count(self):
        from superstore import superstoreParallel

        # Test with a larger count to ensure parallelization works
        df = superstoreParallel(10000, seed=42)
        assert len(df) == 10000
