"""Test new features: Arrow IPC, Parquet/CSV export, and correlation matrices."""

import os
import tempfile

import pytest


class TestArrowIpc:
    """Test Arrow IPC export functions."""

    def test_superstore_arrow_ipc_basic(self):
        from superstore import superstoreArrowIpc

        ipc_bytes = superstoreArrowIpc(100, seed=42)
        assert isinstance(ipc_bytes, bytes)
        assert len(ipc_bytes) > 0

    def test_employees_arrow_ipc_basic(self):
        from superstore import employeesArrowIpc

        ipc_bytes = employeesArrowIpc(100, seed=42)
        assert isinstance(ipc_bytes, bytes)
        assert len(ipc_bytes) > 0

    def test_arrow_ipc_reproducibility(self):
        from superstore import superstoreArrowIpc

        ipc1 = superstoreArrowIpc(50, seed=12345)
        ipc2 = superstoreArrowIpc(50, seed=12345)
        assert ipc1 == ipc2

    def test_arrow_ipc_with_pyarrow(self):
        """Test that Arrow IPC bytes can be read by PyArrow."""
        pytest.importorskip("pyarrow")
        import pyarrow as pa

        from superstore import superstoreArrowIpc

        ipc_bytes = superstoreArrowIpc(100, seed=42)
        reader = pa.ipc.open_stream(ipc_bytes)
        table = reader.read_all()
        assert table.num_rows == 100
        assert "row_id" in table.column_names


class TestFileExport:
    """Test Parquet and CSV export functions."""

    def test_superstore_to_parquet(self):
        from superstore import superstoreToParquet

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            count = superstoreToParquet(path, 100, seed=42)
            assert count == 100
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_employees_to_parquet(self):
        from superstore import employeesToParquet

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            count = employeesToParquet(path, 50, seed=42)
            assert count == 50
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_parquet_compression_options(self):
        from superstore import superstoreToParquet

        paths = {}
        try:
            for compression in ["none", "snappy", "zstd"]:
                with tempfile.NamedTemporaryFile(suffix=f"_{compression}.parquet", delete=False) as f:
                    path = f.name
                paths[compression] = path
                superstoreToParquet(path, 100, seed=42, compression=compression)

            # All files should exist
            for path in paths.values():
                assert os.path.exists(path)
        finally:
            for path in paths.values():
                os.unlink(path)

    def test_superstore_to_csv(self):
        from superstore import superstoreToCsv

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            count = superstoreToCsv(path, 100, seed=42)
            assert count == 100
            assert os.path.exists(path)

            # Verify CSV content
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 101  # header + 100 rows
            assert "row_id" in lines[0]
        finally:
            os.unlink(path)

    def test_employees_to_csv(self):
        from superstore import employeesToCsv

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            count = employeesToCsv(path, 50, seed=42)
            assert count == 50
        finally:
            os.unlink(path)

    def test_parquet_readable_by_pyarrow(self):
        """Test that Parquet files can be read by PyArrow."""
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        from superstore import superstoreToParquet

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            superstoreToParquet(path, 100, seed=42)
            table = pq.read_table(path)
            assert table.num_rows == 100
        finally:
            os.unlink(path)


class TestCorrelationMatrix:
    """Test correlation matrix support."""

    def test_identity_matrix(self):
        from superstore import CorrelationMatrix

        corr = CorrelationMatrix.identity(3)
        assert corr.dim == 3

    def test_uniform_matrix(self):
        from superstore import CorrelationMatrix

        corr = CorrelationMatrix.uniform(4, 0.5)
        assert corr.dim == 4

    def test_ar1_matrix(self):
        from superstore import CorrelationMatrix

        corr = CorrelationMatrix.ar1(5, 0.9)
        assert corr.dim == 5

    def test_custom_matrix(self):
        from superstore import CorrelationMatrix

        # 2x2 correlation matrix
        corr = CorrelationMatrix([1.0, 0.8, 0.8, 1.0])
        assert corr.dim == 2

    def test_named_variables(self):
        from superstore import CorrelationMatrix

        names = ["sales", "profit", "quantity"]
        corr = CorrelationMatrix.identity(3, names=names)
        assert corr.names == names

    def test_sample(self):
        from superstore import CorrelationMatrix

        corr = CorrelationMatrix.uniform(3, 0.7)
        samples = corr.sample(100, [0.0, 10.0, 20.0], [1.0, 2.0, 3.0], seed=42)

        assert len(samples) == 100
        assert len(samples[0]) == 3

    def test_sample_columns(self):
        from superstore import CorrelationMatrix

        corr = CorrelationMatrix.identity(2)
        columns = corr.sample_columns(100, [0.0, 10.0], [1.0, 2.0], seed=42)

        assert len(columns) == 2
        assert len(columns[0]) == 100
        assert len(columns[1]) == 100

    def test_sample_reproducibility(self):
        from superstore import CorrelationMatrix

        corr = CorrelationMatrix.uniform(3, 0.7)
        s1 = corr.sample(50, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], seed=12345)
        s2 = corr.sample(50, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], seed=12345)

        for r1, r2 in zip(s1, s2):
            assert r1 == r2

    def test_invalid_correlation(self):
        from superstore import CorrelationMatrix

        # Correlation values must be between -1 and 1
        with pytest.raises(ValueError):
            CorrelationMatrix([1.0, 1.5, 1.5, 1.0])

    def test_bivariate_sample(self):
        from superstore import pearsonCorrelation, sampleBivariate

        x, y = sampleBivariate(10000, 0.8, seed=42)

        assert len(x) == 10000
        assert len(y) == 10000

        r = pearsonCorrelation(x, y)
        assert abs(r - 0.8) < 0.05  # Allow 5% tolerance

    def test_pearson_correlation(self):
        from superstore import pearsonCorrelation

        # Perfect positive correlation
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        r = pearsonCorrelation(x, y)
        assert abs(r - 1.0) < 1e-10

        # Perfect negative correlation
        y_neg = [10.0, 8.0, 6.0, 4.0, 2.0]
        r_neg = pearsonCorrelation(x, y_neg)
        assert abs(r_neg + 1.0) < 1e-10
