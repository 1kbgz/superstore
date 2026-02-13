"""Benchmarks for data export functions."""

import os
import tempfile

import superstore as ss


def _set_deterministic():
    """Set deterministic mode if available."""
    if hasattr(ss, "setDeterministicMode"):
        ss.setDeterministicMode(1)


class ParquetExportSuite:
    """Benchmark suite for Parquet export."""

    params = [10_000, 100_000]
    param_names = ["n_rows"]
    timeout = 120

    def setup(self, n_rows):
        _set_deterministic()
        self.tmpdir = tempfile.mkdtemp()
        self.filepath = os.path.join(self.tmpdir, "test.parquet")

    def teardown(self, n_rows):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        os.rmdir(self.tmpdir)

    def time_export_parquet(self, n_rows):
        """Time Parquet export."""
        ss.superstoreToParquet(self.filepath, n_rows)


class CSVExportSuite:
    """Benchmark suite for CSV export."""

    params = [10_000, 100_000]
    param_names = ["n_rows"]
    timeout = 120

    def setup(self, n_rows):
        _set_deterministic()
        self.tmpdir = tempfile.mkdtemp()
        self.filepath = os.path.join(self.tmpdir, "test.csv")

    def teardown(self, n_rows):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        os.rmdir(self.tmpdir)

    def time_export_csv(self, n_rows):
        """Time CSV export."""
        ss.superstoreToCsv(self.filepath, n_rows)


class ArrowExportSuite:
    """Benchmark suite for Arrow/IPC export (returns bytes)."""

    params = [10_000, 100_000]
    param_names = ["n_rows"]
    timeout = 120

    def setup(self, n_rows):
        _set_deterministic()

    def time_export_arrow(self, n_rows):
        """Time Arrow/IPC byte generation."""
        ss.superstoreArrowIpc(n_rows)
