"""Exercise the compiled wheel without optional DataFrame dependencies."""

import csv

import pytest

import superstore as ss


@pytest.mark.parametrize("generator", [ss.superstore, ss.employees, ss.weather])
def test_generators(generator):
    rows = generator(count=12, output="dict", seed=42)
    assert len(rows) == 12
    assert isinstance(rows[0], dict)
    assert rows == generator(count=12, output="dict", seed=42)
    assert rows != generator(count=12, output="dict", seed=123)


def test_ecommerce_catalog():
    columns = ss.ecommerce_products(count=12, output="dict", seed=42)
    assert len(columns["product_id"]) == 12
    assert columns == ss.ecommerce_products(count=12, output="dict", seed=42)


def test_unseeded_randomness():
    first = ss.superstore(count=12, output="dict")
    second = ss.superstore(count=12, output="dict")
    assert first != second


@pytest.mark.parametrize("generator", [ss.superstoreParallel, ss.employeesParallel])
def test_parallel_generators(generator):
    assert ss.numThreads() >= 1
    rows = generator(count=12, output="dict", seed=42)
    assert len(rows) == 12
    assert rows == generator(count=12, output="dict", seed=42)


@pytest.mark.parametrize("stream", [ss.superstoreStream, ss.employeesStream])
def test_streaming(stream):
    chunks = list(stream(12, chunk_size=5, seed=42))
    assert [len(chunk) for chunk in chunks] == [5, 5, 2]
    assert chunks == list(stream(12, chunk_size=5, seed=42))


@pytest.mark.parametrize("export", [ss.superstoreArrowIpc, ss.employeesArrowIpc])
def test_arrow_ipc(export):
    data = export(12, seed=42)
    assert isinstance(data, bytes)
    assert len(data) > 0
    assert data == export(12, seed=42)


@pytest.mark.parametrize("export", [ss.superstoreToCsv, ss.employeesToCsv])
def test_csv_export(export, tmp_path):
    path = tmp_path / "data.csv"
    assert export(str(path), 12, seed=42) == 12
    with path.open() as file:
        assert len(list(csv.DictReader(file))) == 12


@pytest.mark.parametrize("compression", ["none", "snappy", "zstd"])
def test_parquet_export(compression, tmp_path):
    path = tmp_path / "data.parquet"
    assert ss.superstoreToParquet(str(path), 12, seed=42, compression=compression) == 12
    data = path.read_bytes()
    assert data[:4] == data[-4:] == b"PAR1"


def test_distribution():
    values = ss.sampleNormal(0.0, 1.0, n=12, seed=42)
    assert len(values) == 12
    assert values == ss.sampleNormal(0.0, 1.0, n=12, seed=42)
