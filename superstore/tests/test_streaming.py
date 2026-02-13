"""Tests for streaming data generation."""


class TestStreaming:
    def test_superstore_stream_basic(self):
        from superstore import superstoreStream

        chunks = list(superstoreStream(100, chunk_size=30, seed=42))
        assert len(chunks) == 4  # 30 + 30 + 30 + 10
        assert len(chunks[0]) == 30
        assert len(chunks[1]) == 30
        assert len(chunks[2]) == 30
        assert len(chunks[3]) == 10

    def test_superstore_stream_total_count(self):
        from superstore import superstoreStream

        total = sum(len(chunk) for chunk in superstoreStream(1000, chunk_size=100))
        assert total == 1000

    def test_superstore_stream_reproducible(self):
        from superstore import superstoreStream

        chunks1 = list(superstoreStream(100, chunk_size=50, seed=42))
        chunks2 = list(superstoreStream(100, chunk_size=50, seed=42))

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            for r1, r2 in zip(c1, c2):
                assert r1["Order ID"] == r2["Order ID"]
                assert r1["City"] == r2["City"]

    def test_superstore_stream_dict_format(self):
        from superstore import superstoreStream

        for chunk in superstoreStream(10, chunk_size=5, seed=42):
            assert isinstance(chunk, list)
            assert isinstance(chunk[0], dict)
            assert "Order ID" in chunk[0]
            assert "Sales" in chunk[0]
            break

    def test_employees_stream_basic(self):
        from superstore import employeesStream

        chunks = list(employeesStream(100, chunk_size=30, seed=42))
        assert len(chunks) == 4
        total = sum(len(c) for c in chunks)
        assert total == 100

    def test_employees_stream_reproducible(self):
        from superstore import employeesStream

        chunks1 = list(employeesStream(50, chunk_size=20, seed=123))
        chunks2 = list(employeesStream(50, chunk_size=20, seed=123))

        for c1, c2 in zip(chunks1, chunks2):
            for r1, r2 in zip(c1, c2):
                assert r1["Employee ID"] == r2["Employee ID"]
                assert r1["First Name"] == r2["First Name"]

    def test_stream_iterator_protocol(self):
        from superstore import superstoreStream

        stream = superstoreStream(50, chunk_size=25, seed=42)

        # Should be iterable
        first_chunk = next(iter(stream))
        assert len(first_chunk) == 25

        second_chunk = next(iter(stream))
        assert len(second_chunk) == 25

        # Should be exhausted now
        try:
            next(iter(stream))
            assert False, "Should have raised StopIteration"
        except StopIteration:
            pass
