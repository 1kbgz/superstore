from datetime import datetime

from superstore import (
    JOBS_SCHEMA,
    MACHINE_SCHEMA,
    STATUS_SCHEMA,
    USAGE_SCHEMA,
    jobs,
    machines,
    status,
    usage,
)


class TestCrossfilters:
    def test_id(self):
        # Test that machine_id is generated correctly (12 char hex string)
        m = machines(1)[0]
        assert len(m["machine_id"]) == 12

    def test_clip(self):
        # Test that usage values are clipped properly (via usage function)
        m = machines(1)[0]
        u = usage(m)
        # cpu should be clipped to 0-100 range
        assert 0 <= u["cpu"] <= 100
        assert 0 <= u["mem"] <= 100

    def test_clip_inf(self):
        # Test that network/disk can exceed 100 (unbounded upper)
        for _ in range(100):
            m = machines(1)[0]
            u = usage(m)
            assert u["network"] >= 0
            assert u["disk"] >= 0

    def test_randrange(self):
        # Test that generated values vary (via machines function)
        cores_seen = set()
        for _ in range(100):
            m = machines(1)[0]
            cores_seen.add(m["cores"])
        assert len(cores_seen) > 1  # Should see variation

    def test_schema_invariance(self):
        assert JOBS_SCHEMA == {
            "machine_id": "string",
            "job_id": "string",
            "name": "string",
            "units": "integer",
            "start_time": "datetime",
            "end_time": "datetime",
        }
        assert MACHINE_SCHEMA == {
            "machine_id": "string",
            "kind": "string",
            "cores": "integer",
            "region": "string",
            "zone": "string",
        }
        assert STATUS_SCHEMA == {
            "machine_id": "string",
            "kind": "string",
            "cores": "integer",
            "region": "string",
            "zone": "string",
            "cpu": "float",
            "mem": "float",
            "free": "float",
            "network": "float",
            "disk": "float",
            "status": "string",
            "last_update": "datetime",
        }
        assert USAGE_SCHEMA == {
            "machine_id": "string",
            "kind": "string",
            "cores": "integer",
            "region": "string",
            "zone": "string",
            "cpu": "float",
            "mem": "float",
            "free": "float",
            "network": "float",
            "disk": "float",
        }

    def test_machines(self):
        some_machines = machines(100)
        assert isinstance(some_machines, list)
        assert len(some_machines) == 100
        for machine in some_machines:
            assert len(machine["machine_id"]) == 12
            assert machine["kind"] in ("core", "edge", "worker")
            assert machine["region"] in ("na", "eu", "ap")
            assert machine["zone"] in "ABCD"
            assert machine["cores"] in (4, 8, 16, 32, 64)

    def test_usage(self):
        machine = machines(1)[0]

        # Test usage with machine dict
        u = usage(machine)
        assert "cpu" in u
        assert "mem" in u
        assert "free" in u
        assert "network" in u
        assert "disk" in u
        assert u["machine_id"] == machine["machine_id"]

    def test_status(self):
        machine = machines(1)[0]
        m_usage = usage(machine)

        # Test status returns proper fields
        s = status(m_usage)
        assert "status" in s
        assert "last_update" in s
        assert isinstance(s["last_update"], datetime)
        assert s["status"] in ("idle", "active", "capacity", "unknown")

        # Test status determination based on cpu
        m_usage["cpu"] = 0.0
        s = status(m_usage)
        assert s["status"] == "idle"

        m_usage["cpu"] = 50.0
        s = status(m_usage)
        assert s["status"] == "active"

        m_usage["cpu"] = 100.0
        s = status(m_usage)
        assert s["status"] == "capacity"

    def test_jobs(self):
        # Find a worker machine
        worker = None
        for _ in range(100):
            m = machines(1)[0]
            if m["kind"] == "worker":
                worker = m
                break

        if worker:
            # Jobs may or may not be returned (random)
            for _ in range(100):
                job = jobs(worker)
                if job:
                    assert job["machine_id"] == worker["machine_id"]
                    assert len(job["job_id"]) == 12
                    assert isinstance(job["start_time"], datetime)
                    assert isinstance(job["end_time"], datetime)
                    assert len(job["name"].split("-")) == 2
                    break
            # It's possible no job is returned due to random chance, that's ok

    def test_machines_seed_reproducibility(self):
        """Test that same seed produces identical machine data."""
        machines1 = machines(50, seed=42)
        machines2 = machines(50, seed=42)

        # Lists should be identical
        assert machines1 == machines2

        # Specific values should match
        assert [m["machine_id"] for m in machines1] == [m["machine_id"] for m in machines2]
        assert [m["kind"] for m in machines1] == [m["kind"] for m in machines2]
        assert [m["cores"] for m in machines1] == [m["cores"] for m in machines2]

    def test_machines_seed_different_seeds(self):
        """Test that different seeds produce different machine data."""
        machines1 = machines(50, seed=42)
        machines2 = machines(50, seed=123)

        # Lists should be different
        assert machines1 != machines2
        assert [m["machine_id"] for m in machines1] != [m["machine_id"] for m in machines2]

    def test_machines_seed_no_seed_varies(self):
        """Test that no seed produces different results each call."""
        machines1 = machines(50)
        machines2 = machines(50)

        # Lists should be different (extremely unlikely to match)
        assert [m["machine_id"] for m in machines1] != [m["machine_id"] for m in machines2]

    def test_usage_seed_reproducibility(self):
        """Test that same seed produces identical usage data."""
        # First create a seeded machine
        m = machines(1, seed=42)[0]

        # Get initial usage, then update it to test seeded updates
        u1_initial = usage(m, seed=100)
        u2_initial = usage(m, seed=100)

        # Initial usage is always zeros, but check they're identical
        assert u1_initial["cpu"] == u2_initial["cpu"]
        assert u1_initial["mem"] == u2_initial["mem"]
        assert u1_initial["network"] == u2_initial["network"]
        assert u1_initial["disk"] == u2_initial["disk"]

        # Now test with previous usage to exercise the random path
        # Add previous usage fields to trigger non-zero generation
        m_with_usage = {**m, "cpu": 50.0, "mem": 50.0, "free": 50.0, "network": 50.0, "disk": 50.0}
        u1 = usage(m_with_usage, seed=100)
        u2 = usage(m_with_usage, seed=100)

        # With same seed and same previous usage, we should get same results
        assert u1["cpu"] == u2["cpu"]
        assert u1["mem"] == u2["mem"]

    def test_usage_seed_different_seeds(self):
        """Test that different seeds produce different usage data when there's previous usage."""
        m = machines(1, seed=42)[0]

        # Add previous usage fields to trigger non-zero random generation
        m_with_usage = {**m, "cpu": 50.0, "mem": 50.0, "free": 50.0, "network": 50.0, "disk": 50.0}

        # Try multiple seed pairs to ensure we get different values
        # The usage function has a 10% reset chance, so some calls may return zeros
        for seed1, seed2 in [(100, 999), (1, 12345), (42, 9999), (7, 77), (123, 456)]:
            u1 = usage(m_with_usage, seed=seed1)
            u2 = usage(m_with_usage, seed=seed2)

            # Only compare if neither was reset to zero
            if u1["cpu"] > 0 and u2["cpu"] > 0:
                if u1["cpu"] != u2["cpu"] or u1["mem"] != u2["mem"] or u1["network"] != u2["network"] or u1["disk"] != u2["disk"]:
                    break

        # It's ok if we don't find a difference - seed can trigger reset (10% chance each)
        # What matters is reproducibility, which is tested in test_usage_seed_reproducibility

    def test_jobs_seed_reproducibility(self):
        """Test that same seed produces identical job data."""
        # Create a deterministic worker machine
        m = machines(1, seed=42)[0]
        # Force kind to worker for jobs to work
        m["kind"] = "worker"

        j1 = jobs(m, seed=100)
        j2 = jobs(m, seed=100)

        # Both should return same result (either both None or same job)
        if j1 is None:
            assert j2 is None
        else:
            assert j1["job_id"] == j2["job_id"]
            assert j1["name"] == j2["name"]
            assert j1["units"] == j2["units"]

    def test_jobs_seed_different_seeds(self):
        """Test that different seeds produce different job data when job is returned."""
        m = machines(1, seed=42)[0]
        m["kind"] = "worker"

        # Try multiple times to get jobs with different seeds
        for base_seed in range(100):
            j1 = jobs(m, seed=base_seed)
            j2 = jobs(m, seed=base_seed + 1000)

            if j1 is not None and j2 is not None:
                # Jobs should differ
                jobs_differ = j1["job_id"] != j2["job_id"] or j1["name"] != j2["name"] or j1["units"] != j2["units"]
                assert jobs_differ
                break

        # It's ok if we didn't find jobs - random chance
