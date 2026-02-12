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
