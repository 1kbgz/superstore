from datetime import datetime, timedelta

import pytest

try:
    import csp
except ImportError:
    csp = None


class TestCrossfilter:
    @pytest.mark.skipif(csp is None, reason="csp required for csp tests")
    def test_machines(self):
        from superstore.csp import machines

        start = datetime.today()  # noqa: DTZ002
        out = csp.run(machines, realtime=False, starttime=start, endtime=timedelta(seconds=1))
        assert len(out) == 1
        assert len(out[0]) == 1
        assert len(out[0][0]) == 2
        assert out[0][0][0] == start
        assert len(out[0][0][1]) == 100

    @pytest.mark.skipif(csp is None, reason="csp required for csp tests")
    def test_usage(self):
        from superstore.csp import machines, usage

        start = datetime.today()  # noqa: DTZ002

        @csp.graph
        def _graph():
            ms = machines()
            csp.add_graph_output("usage", usage(ms))

        out = csp.run(_graph, realtime=False, starttime=start, endtime=timedelta(seconds=1))
        assert len(out) == 1
        assert len(out["usage"]) == 1
        assert len(out["usage"][0]) == 2
        assert out["usage"][0][0] == start
        assert len(out["usage"][0][1]) == 100

    @pytest.mark.skipif(csp is None, reason="csp required for csp tests")
    def test_status(self):
        from superstore.csp import machines, status, usage

        start = datetime.today()  # noqa: DTZ002

        @csp.graph
        def _graph():
            ms = machines()
            us = usage(ms)
            csp.add_graph_output("status", status(us))

        out = csp.run(_graph, realtime=False, starttime=start, endtime=timedelta(seconds=1))
        assert len(out) == 1
        assert len(out["status"]) == 1
        assert len(out["status"][0]) == 2
        assert out["status"][0][0] == start
        assert len(out["status"][0][1]) == 100

    @pytest.mark.skipif(csp is None, reason="csp required for csp tests")
    def test_jobs(self):
        from superstore.csp import jobs, machines

        start = datetime.today()  # noqa: DTZ002

        @csp.graph
        def _graph():
            ms = machines()
            csp.add_graph_output("jobs", jobs(ms))

        out = csp.run(_graph, realtime=False, starttime=start, endtime=timedelta(seconds=1))
        assert len(out) == 1
        assert len(out["jobs"]) == 1
        assert len(out["jobs"][0]) == 2
        assert out["jobs"][0][0] == start
        assert 1 < len(out["jobs"][0][1]) < 30
