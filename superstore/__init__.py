__version__ = "0.2.6"

# Import directly from the native Rust module
from .superstore import (
    JOBS_SCHEMA,
    MACHINE_SCHEMA,
    STATUS_SCHEMA,
    USAGE_SCHEMA,
    employees,
    getTimeSeries,
    getTimeSeriesData,
    jobs,
    machines,
    status,
    superstore,
    usage,
)

__all__ = (
    "JOBS_SCHEMA",
    "MACHINE_SCHEMA",
    "STATUS_SCHEMA",
    "USAGE_SCHEMA",
    "employees",
    "getTimeSeries",
    "getTimeSeriesData",
    "jobs",
    "machines",
    "status",
    "superstore",
    "usage",
)
