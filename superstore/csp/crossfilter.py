from datetime import datetime, timedelta
from typing import List

import csp
from csp import ts

from .. import (
    JOBS_SCHEMA,
    MACHINE_SCHEMA,
    STATUS_SCHEMA,
    USAGE_SCHEMA,
    jobs as _jobs,
    machines as _machines,
    status as _status,
    usage as _usage,
)

__all__ = (
    "machines",
    "Machine",
    "MACHINE_SCHEMA",
    "usage",
    "Usage",
    "USAGE_SCHEMA",
    "status",
    "Status",
    "STATUS_SCHEMA",
    "jobs",
    "Job",
    "JOBS_SCHEMA",
)


class Machine(csp.Struct):
    machine_id: str
    kind: str
    cores: int
    region: str
    zone: str


class Usage(Machine):
    cpu: float
    mem: float
    free: float
    network: float
    disk: float


class Status(Usage):
    status: str
    last_update: datetime


class Job(csp.Struct):
    machine_id: str
    job_id: str
    name: str
    units: int
    start_time: datetime
    end_time: datetime


@csp.graph
def machines() -> ts[List[Machine]]:
    return csp.const([Machine(**d) for d in _machines()])


@csp.node
def usage(machines: ts[List[Machine]], interval: timedelta = timedelta(seconds=5)) -> ts[List[Usage]]:
    with csp.alarms():
        a_tick = csp.alarm(bool)

    with csp.state():
        s_machines = []
        s_machine_usage = {}

    with csp.start():
        csp.schedule_alarm(a_tick, timedelta(), True)

    if csp.ticked(machines):
        s_machines = machines.copy()

    if csp.ticked(a_tick):
        for machine in s_machines:
            machine_id = machine.machine_id
            if machine_id not in s_machine_usage:
                s_machine_usage[machine_id] = Usage(**machine.to_dict())
            s_machine_usage[machine_id] = Usage(**_usage(s_machine_usage[machine_id].to_dict()))
        csp.output(s_machine_usage.values())
        csp.schedule_alarm(a_tick, interval, True)


@csp.node
def status(usage: ts[List[Usage]], interval: timedelta = timedelta(seconds=5)) -> ts[List[Status]]:
    with csp.alarms():
        a_tick = csp.alarm(bool)

    with csp.state():
        csp.make_passive(usage)

    with csp.start():
        csp.schedule_alarm(a_tick, timedelta(), True)

    if csp.ticked(a_tick) and csp.valid(usage):
        ret = []
        for machine in usage:
            ret.append(Status(**_status(machine.to_dict())))
        csp.output(ret)
        csp.schedule_alarm(a_tick, interval, True)


@csp.node
def jobs(machines: ts[List[Machine]], interval: timedelta = timedelta(seconds=5)) -> ts[List[Job]]:
    with csp.alarms():
        a_tick = csp.alarm(bool)

    with csp.start():
        csp.schedule_alarm(a_tick, timedelta(), True)
        csp.make_passive(machines)

    if csp.ticked(a_tick):
        ret = []
        for machine in machines:
            job = _jobs(machine.to_dict())
            if job:
                ret.append(Job(**job))
        csp.output(ret)
        csp.schedule_alarm(a_tick, interval, True)
