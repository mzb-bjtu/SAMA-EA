"""Problem-specific comparator reproduction."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

CODE_DIR = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class B23Parameters:

    population_size: int = 250
    mutation_rate: float = 0.20
    gene_mutation_probability: float = 0.02
    elite_fraction: float = 0.10
    sfp: float = 1.00
    svp: float = 0.25
    sbf: float = 0.50
    x1: float = 0.00
    x2: float = 0.20
    y1: float = 0.60
    y2: float = 1.00
    z1: float = 0.00
    z2: float = 0.60
    reduction_mode: str = "linear"
    schedule_generation_budget: int = 100_000

PAPER_PARAMETERS = B23Parameters()

@dataclass
class Problem:
    n: int
    dur: List[int]
    dem: List[List[int]]
    pred: List[List[int]]
    succ: List[List[int]]
    cap: List[int]
    project_id: List[int]
    local_activity_no: List[int]
    release: List[int]
    is_dummy: List[bool]
    project_of: List[List[int]]
    project_start: List[int]
    project_end: List[int]
    n_projects: int
    n_non_dummy: int
    horizon: int
    mcr: int

    @staticmethod
    def from_rcmp_json(data: dict) -> "Problem":

        projects = data["projects"]
        n_projects = len(projects)
        cap = [int(v) for v in data["resource_avail"]]
        n_resources = int(data["num_resources"])
        if len(cap) != n_resources:
            raise ValueError("resource_avail length does not match num_resources")

        activity_rows: List[Tuple[int, int, dict]] = []
        index_by_id: Dict[str, int] = {}
        project_release = [0] * n_projects
        project_of: List[List[int]] = [[] for _ in range(n_projects)]
        for pj, project in enumerate(projects):
            project_release[pj] = int(project.get("release_date", 0))
            for local_pos, activity in enumerate(project["activities"], start=1):
                idx = len(activity_rows)
                aid = str(activity["activity_id"])
                if aid in index_by_id:
                    raise ValueError(f"duplicate activity id: {aid}")
                index_by_id[aid] = idx
                try:
                    local_no = int(aid.rsplit(".", 1)[1])
                except (IndexError, ValueError):
                    local_no = local_pos
                activity_rows.append((pj, local_no, activity))
                project_of[pj].append(idx)

        n = len(activity_rows)
        dur = [0] * n
        dem = [[0] * n_resources for _ in range(n)]
        pred: List[List[int]] = [[] for _ in range(n)]
        succ: List[List[int]] = [[] for _ in range(n)]
        project_id = [0] * n
        local_activity_no = [0] * n
        release = [0] * n
        for idx, (pj, local_no, activity) in enumerate(activity_rows):
            project_id[idx] = pj
            local_activity_no[idx] = local_no
            release[idx] = project_release[pj]
            dur[idx] = int(activity["duration"])
            req = [int(v) for v in activity["resource_demands"]]
            if len(req) != n_resources:
                raise ValueError(f"activity {activity['activity_id']} has an invalid demand vector")
            if any(req[k] > cap[k] for k in range(n_resources)):
                raise ValueError(f"activity {activity['activity_id']} exceeds resource capacity")
            dem[idx] = req

        for idx, (_, _, activity) in enumerate(activity_rows):
            for successor in activity.get("successors", []):
                sid = f"{successor[0]}.{successor[1]}"
                if sid not in index_by_id:
                    raise ValueError(f"unknown successor id {sid}")
                target = index_by_id[sid]
                succ[idx].append(target)
                pred[target].append(idx)
        for idx in range(n):
            succ[idx] = sorted(set(succ[idx]))
            pred[idx] = sorted(set(pred[idx]))

        project_start: List[int] = []
        project_end: List[int] = []
        is_dummy = [False] * n
        for pj, activities in enumerate(project_of):
            starts = [a for a in activities if not any(project_id[p] == pj for p in pred[a])]
            ends = [a for a in activities if not any(project_id[s] == pj for s in succ[a])]
            if len(starts) != 1 or len(ends) != 1:
                raise ValueError(f"project {pj + 1} must have one dummy start and one dummy end")
            project_start.append(starts[0])
            project_end.append(ends[0])
            is_dummy[starts[0]] = True
            is_dummy[ends[0]] = True

        n_non_dummy = sum(not flag for flag in is_dummy)
        horizon = max(project_release, default=0) + sum(dur) + 1
        work_by_resource = [
            sum(dur[i] * dem[i][k] for i in range(n)) for k in range(n_resources)
        ]
        mcr = max(range(n_resources), key=lambda k: (work_by_resource[k] / cap[k], -k))
        problem = Problem(
            n, dur, dem, pred, succ, cap, project_id, local_activity_no,
            release, is_dummy, project_of, project_start, project_end,
            n_projects, n_non_dummy, horizon, mcr,
        )
        if len(problem.topological_order()) != n:
            raise ValueError("precedence graph contains a cycle")
        return problem

    def tie_key(self, activity: int) -> Tuple[int, int]:
        return self.project_id[activity], self.local_activity_no[activity]

    def topological_order(self, priorities: Optional[Sequence[float]] = None) -> List[int]:
        if priorities is None:
            priorities = [0.0] * self.n
        indegree = [len(self.pred[i]) for i in range(self.n)]
        ready = [i for i, degree in enumerate(indegree) if degree == 0]
        result: List[int] = []
        while ready:
            activity = min(ready, key=lambda i: (priorities[i], *self.tie_key(i)))
            ready.remove(activity)
            result.append(activity)
            for successor in self.succ[activity]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        return result

@dataclass
class Schedule:
    start: List[int]
    finish: List[int]
    cmax: float
    tpm: float

@dataclass
class Individual:
    rk: List[float]
    schedule: Schedule

    @property
    def fitness(self) -> float:
        return self.schedule.tpm

@dataclass
class B23Result:
    best_tpm: float
    best_cmax: float
    best_start: List[int]
    best_finish: List[int]
    best_rk: List[float]
    n_sgs: int
    runtime_s: float
    history: List[Tuple[int, float]] = field(default_factory=list)

def _schedule_from_times(prob: Problem, start: List[int], finish: List[int]) -> Schedule:
    cmax = max(finish) if finish else 0
    project_finishes = [finish[a] for a in prob.project_end]
    project_starts = [start[a] for a in prob.project_start]
    tpm = max(project_finishes) - min(project_starts) if project_finishes else 0
    return Schedule(start, finish, float(cmax), float(tpm))

USE_FAST_SGS = True

def _earliest_start_fast(
    prob: Problem,
    activity: int,
    earliest: int,
    used: List[List[int]],
    project_mcr_used: Optional[List[List[int]]] = None,
    project_cap: Optional[Dict[int, int]] = None,
    portfolio_cap: Optional[float] = None,
) -> int:
    duration = prob.dur[activity]
    if duration == 0:
        return earliest
    latest = prob.horizon - duration
    dem = prob.dem[activity]
    cap = prob.cap
    mcr = prob.mcr

    limits = []
    for k, requirement in enumerate(dem):
        if requirement == 0:
            continue
        limit = cap[k]
        if k == mcr and portfolio_cap is not None:
            limit = min(limit, portfolio_cap)

        limits.append((k, limit - requirement + 1e-12))

    project_restriction = None
    if project_cap is not None:
        pj = prob.project_id[activity]
        if pj in project_cap:
            assert project_mcr_used is not None
            project_restriction = (pj, project_cap[pj], dem[mcr])

    start = earliest
    while start <= latest:
        end = start + duration
        blocking_offset = None

        for k, threshold in limits:
            row = used[k]
            t = start
            while t < end:
                if row[t] > threshold:
                    blocking_offset = t - start
                    break
                t += 1
            if blocking_offset is not None:
                break

        if blocking_offset is None and project_restriction is not None:
            pj, restriction, requirement = project_restriction
            row = project_mcr_used[pj]
            t = start
            while t < end:
                if row[t] + requirement > restriction:
                    blocking_offset = t - start
                    break
                t += 1

        if blocking_offset is None:
            return start

        start = start + blocking_offset + 1

    raise RuntimeError(f"no feasible start found for activity {activity}")

def _earliest_start_reference(
    prob: Problem,
    activity: int,
    earliest: int,
    used: List[List[int]],
    project_mcr_used: Optional[List[List[int]]] = None,
    project_cap: Optional[Dict[int, int]] = None,
    portfolio_cap: Optional[float] = None,
) -> int:

    duration = prob.dur[activity]
    if duration == 0:
        return earliest
    latest = prob.horizon - duration
    for start in range(earliest, latest + 1):
        feasible = True
        for k, requirement in enumerate(prob.dem[activity]):
            if requirement == 0:
                continue
            limit = prob.cap[k]
            if k == prob.mcr and portfolio_cap is not None:
                limit = min(limit, portfolio_cap)
            if any(used[k][t] + requirement > limit + 1e-12 for t in range(start, start + duration)):
                feasible = False
                break
        if not feasible:
            continue
        if project_cap is not None and prob.project_id[activity] in project_cap:
            assert project_mcr_used is not None
            pj = prob.project_id[activity]
            restriction = project_cap[pj]
            requirement = prob.dem[activity][prob.mcr]
            if any(project_mcr_used[pj][t] + requirement > restriction for t in range(start, start + duration)):
                continue
        return start
    raise RuntimeError(f"no feasible start found for activity {activity}")

def _earliest_start(
    prob: Problem,
    activity: int,
    earliest: int,
    used: List[List[int]],
    project_mcr_used: Optional[List[List[int]]] = None,
    project_cap: Optional[Dict[int, int]] = None,
    portfolio_cap: Optional[float] = None,
) -> int:

    if USE_FAST_SGS:
        return _earliest_start_fast(
            prob, activity, earliest, used, project_mcr_used, project_cap, portfolio_cap,
        )
    return _earliest_start_reference(
        prob, activity, earliest, used, project_mcr_used, project_cap, portfolio_cap,
    )

def serial_sgs(
    prob: Problem,
    order: Sequence[int],
    *,
    project_cap: Optional[Dict[int, int]] = None,
    portfolio_cap_at: Optional[Callable[[int], float]] = None,
) -> Schedule:

    if len(order) != prob.n or len(set(order)) != prob.n:
        raise ValueError("order must contain every activity exactly once")
    start = [0] * prob.n
    finish = [0] * prob.n
    scheduled = [False] * prob.n
    used = [[0] * prob.horizon for _ in prob.cap]
    project_mcr_used = [[0] * prob.horizon for _ in range(prob.n_projects)] if project_cap else None
    placed_non_dummy = 0
    for activity in order:
        if any(not scheduled[p] for p in prob.pred[activity]):
            raise ValueError("serial SGS received a precedence-infeasible order")
        earliest = max(
            prob.release[activity],
            max((finish[p] for p in prob.pred[activity]), default=prob.release[activity]),
        )
        portfolio_cap = portfolio_cap_at(placed_non_dummy) if portfolio_cap_at else None
        activity_start = _earliest_start(
            prob, activity, earliest, used, project_mcr_used, project_cap, portfolio_cap
        )
        activity_finish = activity_start + prob.dur[activity]
        start[activity] = activity_start
        finish[activity] = activity_finish
        scheduled[activity] = True
        for k, requirement in enumerate(prob.dem[activity]):
            for t in range(activity_start, activity_finish):
                used[k][t] += requirement
        if project_mcr_used is not None:
            pj = prob.project_id[activity]
            requirement = prob.dem[activity][prob.mcr]
            for t in range(activity_start, activity_finish):
                project_mcr_used[pj][t] += requirement
        if not prob.is_dummy[activity]:
            placed_non_dummy += 1
    return _schedule_from_times(prob, start, finish)

def topo_order_from_rk(prob: Problem, rk: Sequence[float]) -> List[int]:
    return prob.topological_order(rk)

def rk_from_schedule(prob: Problem, schedule: Schedule) -> List[float]:
    earliest = min(schedule.start)
    latest = max(schedule.start)
    if latest == earliest:
        return [0.0] * prob.n
    span = latest - earliest
    return [(schedule.start[i] - earliest) / span for i in range(prob.n)]

@dataclass(frozen=True)
class PriorityMetrics:
    es: Tuple[int, ...]
    ef: Tuple[int, ...]
    ls: Tuple[int, ...]
    lf: Tuple[int, ...]
    slack: Tuple[int, ...]
    bottom_duration: Tuple[int, ...]
    project_cp: Tuple[int, ...]
    project_sp: Tuple[float, ...]
    project_twk: Tuple[int, ...]
    project_twkmcr: Tuple[int, ...]
    rlb1_due_date: Tuple[int, ...]

def calculate_priority_metrics(prob: Problem) -> PriorityMetrics:
    order = prob.topological_order()
    es = [0] * prob.n
    ef = [0] * prob.n
    for activity in order:
        es[activity] = max(
            prob.release[activity],
            max((ef[p] for p in prob.pred[activity]), default=prob.release[activity]),
        )
        ef[activity] = es[activity] + prob.dur[activity]
    project_cp = [ef[prob.project_end[pj]] - prob.release[prob.project_end[pj]] for pj in range(prob.n_projects)]
    ls = [0] * prob.n
    lf = [0] * prob.n
    for activity in reversed(order):
        pj = prob.project_id[activity]
        deadline = prob.release[activity] + project_cp[pj]
        lf[activity] = min((ls[s] for s in prob.succ[activity]), default=deadline)
        ls[activity] = lf[activity] - prob.dur[activity]
    bottom = [0] * prob.n
    levels = [0] * prob.n
    for activity in reversed(order):
        bottom[activity] = prob.dur[activity] + max((bottom[s] for s in prob.succ[activity]), default=0)
    for activity in order:
        own = 0 if prob.is_dummy[activity] else 1
        levels[activity] = own + max((levels[p] for p in prob.pred[activity]), default=0)

    project_sp: List[float] = []
    project_twk: List[int] = []
    project_twkmcr: List[int] = []
    for activities in prob.project_of:
        non_dummy = [a for a in activities if not prob.is_dummy[a]]
        count = len(non_dummy)
        longest_chain = max((levels[a] for a in non_dummy), default=1)
        project_sp.append(1.0 if count <= 1 else (longest_chain - 1) / (count - 1))
        project_twk.append(sum(prob.dur[a] * sum(prob.dem[a]) for a in non_dummy))
        project_twkmcr.append(sum(prob.dur[a] * prob.dem[a][prob.mcr] for a in non_dummy))
    work_by_resource = [
        sum(prob.dur[a] * prob.dem[a][k] for a in range(prob.n))
        for k in range(len(prob.cap))
    ]
    rlb1 = math.ceil(max(work_by_resource[k] / prob.cap[k] for k in range(len(prob.cap))))
    due_dates = tuple(prob.release[prob.project_start[pj]] + rlb1 for pj in range(prob.n_projects))
    return PriorityMetrics(
        tuple(es), tuple(ef), tuple(ls), tuple(lf),
        tuple(ls[i] - es[i] for i in range(prob.n)), tuple(bottom),
        tuple(project_cp), tuple(project_sp), tuple(project_twk),
        tuple(project_twkmcr), due_dates,
    )

STATIC_PROJECT_RULES: Tuple[Tuple[str, bool], ...] = (
    ("CP", False), ("CP", True), ("SP", False), ("SP", True),
    ("TWK", False), ("TWK", True), ("RELEASE", False), ("RELEASE", True),
    ("TWKMCR", False), ("TWKMCR", True),
)
ACTIVITY_RULES: Tuple[Tuple[str, bool], ...] = (
    ("ES", False), ("ES", True), ("LS", False), ("LS", True),
    ("EF", False), ("EF", True), ("LF", False), ("LF", True),
    ("SLACK", False), ("SLACK", True), ("DURATION", False), ("DURATION", True),
)
DYNAMIC_PROJECT_RULES: Tuple[str, ...] = (
    "REM_CP", "REM_TWK", "REM_TWKMCR", "EXPECTED_MAKESPAN", "EXPECTED_DELAY"
)

def initial_schedule_count(population_size: int = PAPER_PARAMETERS.population_size) -> int:
    if population_size % 5 != 0:
        raise ValueError("B23 population size must be divisible by five")
    group_size = population_size // 5
    return (
        len(STATIC_PROJECT_RULES) * len(ACTIVITY_RULES)
        + len(DYNAMIC_PROJECT_RULES) * len(ACTIVITY_RULES)
        + 3 * group_size
    )

INITIAL_SCHEDULE_COUNT = initial_schedule_count(PAPER_PARAMETERS.population_size)

def _rank_key(value: float, maximize: bool, tie: Tuple[int, ...]) -> Tuple[float, ...]:
    return ((-value if maximize else value), *tie)

def _activity_value(metrics: PriorityMetrics, prob: Problem, activity: int, name: str) -> float:
    values = {
        "ES": metrics.es[activity], "LS": metrics.ls[activity],
        "EF": metrics.ef[activity], "LF": metrics.lf[activity],
        "SLACK": metrics.slack[activity], "DURATION": prob.dur[activity],
    }
    if name not in values:
        raise ValueError(f"unknown activity rule {name}")
    return values[name]

def _project_value(metrics: PriorityMetrics, prob: Problem, project: int, name: str) -> float:
    values = {
        "CP": metrics.project_cp[project], "SP": metrics.project_sp[project],
        "TWK": metrics.project_twk[project],
        "RELEASE": prob.release[prob.project_start[project]],
        "TWKMCR": metrics.project_twkmcr[project],
    }
    if name not in values:
        raise ValueError(f"unknown project rule {name}")
    return values[name]

def _topological_project_order(
    prob: Problem,
    project: int,
    metrics: PriorityMetrics,
    activity_rule: Tuple[str, bool],
) -> List[int]:
    name, maximize = activity_rule
    activities = set(prob.project_of[project])
    indegree = {a: sum(p in activities for p in prob.pred[a]) for a in activities}
    ready = [a for a in activities if indegree[a] == 0]
    result: List[int] = []
    while ready:
        activity = min(
            ready,
            key=lambda a: _rank_key(
                _activity_value(metrics, prob, a, name), maximize,
                (prob.local_activity_no[a],),
            ),
        )
        ready.remove(activity)
        result.append(activity)
        for successor in prob.succ[activity]:
            if successor in activities:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
    return result

def static_priority_order(
    prob: Problem,
    metrics: PriorityMetrics,
    project_rule: Tuple[str, bool],
    activity_rule: Tuple[str, bool],
) -> List[int]:
    name, maximize = project_rule
    projects = sorted(
        range(prob.n_projects),
        key=lambda pj: _rank_key(_project_value(metrics, prob, pj, name), maximize, (pj,)),
    )
    result: List[int] = []
    for project in projects:
        result.extend(_topological_project_order(prob, project, metrics, activity_rule))
    return result

def dynamic_priority_schedule(
    prob: Problem,
    metrics: PriorityMetrics,
    project_rule: str,
    activity_rule: Tuple[str, bool],
) -> Schedule:

    start = [0] * prob.n
    finish = [0] * prob.n
    scheduled = [False] * prob.n
    used = [[0] * prob.horizon for _ in prob.cap]
    remaining = [set(activities) for activities in prob.project_of]
    activity_name, activity_maximize = activity_rule
    while any(remaining):
        ready_by_project: Dict[int, List[int]] = {}
        for pj, activities in enumerate(remaining):
            ready = [a for a in activities if all(scheduled[p] for p in prob.pred[a])]
            if ready:
                ready_by_project[pj] = ready
        if not ready_by_project:
            raise ValueError("dynamic priority construction reached a precedence deadlock")
        project_scores: Dict[int, float] = {}
        for pj, ready in ready_by_project.items():
            non_dummy_remaining = [a for a in remaining[pj] if not prob.is_dummy[a]]
            rem_cp = max((metrics.bottom_duration[a] for a in ready), default=0)
            rem_twk = sum(prob.dur[a] * sum(prob.dem[a]) for a in non_dummy_remaining)
            rem_twkmcr = sum(prob.dur[a] * prob.dem[a][prob.mcr] for a in non_dummy_remaining)
            latest_finish = max(
                (finish[a] for a in prob.project_of[pj] if scheduled[a]),
                default=prob.release[prob.project_start[pj]],
            )
            expected_makespan = latest_finish + rem_cp
            scores = {
                "REM_CP": rem_cp,
                "REM_TWK": rem_twk,
                "REM_TWKMCR": rem_twkmcr,
                "EXPECTED_MAKESPAN": expected_makespan,
                "EXPECTED_DELAY": expected_makespan - metrics.rlb1_due_date[pj],
            }
            if project_rule not in scores:
                raise ValueError(f"unknown dynamic project rule {project_rule}")
            project_scores[pj] = scores[project_rule]
        project = min(ready_by_project, key=lambda pj: (-project_scores[pj], pj))
        activity = min(
            ready_by_project[project],
            key=lambda a: _rank_key(
                _activity_value(metrics, prob, a, activity_name), activity_maximize,
                (prob.local_activity_no[a],),
            ),
        )
        earliest = max(
            prob.release[activity],
            max((finish[p] for p in prob.pred[activity]), default=prob.release[activity]),
        )
        activity_start = _earliest_start(prob, activity, earliest, used)
        activity_finish = activity_start + prob.dur[activity]
        start[activity] = activity_start
        finish[activity] = activity_finish
        scheduled[activity] = True
        remaining[project].remove(activity)
        for k, requirement in enumerate(prob.dem[activity]):
            for t in range(activity_start, activity_finish):
                used[k][t] += requirement
    return _schedule_from_times(prob, start, finish)

def _random_project_topological_order(prob: Problem, project: int, rng: random.Random) -> List[int]:
    activities = set(prob.project_of[project])
    indegree = {a: sum(p in activities for p in prob.pred[a]) for a in activities}
    ready = [a for a in activities if indegree[a] == 0]
    result: List[int] = []
    while ready:
        activity = rng.choice(ready)
        ready.remove(activity)
        result.append(activity)
        for successor in prob.succ[activity]:
            if successor in activities:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
    return result

def sequential_random_order(prob: Problem, rng: random.Random) -> List[int]:
    projects = list(range(prob.n_projects))
    rng.shuffle(projects)
    result: List[int] = []
    for project in projects:
        result.extend(_random_project_topological_order(prob, project, rng))
    return result

def interleaved_random_order(prob: Problem, rng: random.Random) -> List[int]:
    projects = list(range(prob.n_projects))
    rng.shuffle(projects)
    project_orders = {pj: _random_project_topological_order(prob, pj, rng) for pj in projects}
    result: List[int] = []
    position = {pj: 0 for pj in projects}
    while len(result) < prob.n:
        for pj in projects:
            pos = position[pj]
            if pos < len(project_orders[pj]):
                result.append(project_orders[pj][pos])
                position[pj] += 1
    return result

def build_initial_population(
    prob: Problem,
    rng: random.Random,
    population_size: int = PAPER_PARAMETERS.population_size,
) -> Tuple[List[Individual], int]:

    if population_size % 5 != 0:
        raise ValueError("B23 population size must be divisible by five")
    group_size = population_size // 5
    if group_size > 60:
        raise ValueError("dynamic-priority initialization limits p to at most 300")
    metrics = calculate_priority_metrics(prob)
    generated = 0
    static_candidates: List[Individual] = []
    for project_rule in STATIC_PROJECT_RULES:
        for activity_rule in ACTIVITY_RULES:
            schedule = serial_sgs(prob, static_priority_order(prob, metrics, project_rule, activity_rule))
            static_candidates.append(Individual(rk_from_schedule(prob, schedule), schedule))
            generated += 1
    static_candidates.sort(key=lambda ind: ind.fitness)
    dynamic_candidates: List[Individual] = []
    for project_rule in DYNAMIC_PROJECT_RULES:
        for activity_rule in ACTIVITY_RULES:
            schedule = dynamic_priority_schedule(prob, metrics, project_rule, activity_rule)
            dynamic_candidates.append(Individual(rk_from_schedule(prob, schedule), schedule))
            generated += 1
    dynamic_candidates.sort(key=lambda ind: ind.fitness)
    population = static_candidates[:group_size] + dynamic_candidates[:group_size]
    for _ in range(group_size):
        schedule = serial_sgs(prob, sequential_random_order(prob, rng))
        population.append(Individual(rk_from_schedule(prob, schedule), schedule))
        generated += 1
    for _ in range(group_size):
        schedule = serial_sgs(prob, interleaved_random_order(prob, rng))
        population.append(Individual(rk_from_schedule(prob, schedule), schedule))
        generated += 1
    for _ in range(group_size):
        rk = [rng.random() for _ in range(prob.n)]
        schedule = serial_sgs(prob, topo_order_from_rk(prob, rk))
        population.append(Individual(rk_from_schedule(prob, schedule), schedule))
        generated += 1
    if len(population) != population_size:
        raise AssertionError("initial population size mismatch")
    return population, generated

def fixed_project_restriction(
    prob: Problem,
    baseline: Schedule,
    rng: random.Random,
    x1: float,
    x2: float,
) -> Dict[int, int]:
    portfolio_finish = max(baseline.finish[a] for a in prob.project_end)
    project = min(
        range(prob.n_projects),
        key=lambda pj: (-(portfolio_finish - baseline.finish[prob.project_end[pj]]), pj),
    )
    lower_feasible = max(prob.dem[a][prob.mcr] for a in prob.project_of[project])
    upper_feasible = prob.cap[prob.mcr]
    lower = math.ceil(lower_feasible + x1 * (upper_feasible - lower_feasible))
    upper = math.floor(lower_feasible + x2 * (upper_feasible - lower_feasible))
    return {project: rng.randint(lower, max(lower, upper))}

def variable_portfolio_restriction(
    prob: Problem,
    rng: random.Random,
    y1: float,
    y2: float,
    z1: float,
    z2: float,
    mode: str,
) -> Callable[[int], float]:
    lower_feasible = max(prob.dem[a][prob.mcr] for a in range(prob.n))
    upper_feasible = prob.cap[prob.mcr]
    lower = math.ceil(lower_feasible + y1 * (upper_feasible - lower_feasible))
    upper = math.floor(lower_feasible + y2 * (upper_feasible - lower_feasible))
    initial = rng.randint(lower, max(lower, upper))
    target_lower = math.ceil(1 + z1 * (prob.n_non_dummy - 1))
    target_upper = math.floor(1 + z2 * (prob.n_non_dummy - 1))
    target = rng.randint(target_lower, max(target_lower, target_upper))
    capacity = prob.cap[prob.mcr]

    def restriction(scheduled_non_dummy: int) -> float:
        if scheduled_non_dummy >= target:
            return float(capacity)
        if mode == "constant":
            return float(initial)
        fraction = scheduled_non_dummy / target
        if mode == "linear":
            return initial + (capacity - initial) * fraction
        if mode == "concave":
            cumulative = scheduled_non_dummy * (scheduled_non_dummy + 1) / (target * target)
            return min(float(capacity), initial + (capacity - initial) * cumulative)
        raise ValueError(f"unknown reduction mode {mode}")

    return restriction

def backward_sgs(prob: Problem, forward: Schedule) -> Schedule:

    deadline = int(forward.cmax)
    start = [0] * prob.n
    finish = [0] * prob.n
    scheduled = [False] * prob.n
    used = [[0] * (deadline + 1) for _ in prob.cap]
    remaining_successors = [len(prob.succ[a]) for a in range(prob.n)]
    ready = [a for a in range(prob.n) if remaining_successors[a] == 0]
    while ready:
        activity = min(ready, key=lambda a: (-forward.finish[a], *prob.tie_key(a)))
        ready.remove(activity)
        latest_finish = min((start[s] for s in prob.succ[activity]), default=deadline)
        duration = prob.dur[activity]
        chosen: Optional[int] = None
        for activity_start in range(latest_finish - duration, prob.release[activity] - 1, -1):
            feasible = True
            for k, requirement in enumerate(prob.dem[activity]):
                if requirement and any(
                    used[k][t] + requirement > prob.cap[k]
                    for t in range(activity_start, activity_start + duration)
                ):
                    feasible = False
                    break
            if feasible:
                chosen = activity_start
                break
        if chosen is None:
            raise RuntimeError("backward SGS could not construct a feasible schedule")
        start[activity] = chosen
        finish[activity] = chosen + duration
        scheduled[activity] = True
        for k, requirement in enumerate(prob.dem[activity]):
            for t in range(start[activity], finish[activity]):
                used[k][t] += requirement
        for predecessor in prob.pred[activity]:
            remaining_successors[predecessor] -= 1
            if remaining_successors[predecessor] == 0:
                ready.append(predecessor)
    if not all(scheduled):
        raise ValueError("backward SGS reached a precedence deadlock")
    return _schedule_from_times(prob, start, finish)

def iterative_backward_forward(
    prob: Problem,
    baseline: Schedule,
    max_additional_schedules: int,
) -> Tuple[Schedule, int]:
    best = baseline
    current = baseline
    generated = 0
    while max_additional_schedules - generated >= 2:
        backward = backward_sgs(prob, current)
        generated += 1
        forward = serial_sgs(prob, prob.topological_order(backward.start))
        generated += 1
        if forward.tpm < best.tpm:
            best = forward
            current = forward
        else:
            break
    return best, generated

def two_point_crossover(
    first: Sequence[float], second: Sequence[float], rng: random.Random
) -> Tuple[List[float], List[float]]:
    if len(first) != len(second):
        raise ValueError("parents must have equal chromosome lengths")
    if len(first) < 2:
        return list(first), list(second)
    left, right = sorted(rng.sample(range(len(first)), 2))
    return (
        list(first[:left]) + list(second[left:right]) + list(first[right:]),
        list(second[:left]) + list(first[left:right]) + list(second[right:]),
    )

def mutate(
    rk: Sequence[float],
    rng: random.Random,
    mutation_rate: float = PAPER_PARAMETERS.mutation_rate,
    gene_probability: float = PAPER_PARAMETERS.gene_mutation_probability,
) -> List[float]:
    result = list(rk)
    if rng.random() < mutation_rate:
        for i in range(len(result)):
            if rng.random() < gene_probability:
                result[i] = rng.random()
    return result

def evaluate_rk(
    prob: Problem,
    rk: Sequence[float],
    rng: random.Random,
    params: B23Parameters,
    remaining_budget: int,
) -> Tuple[Individual, int]:

    if remaining_budget < 1:
        raise ValueError("at least one remaining schedule generation is required")
    order = topo_order_from_rk(prob, rk)
    baseline = serial_sgs(prob, order)
    schedules = [baseline]
    generated = 1
    if generated < remaining_budget and rng.random() < params.sfp:
        restriction = fixed_project_restriction(prob, baseline, rng, params.x1, params.x2)
        schedules.append(serial_sgs(prob, order, project_cap=restriction))
        generated += 1
    if generated < remaining_budget and rng.random() < params.svp:
        restriction_at = variable_portfolio_restriction(
            prob, rng, params.y1, params.y2, params.z1, params.z2, params.reduction_mode
        )
        schedules.append(serial_sgs(prob, order, portfolio_cap_at=restriction_at))
        generated += 1
    if remaining_budget - generated >= 2 and rng.random() < params.sbf:
        bf_schedule, bf_generated = iterative_backward_forward(
            prob, baseline, remaining_budget - generated
        )
        schedules.append(bf_schedule)
        generated += bf_generated
    best = min(schedules, key=lambda schedule: schedule.tpm)
    return Individual(rk_from_schedule(prob, best), best), generated

def _tournament_fitness(population: Sequence[Individual], rng: random.Random) -> Individual:
    winners: List[Individual] = []
    for _ in range(2):
        first = population[rng.randrange(len(population))]
        second = population[rng.randrange(len(population))]
        winners.append(first if first.fitness <= second.fitness else second)
    return winners[0] if winners[0].fitness <= winners[1].fitness else winners[1]

def validate_schedule(prob: Problem, schedule: Schedule) -> None:

    if len(schedule.start) != prob.n or len(schedule.finish) != prob.n:
        raise AssertionError("schedule length mismatch")
    for activity in range(prob.n):
        if schedule.start[activity] < prob.release[activity]:
            raise AssertionError(f"release violation at activity {activity}")
        if schedule.finish[activity] != schedule.start[activity] + prob.dur[activity]:
            raise AssertionError(f"duration violation at activity {activity}")
        for predecessor in prob.pred[activity]:
            if schedule.start[activity] < schedule.finish[predecessor]:
                raise AssertionError(f"precedence violation {predecessor}->{activity}")
    horizon = max(schedule.finish, default=0)
    for t in range(horizon):
        for k, capacity in enumerate(prob.cap):
            use = sum(
                prob.dem[a][k]
                for a in range(prob.n)
                if schedule.start[a] <= t < schedule.finish[a]
            )
            if use > capacity:
                raise AssertionError(f"resource {k} violation at time {t}: {use}>{capacity}")

def run_b23(
    rcmp_data: dict,
    *,
    seed: int = 1,
    s_max: int = PAPER_PARAMETERS.schedule_generation_budget,
    verbose: bool = False,
) -> B23Result:

    params = PAPER_PARAMETERS
    rng = random.Random(seed)
    started = time.perf_counter()
    prob = Problem.from_rcmp_json(rcmp_data)
    population, n_sgs = build_initial_population(prob, rng, params.population_size)
    if s_max < n_sgs:
        raise ValueError(f"Sg={s_max} is smaller than B23's {n_sgs}-schedule initialization")
    incumbent = min(population, key=lambda ind: ind.fitness)
    history: List[Tuple[int, float]] = [(n_sgs, incumbent.fitness)]
    elite_count = int(params.elite_fraction * params.population_size)
    while n_sgs < s_max:
        ranked = sorted(population, key=lambda ind: ind.fitness)
        children = [Individual(ind.rk[:], ind.schedule) for ind in ranked[:elite_count]]
        while len(children) < params.population_size and n_sgs < s_max:
            selected_parent = _tournament_fitness(population, rng)
            uniform_parent = population[rng.randrange(len(population))]
            offspring = two_point_crossover(selected_parent.rk, uniform_parent.rk, rng)
            for chromosome in offspring:
                if len(children) >= params.population_size or n_sgs >= s_max:
                    break
                chromosome = mutate(
                    chromosome, rng, params.mutation_rate, params.gene_mutation_probability
                )
                child, generated = evaluate_rk(
                    prob, chromosome, rng, params, s_max - n_sgs
                )
                n_sgs += generated
                children.append(child)
                if child.fitness < incumbent.fitness:
                    incumbent = child
                    history.append((n_sgs, incumbent.fitness))
                    if verbose:
                        print(f"    SGS={n_sgs} new best TPM={incumbent.fitness:.0f}", flush=True)
        if len(children) == params.population_size:
            population = children
        else:
            break
    validate_schedule(prob, incumbent.schedule)
    if n_sgs != s_max:
        raise AssertionError(f"schedule-generation budget mismatch: {n_sgs} != {s_max}")
    return B23Result(
        incumbent.schedule.tpm, incumbent.schedule.cmax,
        incumbent.schedule.start[:], incumbent.schedule.finish[:], incumbent.rk[:],
        n_sgs, time.perf_counter() - started, history,
    )

B23_REF = {
    "MPLIB1_Set1_1": {"bks": 536, "b23": 537},
    "MPLIB1_Set1_2": {"bks": 276, "b23": 276},
    "MPLIB1_Set1_3": {"bks": 464, "b23": 464},
    "MPLIB1_Set1_4": {"bks": 486, "b23": 486},
    "MPLIB1_Set1_5": {"bks": 798, "b23": 798},
    "MPLIB1_Set1_6": {"bks": 587, "b23": 587},
    "MPLIB1_Set1_7": {"bks": 835, "b23": 835},
}

VALIDATION_INSTANCES = (
    "mpj30a2", "mpj30a4", "mpj30a6", "mpj30a10",
    "mpj60a3", "mpj60a5", "mpj90a3", "mpj90a4",
    "mpj120a2", "mpj120a3", "HHH+3", "MP-MD4",
    "MPLIB1_Set1_2", "MPLIB1_Set1_3", "MPLIB1_Set1_4",
    "MPLIB1_Set1_5", "MPLIB1_Set1_6", "MPLIB1_Set1_7",
    "MPLIB2_Set1_2", "MPLIB2_Set1_3", "MPLIB2_Set1_4",
    "MPLIB2_Set1_5", "MPLIB2_Set1_6", "MPLIB2_Set1_7",
)

def _locate_instance(root: Path, instance: str) -> Path:
    direct = root / f"{instance}.json"
    if direct.exists():
        return direct
    matches = list(root.glob(f"*/{instance}.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"cannot find {instance}.json below {root}")
    raise RuntimeError(f"multiple files found for {instance}: {matches}")

def _write_results(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "algorithm", "instance", "run", "seed", "best_tpm", "best_cmax",
        "n_sgs", "runtime_s", "b23_reference", "gap_to_b23_percent",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main() -> int:
    parser = argparse.ArgumentParser(description="Independent B23 reproduction for TPM")
    parser.add_argument(
        "--root", type=Path,
        default=CODE_DIR / "data" / "dataset" / "JSON",
    )
    parser.add_argument(
        "--instances", nargs="*",
        default=[f"MPLIB1_Set1_{i}" for i in range(1, 8)],
    )
    parser.add_argument(
        "--validation-set", action="store_true",
        help="run the 24 comparison instances listed in data/dataset/README.md",
    )
    parser.add_argument(
        "--sg", type=int, default=PAPER_PARAMETERS.schedule_generation_budget,
        help="schedule-generation budget; paper setting is 100000",
    )
    parser.add_argument("--seed", type=int, default=1, help="first paired seed")
    parser.add_argument("--runs", type=int, default=1, help="number of independent runs")
    parser.add_argument("--output", type=Path, default=None, help="optional raw-run CSV")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.sg != PAPER_PARAMETERS.schedule_generation_budget:
        print(f"WARNING: Sg={args.sg} is diagnostic; the paper setting is 100000.")
    print("B23 TPM | p=250 m=0.20 Sfp=1.00 Svp=0.25 Sbf=0.50 " f"Sg={args.sg}")
    print(f"{'instance':<18} {'run':>4} {'seed':>6} {'B23':>7} {'ours':>7} {'gap%':>8} {'SGS':>8} {'time':>9}")
    print("-" * 78)
    rows: List[dict] = []
    instances = VALIDATION_INSTANCES if args.validation_set else args.instances
    for instance in instances:
        json_path = _locate_instance(args.root, instance)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for run in range(1, args.runs + 1):
            seed = args.seed + run - 1
            result = run_b23(data, seed=seed, s_max=args.sg, verbose=not args.quiet)
            reference = B23_REF.get(instance, {}).get("b23")
            gap = 100.0 * (result.best_tpm - reference) / reference if reference is not None else math.nan
            print(
                f"{instance:<18} {run:4d} {seed:6d} {str(reference):>7} "
                f"{result.best_tpm:7.0f} {gap:8.2f} {result.n_sgs:8d} {result.runtime_s:8.1f}s"
            )
            rows.append({
                "algorithm": "B23", "instance": instance, "run": run, "seed": seed,
                "best_tpm": result.best_tpm, "best_cmax": result.best_cmax,
                "n_sgs": result.n_sgs, "runtime_s": result.runtime_s,
                "b23_reference": reference, "gap_to_b23_percent": gap,
            })
    if args.output is not None:
        _write_results(args.output, rows)
        print(f"Raw results written to {args.output.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
