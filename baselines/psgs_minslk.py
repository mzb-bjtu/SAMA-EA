"""Problem-specific comparator reproduction."""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CODE_DIR = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class PSGSConfiguration:

    schedule_generation_scheme: str = "parallel"
    priority_rule: str = "minimum_total_slack"
    rebuild_cpm_each_period: bool = True
    tie_break: str = "random"
    time_step: int = 1

PAPER_CONFIGURATION = PSGSConfiguration()

@dataclass
class Problem:
    n: int
    n_resources: int
    n_projects: int
    activity_id: List[str]
    duration: List[int]
    demand: List[List[int]]
    predecessors: List[List[int]]
    successors: List[List[int]]
    capacity: List[int]
    project_id: List[int]
    local_activity_no: List[int]
    release: List[int]
    project_activities: List[List[int]]
    project_start: List[int]
    project_end: List[int]
    is_dummy: List[bool]
    topological_order: List[int]
    horizon: int

    @staticmethod
    def from_rcmp_json(data: dict) -> "Problem":

        projects = data.get("projects", [])
        n_projects = int(data.get("num_projects", len(projects)))
        if n_projects != len(projects):
            raise ValueError("num_projects does not match the project list")

        n_resources = int(data["num_resources"])
        capacity = [int(value) for value in data["resource_avail"]]
        if len(capacity) != n_resources or any(value <= 0 for value in capacity):
            raise ValueError("invalid renewable-resource capacity vector")

        rows: List[Tuple[int, int, dict]] = []
        index_by_id: Dict[str, int] = {}
        project_activities: List[List[int]] = [[] for _ in projects]
        project_release = [0] * n_projects
        project_flags: List[List[int]] = []

        for project_pos, project in enumerate(projects):
            project_release[project_pos] = int(project.get("release_date", 0))
            if project_release[project_pos] < 0:
                raise ValueError("negative project release dates are not supported")
            flags = [int(value) for value in project.get("resource_flags", [1] * n_resources)]
            if len(flags) != n_resources or any(value not in (0, 1) for value in flags):
                raise ValueError(f"project {project_pos + 1} has invalid resource flags")
            project_flags.append(flags)

            activities = project.get("activities", [])
            if int(project.get("num_activities", len(activities))) != len(activities):
                raise ValueError(f"project {project_pos + 1} activity count mismatch")
            for local_pos, activity in enumerate(activities, start=1):
                activity_index = len(rows)
                activity_key = str(activity["activity_id"])
                if activity_key in index_by_id:
                    raise ValueError(f"duplicate activity id: {activity_key}")
                index_by_id[activity_key] = activity_index
                try:
                    local_no = int(activity_key.rsplit(".", 1)[1])
                except (IndexError, ValueError):
                    local_no = local_pos
                rows.append((project_pos, local_no, activity))
                project_activities[project_pos].append(activity_index)

        n = len(rows)
        activity_id = [""] * n
        duration = [0] * n
        demand = [[0] * n_resources for _ in range(n)]
        predecessors: List[List[int]] = [[] for _ in range(n)]
        successors: List[List[int]] = [[] for _ in range(n)]
        project_id = [0] * n
        local_activity_no = [0] * n
        release = [0] * n

        for activity_index, (project_pos, local_no, activity) in enumerate(rows):
            activity_id[activity_index] = str(activity["activity_id"])
            project_id[activity_index] = project_pos
            local_activity_no[activity_index] = local_no
            release[activity_index] = project_release[project_pos]
            duration[activity_index] = int(activity["duration"])
            if duration[activity_index] < 0:
                raise ValueError(f"activity {activity_id[activity_index]} has negative duration")
            requirements = [int(value) for value in activity["resource_demands"]]
            if len(requirements) != n_resources or any(value < 0 for value in requirements):
                raise ValueError(f"activity {activity_id[activity_index]} has invalid demands")
            for resource, requirement in enumerate(requirements):
                if requirement > capacity[resource]:
                    raise ValueError(
                        f"activity {activity_id[activity_index]} exceeds resource {resource + 1} capacity"
                    )
                if not project_flags[project_pos][resource] and requirement:
                    raise ValueError(
                        f"activity {activity_id[activity_index]} uses a resource disabled for its project"
                    )
            demand[activity_index] = requirements

        for activity_index, (_, _, activity) in enumerate(rows):
            for successor_id in activity.get("successors", []):
                key = f"{successor_id[0]}.{successor_id[1]}"
                if key not in index_by_id:
                    raise ValueError(f"unknown successor id: {key}")
                successor = index_by_id[key]
                successors[activity_index].append(successor)
                predecessors[successor].append(activity_index)

        for activity in range(n):
            predecessors[activity] = sorted(set(predecessors[activity]))
            successors[activity] = sorted(set(successors[activity]))

        indegree = [len(predecessors[activity]) for activity in range(n)]
        ready = [activity for activity, value in enumerate(indegree) if value == 0]
        topo: List[int] = []
        while ready:
            activity = min(ready, key=lambda item: (project_id[item], local_activity_no[item]))
            ready.remove(activity)
            topo.append(activity)
            for successor in successors[activity]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        if len(topo) != n:
            raise ValueError("precedence graph contains a cycle")

        project_start: List[int] = []
        project_end: List[int] = []
        is_dummy = [False] * n
        for project_pos, activities in enumerate(project_activities):
            starts = [
                activity for activity in activities
                if not any(project_id[pred] == project_pos for pred in predecessors[activity])
            ]
            ends = [
                activity for activity in activities
                if not any(project_id[succ] == project_pos for succ in successors[activity])
            ]
            if len(starts) != 1 or len(ends) != 1:
                raise ValueError(f"project {project_pos + 1} must have one start and one end dummy")
            if duration[starts[0]] != 0 or duration[ends[0]] != 0:
                raise ValueError(f"project {project_pos + 1} boundary activities must be zero-duration")
            project_start.append(starts[0])
            project_end.append(ends[0])
            is_dummy[starts[0]] = True
            is_dummy[ends[0]] = True

        horizon = max(project_release, default=0) + sum(duration) + 1
        return Problem(
            n=n,
            n_resources=n_resources,
            n_projects=n_projects,
            activity_id=activity_id,
            duration=duration,
            demand=demand,
            predecessors=predecessors,
            successors=successors,
            capacity=capacity,
            project_id=project_id,
            local_activity_no=local_activity_no,
            release=release,
            project_activities=project_activities,
            project_start=project_start,
            project_end=project_end,
            is_dummy=is_dummy,
            topological_order=topo,
            horizon=horizon,
        )

@dataclass
class CPMState:
    earliest_start: List[int]
    earliest_finish: List[int]
    latest_start: List[int]
    latest_finish: List[int]
    total_slack: List[int]
    tentative_finish: int

@dataclass
class PSGSResult:
    tpm: float
    cmax: float
    start: List[int]
    finish: List[int]
    activity_order: List[int]
    final_slack: List[int]
    seed: int
    runtime_s: float
    periods: int
    cpm_rebuilds: int

    @property
    def start_times(self) -> Dict[int, float]:

        return {activity: float(value) for activity, value in enumerate(self.start)}

    @property
    def finish_times(self) -> Dict[int, float]:
        return {activity: float(value) for activity, value in enumerate(self.finish)}

    @property
    def activity_list(self) -> List[int]:
        return self.activity_order

    @property
    def slack(self) -> Dict[int, float]:
        return {activity: float(value) for activity, value in enumerate(self.final_slack)}

def compute_dynamic_cpm(
    problem: Problem,
    current_time: int,
    start: Sequence[Optional[int]],
    finish: Sequence[Optional[int]],
) -> CPMState:

    es = [0] * problem.n
    ef = [0] * problem.n
    for activity in problem.topological_order:
        if start[activity] is not None:
            es[activity] = int(start[activity])
            ef[activity] = int(finish[activity])
            continue
        predecessor_finish = max(
            (ef[pred] for pred in problem.predecessors[activity]),
            default=problem.release[activity],
        )
        es[activity] = max(current_time, problem.release[activity], predecessor_finish)
        ef[activity] = es[activity] + problem.duration[activity]

    tentative_finish = max(ef, default=current_time)
    ls = [0] * problem.n
    lf = [0] * problem.n
    for activity in reversed(problem.topological_order):
        lf[activity] = min(
            (ls[successor] for successor in problem.successors[activity]),
            default=tentative_finish,
        )
        ls[activity] = lf[activity] - problem.duration[activity]
    total_slack = [ls[activity] - es[activity] for activity in range(problem.n)]
    return CPMState(es, ef, ls, lf, total_slack, tentative_finish)

def _predecessors_finished(
    problem: Problem,
    activity: int,
    current_time: int,
    finish: Sequence[Optional[int]],
) -> bool:
    return all(
        finish[pred] is not None and finish[pred] <= current_time
        for pred in problem.predecessors[activity]
    )

def _fits(
    problem: Problem,
    activity: int,
    start_time: int,
    resource_use: Sequence[Sequence[int]],
) -> bool:
    end_time = start_time + problem.duration[activity]
    for resource in range(problem.n_resources):
        requirement = problem.demand[activity][resource]
        if requirement == 0:
            continue
        for period in range(start_time, end_time):
            if resource_use[resource][period] + requirement > problem.capacity[resource]:
                return False
    return True

def _schedule_activity(
    problem: Problem,
    activity: int,
    start_time: int,
    start: List[Optional[int]],
    finish: List[Optional[int]],
    resource_use: List[List[int]],
    activity_order: List[int],
) -> None:
    end_time = start_time + problem.duration[activity]
    start[activity] = start_time
    finish[activity] = end_time
    for resource in range(problem.n_resources):
        requirement = problem.demand[activity][resource]
        for period in range(start_time, end_time):
            resource_use[resource][period] += requirement
    activity_order.append(activity)

def _schedule_zero_duration_closure(
    problem: Problem,
    current_time: int,
    start: List[Optional[int]],
    finish: List[Optional[int]],
    resource_use: List[List[int]],
    activity_order: List[int],
) -> None:

    while True:
        eligible = [
            activity for activity in problem.topological_order
            if start[activity] is None
            and problem.duration[activity] == 0
            and problem.release[activity] <= current_time
            and _predecessors_finished(problem, activity, current_time, finish)
        ]
        if not eligible:
            return
        for activity in eligible:
            _schedule_activity(
                problem, activity, current_time, start, finish, resource_use, activity_order
            )

def validate_schedule(problem: Problem, result: PSGSResult) -> None:

    if len(result.start) != problem.n or len(result.finish) != problem.n:
        raise AssertionError("schedule length mismatch")
    for activity in range(problem.n):
        if result.start[activity] < problem.release[activity]:
            raise AssertionError(f"release violation at {problem.activity_id[activity]}")
        if result.finish[activity] != result.start[activity] + problem.duration[activity]:
            raise AssertionError(f"duration violation at {problem.activity_id[activity]}")
        for pred in problem.predecessors[activity]:
            if result.start[activity] < result.finish[pred]:
                raise AssertionError(
                    f"precedence violation: {problem.activity_id[pred]} -> "
                    f"{problem.activity_id[activity]}"
                )

    makespan = max(result.finish, default=0)
    for resource in range(problem.n_resources):
        for period in range(makespan):
            usage = sum(
                problem.demand[activity][resource]
                for activity in range(problem.n)
                if result.start[activity] <= period < result.finish[activity]
            )
            if usage > problem.capacity[resource]:
                raise AssertionError(
                    f"resource {resource + 1} violation at {period}: "
                    f"{usage}>{problem.capacity[resource]}"
                )

    expected_tpm = (
        max(result.finish[activity] for activity in problem.project_end)
        - min(result.start[activity] for activity in problem.project_start)
    )
    if result.tpm != expected_tpm:
        raise AssertionError(f"TPM mismatch: {result.tpm}!={expected_tpm}")

def run_psgs_minslk(rcmp_data: dict, *, seed: int = 1) -> PSGSResult:

    started = time.perf_counter()
    rng = random.Random(seed)
    problem = Problem.from_rcmp_json(rcmp_data)
    start: List[Optional[int]] = [None] * problem.n
    finish: List[Optional[int]] = [None] * problem.n
    resource_use = [[0] * problem.horizon for _ in range(problem.n_resources)]
    activity_order: List[int] = []
    final_slack = [0] * problem.n
    initial_time = min(problem.release, default=0)
    current_time = initial_time
    cpm_rebuilds = 0

    while any(value is None for value in start):
        if current_time >= problem.horizon:
            raise RuntimeError("P-SGS/MINSLK exceeded the serial upper-bound horizon")

        _schedule_zero_duration_closure(
            problem, current_time, start, finish, resource_use, activity_order
        )
        if all(value is not None for value in start):
            break

        cpm = compute_dynamic_cpm(problem, current_time, start, finish)
        cpm_rebuilds += 1
        final_slack = cpm.total_slack
        candidates = [
            activity for activity in range(problem.n)
            if start[activity] is None
            and problem.duration[activity] > 0
            and problem.release[activity] <= current_time
            and _predecessors_finished(problem, activity, current_time, finish)
        ]

        random_tie = {activity: rng.random() for activity in candidates}
        candidates.sort(key=lambda activity: (cpm.total_slack[activity], random_tie[activity]))
        for activity in candidates:
            if _fits(problem, activity, current_time, resource_use):
                _schedule_activity(
                    problem, activity, current_time, start, finish, resource_use, activity_order
                )

        current_time += PAPER_CONFIGURATION.time_step

    definitive_start = [int(value) for value in start if value is not None]
    definitive_finish = [int(value) for value in finish if value is not None]
    if len(definitive_start) != problem.n or len(definitive_finish) != problem.n:
        raise RuntimeError("incomplete P-SGS/MINSLK schedule")
    cmax = max(definitive_finish, default=0)
    tpm = (
        max(definitive_finish[activity] for activity in problem.project_end)
        - min(definitive_start[activity] for activity in problem.project_start)
    )
    result = PSGSResult(
        tpm=float(tpm),
        cmax=float(cmax),
        start=definitive_start,
        finish=definitive_finish,
        activity_order=activity_order,
        final_slack=final_slack,
        seed=seed,
        runtime_s=time.perf_counter() - started,
        periods=current_time - initial_time,
        cpm_rebuilds=cpm_rebuilds,
    )
    validate_schedule(problem, result)
    return result

def run_on_rcmp_json(rcmp_data: dict, *, seed: int = 1) -> PSGSResult:

    return run_psgs_minslk(rcmp_data, seed=seed)

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

def _write_csv(path: Path, rows: Iterable[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def _summary_rows(raw_rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = {}
    for row in raw_rows:
        grouped.setdefault(str(row["instance"]), []).append(row)
    summaries = []
    for instance, rows in grouped.items():
        values = [float(row["best_tpm"]) for row in rows]
        runtimes = [float(row["runtime_s"]) for row in rows]
        summaries.append({
            "algorithm": "P-SGS/MINSLK",
            "instance": instance,
            "runs": len(values),
            "best_tpm": min(values),
            "mean_tpm": statistics.fmean(values),
            "std_tpm": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median_tpm": statistics.median(values),
            "mean_runtime_s": statistics.fmean(runtimes),
        })
    return summaries

def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-faithful P-SGS/MINSLK for RCMPSP")
    parser.add_argument(
        "--root", type=Path, default=CODE_DIR / "data" / "dataset" / "JSON"
    )
    parser.add_argument(
        "--instances", nargs="*", default=[f"MPLIB1_Set1_{i}" for i in range(1, 8)]
    )
    parser.add_argument(
        "--validation-set", action="store_true",
        help="run the 24 comparison instances from data/dataset/README.md",
    )
    parser.add_argument("--runs", type=int, default=1, help="independent runs (protocol: 30)")
    parser.add_argument("--seed", type=int, default=1, help="first paired seed")
    parser.add_argument("--output", type=Path, default=None, help="raw-run CSV")
    parser.add_argument("--summary-output", type=Path, default=None, help="summary CSV")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")

    instances = VALIDATION_INSTANCES if args.validation_set else tuple(args.instances)
    raw_rows: List[dict] = []
    if not args.quiet:
        print("P-SGS/MINSLK | dynamic CPM | MINSLK | random ties | time step=1")
        print(f"{'instance':<20} {'run':>4} {'seed':>6} {'TPM':>8} {'Cmax':>8} {'time':>9}")
        print("-" * 62)

    for instance in instances:
        path = _locate_instance(args.root, instance)
        data = json.loads(path.read_text(encoding="utf-8"))
        for run in range(1, args.runs + 1):
            seed = args.seed + run - 1
            result = run_psgs_minslk(data, seed=seed)
            row = {
                "algorithm": "P-SGS/MINSLK",
                "instance": instance,
                "run": run,
                "seed": seed,
                "best_tpm": result.tpm,
                "best_cmax": result.cmax,
                "runtime_s": round(result.runtime_s, 6),
                "periods": result.periods,
                "cpm_rebuilds": result.cpm_rebuilds,
                "status": "ok",
            }
            raw_rows.append(row)
            if not args.quiet:
                print(
                    f"{instance:<20} {run:4d} {seed:6d} {result.tpm:8.0f} "
                    f"{result.cmax:8.0f} {result.runtime_s:9.4f}"
                )

    if args.output is not None:
        _write_csv(
            args.output,
            raw_rows,
            ("algorithm", "instance", "run", "seed", "best_tpm", "best_cmax",
             "runtime_s", "periods", "cpm_rebuilds", "status"),
        )
    if args.summary_output is not None:
        _write_csv(
            args.summary_output,
            _summary_rows(raw_rows),
            ("algorithm", "instance", "runs", "best_tpm", "mean_tpm", "std_tpm",
             "median_tpm", "mean_runtime_s"),
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
