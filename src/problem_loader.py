"""Benchmark loading, data hashing and independent schedule validation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from experiment_config import (
    INSTANCE_GROUP, JSON_ROOT, VALIDATION_INSTANCES, instance_json_path,
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def sha256_tree(root: Path, suffixes: Sequence[str] | None = None) -> str:

    digest = hashlib.sha256()
    entries: List[tuple] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        if "__pycache__" in path.parts:
            continue
        entries.append((path.relative_to(root).as_posix(), path))
    for rel, path in entries:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()

def data_hashes() -> Dict[str, str]:
    import experiment_config as cfg

    return {name: sha256_file(instance_json_path(name)) for name in VALIDATION_INSTANCES}

def code_hash() -> str:
    root = Path(__file__).resolve().parent
    return sha256_tree(root, suffixes=(".py",))

@dataclass(frozen=True)
class Instance:
    name: str
    group: str
    path: Path
    raw: dict
    num_projects: int
    num_resources: int
    resource_avail: List[int]

    durations: List[int]
    demands: List[List[int]]
    successors: List[List[int]]
    project_of: List[int]
    release: List[int]
    is_dummy: List[bool]

    @property
    def n_nodes(self) -> int:

        return len(self.durations) + 2

    @property
    def n_real_activities(self) -> int:
        return len(self.durations)

def load_instance(name: str) -> Instance:
    path = instance_json_path(name)
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return build_instance(name, path, raw)

def load_dev_instance(name: str) -> Instance:

    from experiment_config import dev_instance_json_path

    path = dev_instance_json_path(name)
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return build_instance(name, path, raw)

def build_instance(name: str, path: Path, raw: dict) -> Instance:
    n_res = int(raw["num_resources"])
    cap = [int(v) for v in raw["resource_avail"]]

    durations: List[int] = []
    demands: List[List[int]] = []
    successors: List[List[int]] = []
    project_of: List[int] = []
    release: List[int] = []
    is_dummy: List[bool] = []
    index_by_id: Dict[str, int] = {}

    projects = raw["projects"]
    for pj, project in enumerate(projects):
        rel = int(project.get("release_date", 0))
        for activity in project["activities"]:
            aid = str(activity["activity_id"])
            if aid in index_by_id:
                raise ValueError(f"{name}: duplicate activity id {aid}")
            index_by_id[aid] = len(durations)
            durations.append(int(activity["duration"]))
            demands.append([int(v) for v in activity["resource_demands"]])
            successors.append([])
            project_of.append(pj)
            release.append(rel)
            is_dummy.append(int(activity["duration"]) == 0)

    for pj, project in enumerate(projects):
        for activity in project["activities"]:
            src = index_by_id[str(activity["activity_id"])]
            for succ in activity["successors"]:
                tgt_id = f"{succ[0]}.{succ[1]}" if isinstance(succ, (list, tuple)) else str(succ)
                if tgt_id not in index_by_id:
                    raise ValueError(f"{name}: unknown successor {tgt_id}")
                successors[src].append(index_by_id[tgt_id])

    for idx in range(len(durations)):
        if any(req > cap[k] for k, req in enumerate(demands[idx])):
            raise ValueError(f"{name}: activity {idx} demand exceeds capacity")

    return Instance(
        name=name,
        group=INSTANCE_GROUP.get(name, "UNKNOWN"),
        path=path,
        raw=raw,
        num_projects=len(projects),
        num_resources=n_res,
        resource_avail=cap,
        durations=durations,
        demands=demands,
        successors=successors,
        project_of=project_of,
        release=release,
        is_dummy=is_dummy,
    )

class ScheduleInvalid(AssertionError):
    pass

def validate_schedule(
    inst: Instance,
    start: Sequence[int],
    finish: Sequence[int],
    *,
    strict: bool = True,
) -> dict:

    n_real = len(inst.durations)
    n_nodes = n_real + 2
    if len(start) != n_nodes or len(finish) != n_nodes:
        raise ScheduleInvalid(f"schedule length {len(start)} != node count {n_nodes}")

    dur = [0] + list(inst.durations) + [0]
    succ = [[] for _ in range(n_nodes)]
    pred = [[] for _ in range(n_nodes)]
    for i in range(n_real):
        for j in inst.successors[i]:
            succ[i + 1].append(j + 1)
            pred[j + 1].append(i + 1)
    for i in range(n_real):
        succ[0].append(i + 1)
        pred[i + 1].append(0)
        if not inst.successors[i]:
            succ[i + 1].append(n_real + 1)
            pred[n_real + 1].append(i + 1)

    for node in range(n_nodes):
        if start[node] < 0:
            raise ScheduleInvalid(f"node {node} starts before time 0")
        if finish[node] != start[node] + dur[node]:
            raise ScheduleInvalid(
                f"node {node}: finish {finish[node]} != start {start[node]} + duration {dur[node]}"
            )
        for p in pred[node]:
            if finish[p] > start[node]:
                raise ScheduleInvalid(
                    f"precedence violated: node {node} starts {start[node]} before predecessor {p} finishes {finish[p]}"
                )

    horizon = max(finish) if finish else 0
    if horizon > 2_000_000:
        raise ScheduleInvalid(f"implausible horizon {horizon}")

    profile = [[0] * (horizon + 1) for _ in range(inst.num_resources)]
    for node in range(1, n_real + 1):
        for k in range(inst.num_resources):
            need = inst.demands[node - 1][k]
            if need == 0:
                continue
            for t in range(start[node], finish[node]):
                profile[k][t] += need

    for k in range(inst.num_resources):
        cap = inst.resource_avail[k]
        if cap <= 0:

            limit = max(profile[k]) if profile[k] else 0
        else:
            limit = cap
        for t, used in enumerate(profile[k]):
            if used > limit:
                raise ScheduleInvalid(
                    f"resource {k} capacity {limit} exceeded at t={t}: {used}"
                )

    project_start = [min(start[1 + i] for i in range(n_real) if inst.project_of[i] == pj)
                     for pj in range(inst.num_projects)]
    project_finish = [max(finish[1 + i] for i in range(n_real) if inst.project_of[i] == pj)
                      for pj in range(inst.num_projects)]
    tpm = max(project_finish) - min(project_start)
    cmax = max(finish) - min(start)

    return {
        "valid": True,
        "tpm": int(tpm),
        "cmax": int(cmax),
        "project_start": [int(v) for v in project_start],
        "project_finish": [int(v) for v in project_finish],
        "horizon": int(horizon),
    }
