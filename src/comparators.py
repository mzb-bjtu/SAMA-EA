"""Adapters for common-objective comparator experiments."""
from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from evaluator import Evaluator, precedence_lists, profile_change_points, random_keys_to_order
from problem_loader import Instance, validate_schedule

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

def _find_code_dir() -> str:
    here = _HERE
    for _ in range(6):
        parent = os.path.dirname(here)
        if parent == here:
            break
        if os.path.isdir(os.path.join(parent, "data", "dataset", "JSON")):
            return parent
        here = parent

    return os.path.dirname(_HERE)

_CODE_DIR = _find_code_dir()
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
_BASELINES_DIR = os.path.join(_CODE_DIR, "baselines")
if os.path.isdir(_BASELINES_DIR) and _BASELINES_DIR not in sys.path:
    sys.path.insert(0, _BASELINES_DIR)

@dataclass
class ComparatorOutcome:

    algorithm: str
    instance: str
    seed: int
    best_tpm: int
    cmax: int
    best_order: List[int]
    start: List[int]
    finish: List[int]
    n_exact_sgs: int
    wall_time_s: float
    cpu_time_s: float
    decode_time_s: float
    valid_schedule: bool
    validation: Dict = field(default_factory=dict)
    convergence: List[dict] = field(default_factory=list)
    notes: str = ""
    extra: Dict = field(default_factory=dict)

def _load_rcmp_json(instance: str) -> dict:
    from experiment_config import instance_json_path
    import json

    with open(instance_json_path(instance), "r", encoding="utf-8") as handle:
        return json.load(handle)

def _finalize(
    inst: Instance,
    order: List[int],
    *,
    evaluator: Evaluator,
    algorithm: str,
    instance: str,
    seed: int,
    t_wall0: float,
    t_cpu0: float,
    convergence: Optional[List[dict]] = None,
    notes: str = "",
    extra: Optional[Dict] = None,
) -> ComparatorOutcome:

    start = _start_times(inst, order)
    finish = [s + d for s, d in zip(start, [0] + list(inst.durations) + [0])]
    validation = validate_schedule(inst, start, finish)
    return ComparatorOutcome(
        algorithm=algorithm,
        instance=instance,
        seed=seed,
        best_tpm=int(validation.get("tpm", 0)),
        cmax=int(validation.get("cmax", 0)),
        best_order=list(order),
        start=list(start),
        finish=list(finish),
        n_exact_sgs=evaluator.n_exact_sgs,
        wall_time_s=round(time.perf_counter() - t_wall0, 6),
        cpu_time_s=round(time.process_time() - t_cpu0, 6),
        decode_time_s=round(evaluator.decode_time_s, 6),
        valid_schedule=bool(validation.get("valid", False)),
        validation=validation,
        convergence=convergence or [],
        notes=notes,
        extra=extra or {},
    )

def _make_evaluator(inst: Instance) -> Evaluator:
    ev = Evaluator(inst)
    ev.bind_precedence(precedence_lists(inst))
    ev.bind_profile_changes(profile_change_points(inst))
    return ev

def _start_times(inst: Instance, order: List[int]) -> List[int]:

    from core_engine import _start_times as engine_start_times

    return engine_start_times(inst, order)

def embed_native_schedule(
    inst: Instance,
    start: List[int],
    finish: List[int],
) -> Tuple[List[int], List[int]]:

    n_real = len(inst.durations)
    if len(start) != n_real or len(finish) != n_real:
        raise ValueError(
            f"native schedule length {len(start)} != real activity count {n_real}"
        )

    earliest = min(start) if start else 0
    latest = max(finish) if finish else 0
    full_start = [earliest] + [int(v) for v in start] + [latest]
    full_finish = [earliest] + [int(v) for v in finish] + [latest]
    return full_start, full_finish

_MEALPY_MODULES = {
    "SADE": ("mealpy", "DE", "SADE"),
    "QLESCA": ("mealpy", "SCA", "QleSCA"),
}

def _mealpy_problem_dict(inst: Instance, evaluator: Evaluator) -> dict:

    from mealpy import FloatVar

    n = len(inst.durations)

    def obj_func(x):
        keys = np.asarray(x, dtype=float)
        order = random_keys_to_order(inst, keys)
        rec = evaluator.evaluate(order)
        return float(rec.fitness)

    return {
        "obj_func": obj_func,
        "bounds": FloatVar(lb=(0.0,) * n, ub=(1.0,) * n),
        "minmax": "min",
        "log_to": None,
    }

def run_mealpy(
    inst: Instance,
    *,
    algorithm: str,
    seed: int,
    epoch: int,
    pop_size: int,
) -> ComparatorOutcome:

    if algorithm not in _MEALPY_MODULES:
        raise KeyError(f"{algorithm} is not a mealpy-backed comparator")

    t_wall0 = time.perf_counter()
    t_cpu0 = time.process_time()
    evaluator = _make_evaluator(inst)
    problem = _mealpy_problem_dict(inst, evaluator)

    pkg, module_name, class_name = _MEALPY_MODULES[algorithm]

    mealpy = importlib.import_module(pkg)
    sub = getattr(mealpy, module_name)
    opt_class = getattr(sub, class_name)
    model = opt_class(epoch=epoch, pop_size=pop_size)

    best_agent = model.solve(
        problem, mode="single", termination={"max_epoch": epoch}, seed=seed,
    )

    best_pos = np.asarray(best_agent.solution, dtype=float)
    order = random_keys_to_order(inst, best_pos)

    convergence = []

    history = getattr(model, "history", None)
    best_list = list(getattr(history, "list_global_best_fit", None) or [])

    for gen, value in enumerate(best_list):
        convergence.append({
            "generation": gen,
            "best_verified_tpm": float(value),
            "n_exact_sgs_cum": pop_size * (gen + 2),
            "selected_algorithm": algorithm,
        })

    return _finalize(
        inst, order,
        evaluator=evaluator,
        algorithm=algorithm,
        instance=inst.name,
        seed=seed,
        t_wall0=t_wall0,
        t_cpu0=t_cpu0,
        convergence=convergence,
        notes="mealpy native random initialisation",
        extra={"best_fitness_reported": float(best_agent.target.fitness)},
    )

def run_b23_common_budget(
    inst: Instance,
    *,
    seed: int,
    pop_size: int,
    generations: int,
) -> ComparatorOutcome:

    import bredael_b23_ga as b23

    t_wall0 = time.perf_counter()
    t_cpu0 = time.process_time()
    evaluator = _make_evaluator(inst)
    data = _load_rcmp_json(inst.name)

    s_max = int(pop_size * generations)
    init_count = b23.initial_schedule_count(b23.PAPER_PARAMETERS.population_size)
    if s_max <= init_count:
        raise ValueError(
            f"B23_GA_CB needs a budget above its {init_count}-schedule initialisation, "
            f"got Sg={s_max}"
        )
    result = b23.run_b23(data, seed=seed, s_max=s_max, verbose=False)

    start, finish = embed_native_schedule(inst, result.best_start, result.best_finish)
    order = sorted(range(len(inst.durations)), key=lambda a: start[a + 1])

    validation = validate_schedule(inst, start, finish)

    convergence = []
    for n_sgs, value in (result.history or []):
        convergence.append({
            "generation": int(n_sgs // max(1, pop_size)),
            "n_exact_sgs_cum": int(n_sgs),
            "best_verified_tpm": float(value),
            "selected_algorithm": "B23_GA_CB",
        })

    return ComparatorOutcome(
        algorithm="B23_GA_CB",
        instance=inst.name,
        seed=seed,
        best_tpm=int(round(result.best_tpm)),
        cmax=int(round(result.best_cmax)),
        best_order=list(order),
        start=list(start),
        finish=list(finish),
        n_exact_sgs=int(result.n_sgs),
        wall_time_s=round(time.perf_counter() - t_wall0, 6),
        cpu_time_s=round(time.process_time() - t_cpu0, 6),
        decode_time_s=round(evaluator.decode_time_s, 6),
        valid_schedule=bool(validation.get("valid", False)),
        validation=validation,
        convergence=convergence,
        notes=f"B23 mechanism, common budget pop={pop_size} x gen={generations} (Sg={s_max})",
        extra={"s_max": s_max, "paper_population_size": b23.PAPER_PARAMETERS.population_size},
    )

def run_psgs_minslk(inst: Instance, *, seed: int) -> ComparatorOutcome:

    import psgs_minslk

    t_wall0 = time.perf_counter()
    t_cpu0 = time.process_time()
    evaluator = _make_evaluator(inst)
    data = _load_rcmp_json(inst.name)

    result = psgs_minslk.run_on_rcmp_json(data, seed=seed)

    start, finish = embed_native_schedule(inst, result.start, result.finish)

    order = [int(a) for a in result.activity_order]
    order = [a - 1 for a in order if 0 < a < len(inst.durations) + 1]
    if sorted(order) != list(range(len(inst.durations))):

        order = sorted(range(len(inst.durations)), key=lambda a: start[a + 1])

    validation = validate_schedule(inst, start, finish)

    return ComparatorOutcome(
        algorithm="PSGS_MINSLK",
        instance=inst.name,
        seed=seed,
        best_tpm=int(round(result.tpm)),
        cmax=int(round(result.cmax)),
        best_order=list(order),
        start=list(start),
        finish=list(finish),
        n_exact_sgs=int(result.periods),
        wall_time_s=round(time.perf_counter() - t_wall0, 6),
        cpu_time_s=round(time.process_time() - t_cpu0, 6),
        decode_time_s=round(evaluator.decode_time_s, 6),
        valid_schedule=bool(validation.get("valid", False)),
        validation=validation,
        convergence=[{
            "generation": 0,
            "best_verified_tpm": float(result.tpm),
            "n_exact_sgs_cum": int(result.periods),
            "selected_algorithm": "PSGS_MINSLK",
        }],
        notes="priority-rule serial SGS, no hyper-parameters",
        extra={"cpm_rebuilds": int(result.cpm_rebuilds)},
    )

def run_comparator(
    inst: Instance,
    *,
    algorithm: str,
    seed: int,
    pop_size: int,
    generations: int,
) -> ComparatorOutcome:

    if algorithm in ("SADE", "QLESCA"):
        return run_mealpy(
            inst, algorithm=algorithm, seed=seed,
            epoch=generations, pop_size=pop_size,
        )
    if algorithm == "B23_GA_CB":
        return run_b23_common_budget(
            inst, seed=seed, pop_size=pop_size, generations=generations,
        )
    if algorithm == "PSGS_MINSLK":
        return run_psgs_minslk(inst, seed=seed)
    raise KeyError(f"unknown comparator {algorithm!r}")
