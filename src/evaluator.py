"""Schedule generation, realignment, variation operators and evaluation accounting."""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiment_config import (
    BUDGET_DECAY, BUDGET_K_MAX, BUDGET_K_MIN, ENTROPY_COEF, N_GEN, POP_SIZE,
    STAGNATION_SCALE, SURROGATE_PARAMS, SURROGATE_TYPE,
)
from problem_loader import Instance, validate_schedule
from seeding import set_seed

@dataclass
class EvalRecord:
    fitness: int
    decode_s: float
    fully_evaluated: bool

    n_units: int = 1

class Evaluator:

    def __init__(self, inst: Instance, *, use_cache: bool = True):
        self.inst = inst
        self.n_exact_sgs = 0
        self.n_candidates = 0
        self.n_cache_hits = 0
        self.decode_time_s = 0.0
        self._cache: "OrderedDict[bytes, int]" = OrderedDict()

        self._use_cache = bool(use_cache)
        self._cache_capacity = 200_000

        self.max_seq_bound = 10 * (sum(inst.durations) + 1) + 1_000_000
        self._starts: List[int] = []

        n = len(inst.durations)
        self.max_demand = [
            max((inst.demands[i][k] for i in range(n)), default=0)
            for k in range(inst.num_resources)
        ]
        self.n_non_dummy = sum(1 for d in inst.durations if d > 0)

        self.horizon_bound = sum(inst.durations) + max(inst.release) + 2

    def buffer_profile(self, beta: float, tau: float):

        cap = self.inst.resource_avail
        maxdem = self.max_demand
        target = max(1, int(round(tau * self.n_non_dummy)))
        if beta >= 1.0 or tau <= 0.0:
            full = list(cap)
            return lambda placed: full
        lows = [max(maxdem[k], int(beta * cap[k])) for k in range(len(cap))]
        cache: Dict[int, List[int]] = {}

        def cap_at(placed: int) -> List[int]:
            got = cache.get(placed)
            if got is None:
                if placed >= target:
                    got = list(cap)
                else:
                    frac = placed / target
                    got = [
                        max(maxdem[k], int(lows[k] + (cap[k] - lows[k]) * frac))
                        for k in range(len(cap))
                    ]
                cache[placed] = got
            return got

        return cap_at

    def decode(self, order: Sequence[int], cap_at=None) -> int:

        inst = self.inst
        dur = inst.durations
        dem = inst.demands
        n_res = inst.num_resources
        cap_full = inst.resource_avail
        rel = inst.release
        pred = self._pred

        used = [[0] * self.horizon_bound for _ in range(n_res)]

        starts = [0] * len(dur)
        finish = [0] * len(dur)
        bound = self.max_seq_bound
        placed = 0
        is_dummy = inst.is_dummy

        for idx in order:
            s = rel[idx]
            for p in pred[idx]:
                fp = finish[p]
                if fp > s:
                    s = fp

            d = dur[idx]
            if d <= 0:
                starts[idx] = s
                finish[idx] = s
                continue

            row_dem = dem[idx]
            limit = cap_full if cap_at is None else cap_at(placed)

            while True:
                fe = s + d
                if fe > self.horizon_bound:
                    raise RuntimeError(
                        f"no feasible start for activity {idx} within horizon "
                        f"{self.horizon_bound}"
                    )
                blocked_at = None
                for k in range(n_res):
                    q = row_dem[k]
                    if q <= 0:
                        continue
                    threshold = limit[k] - q
                    row = used[k]
                    t = s
                    while t < fe:
                        if row[t] > threshold:
                            blocked_at = t
                            break
                        t += 1
                    if blocked_at is not None:
                        break
                if blocked_at is None:
                    break
                s = blocked_at + 1
                if s > bound:
                    raise RuntimeError(
                        f"no feasible start for activity {idx} within bound {bound}"
                    )

            starts[idx] = s
            finish[idx] = s + d
            for k in range(n_res):
                q = row_dem[k]
                if q <= 0:
                    continue
                row = used[k]
                for t in range(s, s + d):
                    row[t] += q
            if not is_dummy[idx]:
                placed += 1

        self._starts = starts
        return max(finish) if finish else 0

    def start_times_cache(self) -> List[int]:

        return self._starts

    def start_times(self, order: Sequence[int]) -> List[int]:

        self.decode(order)
        return [0] + [int(v) for v in self._starts] + [max(self._starts) if self._starts else 0]

    def _topo_order_from_starts(self, starts: Sequence[int]) -> List[int]:

        inst = self.inst
        n = len(starts)
        indeg = [0] * n
        for i in range(n):
            for j in inst.successors[i]:
                indeg[j] += 1
        ready = [i for i in range(n) if indeg[i] == 0]
        order: List[int] = []
        while ready:
            ready.sort(key=lambda i: (starts[i], i))
            node = ready.pop(0)
            order.append(node)
            for j in inst.successors[node]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    ready.append(j)
        if len(order) != n:
            raise RuntimeError("topological sort failed: graph contains a cycle")
        return order

    def _reverse_arcs(self) -> List[List[int]]:

        cached = getattr(self, "_pred_rev", None)
        if cached is None:
            n = len(self.inst.durations)
            cached = [[] for _ in range(n)]
            for i in range(n):
                for j in self.inst.successors[i]:
                    cached[j].append(i)
            self._pred_rev = cached
        return cached

    def backward_pass(self, order: Sequence[int], deadline: int,
                      cap_at=None, priority: "Optional[Sequence[float]]" = None,
                      starts_hint: "Optional[Sequence[int]]" = None) -> List[int]:

        if priority is None:
            return self._backward_list(order, deadline, cap_at)
        return self._backward_dynamic(deadline, cap_at, priority, starts_hint)

    def _backward_list(self, order: Sequence[int], deadline: int,
                       cap_at=None) -> List[int]:
        inst = self.inst
        dur = inst.durations
        dem = inst.demands
        n_res = inst.num_resources
        cap_full = inst.resource_avail
        rel = inst.release
        succ = inst.successors

        horizon = max(deadline, self.horizon_bound)
        used = [[0] * horizon for _ in range(n_res)]
        starts = [0] * len(dur)
        placed = 0

        for idx in reversed(list(order)):
            d = dur[idx]
            upper_finish = deadline
            for j in succ[idx]:
                sj = starts[j]
                if sj < upper_finish:
                    upper_finish = sj
            if d <= 0:
                starts[idx] = upper_finish
                continue
            row_dem = dem[idx]
            limit = cap_full if cap_at is None else cap_at(placed)
            lower = rel[idx]
            start = upper_finish - d
            if start < lower:
                start = lower
            while True:
                fe = start + d
                blocked_at = None
                for k in range(n_res):
                    q = row_dem[k]
                    if q <= 0:
                        continue
                    threshold = limit[k] - q
                    row = used[k]
                    t = start
                    while t < fe:
                        if row[t] > threshold:
                            blocked_at = t
                            break
                        t += 1
                    if blocked_at is not None:
                        break
                if blocked_at is None:
                    break

                start = blocked_at - d + 1 - 1
                if start < lower:
                    raise RuntimeError(
                        f"backward pass cannot place activity {idx}")
            starts[idx] = start
            for k in range(n_res):
                q = row_dem[k]
                if q <= 0:
                    continue
                row = used[k]
                for t in range(start, start + d):
                    row[t] += q
            if not inst.is_dummy[idx]:
                placed += 1
        return starts

    def _backward_dynamic(self, deadline: int, cap_at,
                          priority: "Sequence[float]",
                          starts_hint: "Optional[Sequence[int]]" = None) -> List[int]:

        inst = self.inst
        dur = inst.durations
        dem = inst.demands
        n_res = inst.num_resources
        cap_full = inst.resource_avail
        rel = inst.release
        succ = inst.successors
        n = len(dur)
        pred = self._reverse_arcs()

        horizon = max(deadline, self.horizon_bound)
        used = [[0] * horizon for _ in range(n_res)]
        starts = [0] * n
        done = [False] * n
        remaining_succ = [len(succ[i]) for i in range(n)]
        eligible = [i for i in range(n) if remaining_succ[i] == 0]
        placed = 0
        guard = 0

        while eligible:
            guard += 1
            if guard > n + 5:
                raise RuntimeError("dynamic backward pass deadlocked")
            idx = min(eligible, key=lambda a: (priority[a], a))
            eligible.remove(idx)
            d = dur[idx]
            upper_finish = deadline
            for j in succ[idx]:
                sj = starts[j]
                if sj < upper_finish:
                    upper_finish = sj
            if d <= 0:
                starts[idx] = upper_finish
                done[idx] = True
                for p in pred[idx]:
                    remaining_succ[p] -= 1
                    if remaining_succ[p] == 0:
                        eligible.append(p)
                continue
            row_dem = dem[idx]
            limit = cap_full if cap_at is None else cap_at(placed)
            lower = rel[idx]
            start = upper_finish - d
            if start < lower:
                start = lower
            while True:
                fe = start + d
                blocked_at = None
                for k in range(n_res):
                    q = row_dem[k]
                    if q <= 0:
                        continue
                    threshold = limit[k] - q
                    row = used[k]
                    t = start
                    while t < fe:
                        if row[t] > threshold:
                            blocked_at = t
                            break
                        t += 1
                    if blocked_at is not None:
                        break
                if blocked_at is None:
                    break
                start = blocked_at - d + 1 - 1
                if start < lower:
                    raise RuntimeError(
                        f"dynamic backward pass cannot place activity {idx}")
            starts[idx] = start
            done[idx] = True
            for k in range(n_res):
                q = row_dem[k]
                if q <= 0:
                    continue
                row = used[k]
                for t in range(start, start + d):
                    row[t] += q
            if not inst.is_dummy[idx]:
                placed += 1
            for p in pred[idx]:
                remaining_succ[p] -= 1
                if remaining_succ[p] == 0:
                    eligible.append(p)
        if not all(done):
            raise RuntimeError("dynamic backward pass left activities unscheduled")
        return starts

    def decode_policy(self, order: "Sequence[int]", policy: "DecodePolicy",
                      reference: "Optional[float]" = None,
                      room: "Optional[int]" = None):

        declared = policy.cost
        allowed = declared if room is None else max(1, min(declared, int(room)))

        fitness = self.decode(order)
        best_order = list(order)
        best_starts = list(self._starts)
        best_fitness = fitness
        cost = 1

        cur_order = list(order)
        cur_fitness = fitness
        cur_starts = list(self._starts)

        if reference is not None and policy.gate >= 0.0:
            if cur_fitness > (1.0 + policy.gate) * float(reference):
                self._starts = best_starts
                return best_fitness, cost, best_order, best_starts

        if policy.buffered and cost + 1 <= allowed:
            cap_at = self.buffer_profile(policy.beta, policy.tau)
            fb = self.decode(order, cap_at)
            cost += 1
            if fb <= cur_fitness:
                cur_fitness = fb
                cur_order = list(order)
                cur_starts = list(self._starts)
            if fb < best_fitness:
                best_fitness = fb
                best_order = list(order)
                best_starts = list(self._starts)

        rounds_left = max(0, (allowed - cost) // 2)
        for _ in range(min(max(0, int(policy.rounds)), rounds_left)):
            priority = None
            if policy.bw == 1:

                dur_local = self.inst.durations
                priority = [-(cur_starts[i] + dur_local[i]) for i in range(len(dur_local))]
            elif policy.bw == 2:
                priority = [float(cur_starts[i]) for i in range(len(cur_starts))]
            try:
                bstarts = self.backward_pass(cur_order, cur_fitness,
                                             priority=priority)
            except RuntimeError:

                break
            nxt_order = self._topo_order_from_starts(bstarts)
            nxt_fitness = self.decode(nxt_order)
            cost += 2
            if nxt_fitness < cur_fitness:
                cur_fitness = nxt_fitness
                cur_order = nxt_order
                cur_starts = list(self._starts)
                if nxt_fitness < best_fitness:
                    best_fitness = nxt_fitness
                    best_order = nxt_order
                    best_starts = list(self._starts)
            else:
                break

        self._starts = best_starts
        return best_fitness, cost, best_order, best_starts

    def evaluate_policy(self, order: "Sequence[int]", policy: "DecodePolicy",
                        reference: "Optional[float]" = None,
                        room: "Optional[int]" = None):

        t0 = time.perf_counter()
        fitness, cost, best_order, best_starts = self.decode_policy(
            order, policy, reference=reference, room=room)
        dt = time.perf_counter() - t0
        self.n_candidates += 1
        self.n_exact_sgs += cost
        self.decode_time_s += dt
        return EvalRecord(fitness, dt, True, cost), best_order, best_starts

    def evaluate(self, order: Sequence[int], *, count: bool = True) -> EvalRecord:

        self.n_candidates += 1
        if len(order) < 65536 and (len(order) == 0 or max(order) < 256):
            key = bytes(order)
        else:
            key = ",".join(map(str, order)).encode("ascii")
        cached = self._cache.get(key) if self._use_cache else None
        if cached is not None:
            self._cache.move_to_end(key)
            self.n_cache_hits += 1
            return EvalRecord(cached, 0.0, False)
        t0 = time.perf_counter()
        value = self.decode(order)
        dt = time.perf_counter() - t0
        self.n_exact_sgs += 1 if count else 0
        self.decode_time_s += dt
        if self._use_cache:
            self._cache[key] = value
            if len(self._cache) > self._cache_capacity:
                self._cache.popitem(last=False)
        return EvalRecord(value, dt, True)

    def bind_precedence(self, pred: Sequence[Sequence[int]]) -> None:
        self._pred = pred

    def bind_profile_changes(self, points: Sequence[int]) -> None:
        self._profile_change_points = sorted(set(points))

@dataclass
class OperatorSet:

    algorithm: str
    operators: Dict[str, str]
    parameters: Dict[str, float]

    def as_log(self) -> Tuple[str, str, str, str, str, str]:
        return (
            self.algorithm,
            self.operators.get("selection", ""),
            self.operators.get("crossover", ""),
            self.operators.get("mutation", ""),
            self.operators.get("local_search", ""),
            ";".join(f"{k}={v:g}" for k, v in sorted(self.parameters.items())),
        )

@dataclass(frozen=True)
class DecodePolicy:

    rounds: int = 0
    beta: float = 1.0
    tau: float = 0.0
    gate: float = -1.0

    bw: int = 0

    @property
    def buffered(self) -> bool:
        return self.beta < 1.0 and self.tau > 0.0

    @property
    def name(self) -> str:
        if self.rounds > 0:
            return "buffered_realign" if self.beta < 1.0 else "realign"
        return "buffered" if self.beta < 1.0 else "serial"

    @property
    def cost(self) -> int:

        buffered = 1 if (self.beta < 1.0 and self.tau > 0.0) else 0
        return 1 + buffered + 2 * max(0, int(self.rounds))

def make_decode_policy(op_name: str, params: Dict[str, float]) -> DecodePolicy:

    gate = float(params.get("gate", -1.0))
    bw = int(params.get("bw", 0))
    if op_name == "serial":
        return DecodePolicy(0, 1.0, 0.0, -1.0, bw)
    if op_name == "buffered":
        return DecodePolicy(0, float(params.get("beta", 0.7)),
                            float(params.get("tau", 0.4)), gate, bw)
    if op_name == "realign":
        return DecodePolicy(int(params.get("rounds", 1)), 1.0, 0.0, gate, bw)
    if op_name == "buffered_realign":
        return DecodePolicy(int(params.get("rounds", 1)),
                            float(params.get("beta", 0.7)),
                            float(params.get("tau", 0.4)), gate, bw)
    raise KeyError(f"unknown decoding policy {op_name!r}")

def _tournament(pop_index: Sequence[int], fitness: Sequence[float], rng, size=3) -> int:
    best = rng.choice(list(pop_index))
    for _ in range(size - 1):
        cand = rng.choice(list(pop_index))
        if fitness[cand] < fitness[best]:
            best = cand
    return best

def _roulette(pop_index: Sequence[int], fitness: Sequence[float], rng) -> int:
    best = min(fitness[i] for i in pop_index)
    weights = [best / max(1e-9, fitness[i]) for i in pop_index]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in zip(pop_index, weights):
        acc += w
        if acc >= r:
            return i
    return pop_index[-1]

def crossover_sbx(p1: np.ndarray, p2: np.ndarray, cr: float, rng) -> Tuple[np.ndarray, np.ndarray]:
    eta = 20.0
    y1, y2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if rng.random() <= cr:
            if abs(p1[i] - p2[i]) > 1e-14:
                u = rng.random()
                beta = (2 * u) ** (1.0 / (eta + 1)) if u <= 0.5 else (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                y1[i] = np.clip(0.5 * ((p1[i] + p2[i]) - beta * (p2[i] - p1[i])), 0.0, 1.0)
                y2[i] = np.clip(0.5 * ((p1[i] + p2[i]) + beta * (p2[i] - p1[i])), 0.0, 1.0)
    return y1, y2

def crossover_two_point(p1: np.ndarray, p2: np.ndarray, cr: float, rng) -> Tuple[np.ndarray, np.ndarray]:
    n = len(p1)
    if rng.random() > cr or n < 2:
        return p1.copy(), p2.copy()
    cuts = np.sort(rng.choice(n, size=2, replace=False))
    a, b = int(cuts[0]), int(cuts[1])
    y1, y2 = p1.copy(), p2.copy()
    y1[a:b], y2[a:b] = p2[a:b].copy(), p1[a:b].copy()
    return y1, y2

def crossover_uniform(p1: np.ndarray, p2: np.ndarray, cr: float, rng) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.array([rng.random() <= cr for _ in range(len(p1))])
    y1, y2 = p1.copy(), p2.copy()
    y1[mask] = p2[mask]
    y2[mask] = p1[mask]
    return y1, y2

def mutate_polynomial(x: np.ndarray, pm: float, rng) -> np.ndarray:
    eta = 20.0
    y = x.copy()
    for i in range(len(y)):
        if rng.random() <= pm:
            u = rng.random()
            d1 = y[i]
            d2 = 1.0 - y[i]
            if u <= 0.5:
                dq = (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** (1.0 / (eta + 1)) - 1
            else:
                dq = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)) ** (1.0 / (eta + 1))
            y[i] = float(np.clip(y[i] + dq, 0.0, 1.0))
    return y

def mutate_gaussian(x: np.ndarray, pm: float, rng, sigma: float = 0.1) -> np.ndarray:
    y = x.copy()
    for i in range(len(y)):
        if rng.random() <= pm:
            y[i] = float(np.clip(y[i] + rng.normal(0.0, sigma), 0.0, 1.0))
    return y

def mutate_reset(x: np.ndarray, pm: float, rng) -> np.ndarray:
    y = x.copy()
    for i in range(len(y)):
        if rng.random() <= pm:
            y[i] = rng.random()
    return y

CROSSOVER_TABLE = {
    "sbx": crossover_sbx,
    "two_point": crossover_two_point,
    "uniform": crossover_uniform,
}
MUTATION_TABLE = {
    "polynomial": mutate_polynomial,
    "gaussian": mutate_gaussian,
    "reset": mutate_reset,
}
SELECTION_TABLE = {"tournament": _tournament, "roulette": _roulette}

def random_keys_to_order(inst: Instance, keys: np.ndarray) -> List[int]:

    n = len(keys)
    indeg = [0] * n
    for i in range(n):
        for j in inst.successors[i]:
            indeg[j] += 1
    ready = [i for i in range(n) if indeg[i] == 0]
    order: List[int] = []
    while ready:
        ready.sort(key=lambda i: (-keys[i], i))
        node = ready.pop(0)
        order.append(node)
        for j in inst.successors[node]:
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
    if len(order) != n:
        raise RuntimeError("topological sort failed: graph contains a cycle")
    return order

def precedence_lists(inst: Instance) -> List[List[int]]:
    n = len(inst.durations)
    pred = [[] for _ in range(n)]
    for i in range(n):
        for j in inst.successors[i]:
            pred[j].append(i)
    return pred

def profile_change_points(inst: Instance) -> List[int]:

    n = len(inst.durations)

    pred = precedence_lists(inst)
    es = [0] * n
    for i in range(n):
        if pred[i]:
            es[i] = max(es[p] + inst.durations[p] for p in pred[i])
        es[i] = max(es[i], inst.release[i])
    pts = {0}
    for i in range(n):
        pts.add(es[i])
        pts.add(es[i] + inst.durations[i])
    return sorted(pts)

class SurrogateWrapper:

    def __init__(self, model_type: str, n_features: int, params: Optional[dict] = None):
        import sys

        from experiment_config import CODE_DIR

        if str(CODE_DIR) not in sys.path:
            sys.path.insert(0, str(CODE_DIR))
        from surrate_net import create_surrogate_model

        params = dict(params or {})
        self.model_type = model_type
        self.model = create_surrogate_model(model_type, n_features, "cpu", **params)
        self.fit_time_s = 0.0
        self.predict_time_s = 0.0
        self.n_fits = 0
        self.n_predictions = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) < 3:
            return
        t0 = time.perf_counter()
        if self.model_type == "dnn":
            self.model.train(X, y, 10)
        else:
            self.model.train(X, y)
        self.fit_time_s += time.perf_counter() - t0
        self.n_fits += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.array([])
        t0 = time.perf_counter()
        out = self.model.predict(X)
        self.predict_time_s += time.perf_counter() - t0
        self.n_predictions += len(X)
        return out

@dataclass
class RunOutcome:
    best_tpm: int
    cmax: int
    best_order: List[int]
    start: List[int]
    finish: List[int]
    n_exact_sgs: int
    n_candidates: int
    wall_time_s: float
    cpu_time_s: float
    decode_time_s: float
    surrogate_fit_time_s: float
    surrogate_predict_time_s: float
    policy_time_s: float
    io_time_s: float
    n_surrogate_retrains: int
    n_surrogate_predictions: int
    algorithm_selection_counts: Dict[str, int]
    operator_selection_counts: Dict[str, int]
    algorithm_operator_success_stats: Dict[str, dict]
    best_found_generation: int
    best_found_exact_eval: int
    valid_schedule: bool
    validation: dict
    convergence: List[dict]
    use_surrogate: bool
    use_meta_adaptation: bool

    budget_k_min: int = 0
    budget_k_max: int = 0
    budget_target_exact: int = 0

    n_cache_hits: int = 0

def build_policy(device: str = "cpu"):

    import sys

    from experiment_config import CODE_DIR

    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    from meta_hyper_policy import MetaHyperPolicy

    return MetaHyperPolicy().to(device)
