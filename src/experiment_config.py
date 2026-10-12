"""Final algorithm settings and benchmark definitions."""
from __future__ import annotations

import os
from pathlib import Path

EXPERIMENT_ID = "SAMA_EA"

EXPERIMENTS_DIR = Path(__file__).resolve().parent
BASE_DIR = EXPERIMENTS_DIR.parent
CODE_DIR = BASE_DIR.parent
PROJECT_DIR = CODE_DIR.parent

def _resolve_code_dir() -> Path:

    here = Path(__file__).resolve().parent
    for _ in range(6):
        if (here / "data" / "dataset" / "JSON").is_dir():
            return here
        if here.parent == here:
            break
        here = here.parent
    raise RuntimeError(
        "cannot locate code/data/dataset/JSON by walking up from "
        f"{Path(__file__).resolve().parent}"
    )

CODE_DIR = _resolve_code_dir()
JSON_ROOT = CODE_DIR / "data" / "dataset" / "JSON"

RESULTS_DIR = BASE_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw"
BASELINE_REF_DIR = RESULTS_DIR / "baseline_reference"
VERSION_DIAG_DIR = RESULTS_DIR / "version_diagnostics"
AGGREGATE_DIR = RESULTS_DIR / "aggregate"
FIGURES_DIR = RESULTS_DIR / "figures"
FAILURES_DIR = RESULTS_DIR / "failures"
MANIFEST_DIR = RESULTS_DIR / "manifest"
SMOKE_DIR = BASE_DIR / "smoke"

DEVELOPMENT_INSTANCES = (
    "mpj60a2", "mpj60a4", "MPLIB1_Set1_1", "MPLIB2_Set1_1",
)

RCMPSPLIB_INSTANCES = (
    "mpj30a2", "mpj30a4", "mpj30a6", "mpj30a10",
    "mpj60a3", "mpj60a5", "mpj90a3", "mpj90a4",
    "mpj120a2", "mpj120a3", "HHH+3", "MP-MD4",
)
MPLIB1_INSTANCES = tuple(f"MPLIB1_Set1_{i}" for i in range(2, 8))
MPLIB2_INSTANCES = tuple(f"MPLIB2_Set1_{i}" for i in range(2, 8))

VALIDATION_INSTANCES = RCMPSPLIB_INSTANCES + MPLIB1_INSTANCES + MPLIB2_INSTANCES

INSTANCE_GROUP = {}
for _n in RCMPSPLIB_INSTANCES:
    INSTANCE_GROUP[_n] = "RCMPSPLIB"
for _n in MPLIB1_INSTANCES:
    INSTANCE_GROUP[_n] = "MPLIB1"
for _n in MPLIB2_INSTANCES:
    INSTANCE_GROUP[_n] = "MPLIB2"

if len(VALIDATION_INSTANCES) != 24:
    raise AssertionError(f"expected 24 comparison instances, got {len(VALIDATION_INSTANCES)}")

N_RUNS = 30
SEEDS = tuple(range(1, N_RUNS + 1))

POP_SIZE = 50
N_GEN = 200

BASELINE_ALGORITHMS = ("SADE", "QLESCA", "B23_GA_CB", "PSGS_MINSLK")

METHOD_SAMA = "SAMA_EA"
METHOD_MA = "MA_EA"

METHOD_FAFO = "FAFO_EA"

ALL_ALGORITHM_LABELS = BASELINE_ALGORITHMS + (
    METHOD_SAMA, METHOD_MA, METHOD_FAFO,
)

SURROGATE_TYPE = "xgb"
SURROGATE_PARAMS = {
    "n_estimators": 50,
    "max_depth": 4,
    "learning_rate": 0.05,
}

SURROGATE_RETRAIN_EVERY = 20

SAMA_EXACT_FRACTION = float(os.environ.get("SAMA_EXACT_FRACTION", "0.60"))

SURROGATE_POOL_FACTOR = float(os.environ.get("SAMA_POOL_FACTOR", "1.0"))

SURROGATE_FEATURE_MODE = int(os.environ.get("SAMA_SUR_FEAT", "1"))

SURROGATE_ARCHIVE_CAP = int(os.environ.get("SAMA_SUR_ARCHIVE", "1500"))

SURROGATE_EXPLORE = float(os.environ.get("SAMA_SUR_EPS", "0.15"))

SURROGATE_DIAG = int(os.environ.get("SAMA_SUR_DIAG", "0"))

SURROGATE_RENEW_POOL = int(os.environ.get("SAMA_SUR_RENPOOL", "8"))

BUDGET_K_MIN = 5
BUDGET_K_MAX = 50
BUDGET_PARITY_MODE = "calibrate"
REAL_MAX = BUDGET_K_MAX
REAL_MIN = BUDGET_K_MIN
BUDGET_DECAY = "adaptive"
STAGNATION_SCALE = 50.0

BUDGET_K_HARD_CAP = 4 * BUDGET_K_MAX

EVALUATOR_USE_CACHE = False

def calibrated_k_bounds(pop_size: int = None,
                        generations: int = None,
                        decay: str = None,
                        target: int = None) -> tuple:

    import math

    if pop_size is None:
        pop_size = POP_SIZE
    if generations is None:
        generations = N_GEN
    if decay is None:
        decay = BUDGET_DECAY

    if target is None:
        target = pop_size * generations

    shapes = []
    for gen in range(generations):
        if decay == "linear":
            shapes.append(gen / (generations - 1))
        elif decay == "exp":
            shapes.append((gen / (generations - 1)) ** 2)
        else:
            progress = gen / (generations - 1)
            shapes.append(1.0 - math.exp(-progress))

    s = sum(shapes)
    if generations - s <= 1e-9:
        k_max = target / max(1, generations)
    else:
        k_max = (target - BUDGET_K_MIN * s) / (generations - s)
    k_max = int(math.ceil(k_max))
    k_max = max(BUDGET_K_MIN, min(BUDGET_K_HARD_CAP, k_max))
    return BUDGET_K_MIN, k_max

META_MODE = os.environ.get("SAMA_META_MODE", "joint").strip().lower()
if META_MODE not in ("off", "algorithm", "operator", "joint"):
    META_MODE = "joint"

DE_JITTER = float(os.environ.get("SAMA_DE_JITTER", "0.0"))

DE_MUT_P = float(os.environ.get("SAMA_DE_MUTP", "0.05"))

ALG_PRIOR_BIAS = float(os.environ.get("SAMA_ALG_PRIOR", "0.0"))

META_ACTION_SLOTS = int(os.environ.get("SAMA_SLOTS", "4"))
if META_ACTION_SLOTS < 1:
    META_ACTION_SLOTS = 1

ABLATION_FIXED_GA_PARAM = {"CR": 0.8, "PM": 0.05}
ABLATION_FIXED_DE_PARAM = {"F": 0.7, "CR": 0.5}

LR_INNER = 1e-4
POLICY_TEMPERATURE = 0.5
ENTROPY_COEF = 0.5

POLICY_STATE_DIM = 9
POLICY_HIDDEN = 32

EXPLOIT_STRENGTH = 1.5

EXPLOIT_MODE = os.environ.get("SAMA_EXPLOIT_MODE", "reward").strip().lower()
if EXPLOIT_MODE not in ("success", "reward"):
    EXPLOIT_MODE = "reward"

REWARD_W_BEST = 1.0
REWARD_W_MEAN = 1.0

GA_SELECTION_OPERATORS = ("tournament", "roulette")
GA_CROSSOVER_OPERATORS = ("sbx", "two_point", "uniform")
GA_MUTATION_OPERATORS = ("polynomial", "gaussian", "reset")
GA_CR_CAND = (0.6, 0.7, 0.8, 0.9, 0.95)
GA_PM_CAND = (0.005, 0.02, 0.05, 0.08, 0.10)

DE_MUTATION_STRATEGIES = ("rand1", "best1", "current_to_best1")
DE_CROSSOVER_OPERATORS = ("binomial", "exponential")
DE_SELECTION_OPERATORS = ("greedy",)
DE_F_CAND = (0.3, 0.5, 0.7, 0.9, 1.0)
DE_CR_CAND = (0.1, 0.3, 0.5, 0.7, 0.9)

OPERATOR_HEAD_WIDTH = 18

SCHEDULE_POLICY_OPERATORS = ("serial", "buffered", "realign", "buffered_realign")

SCHEDULE_POLICY_PARAMS = (
    (0.60, 0.30, 1, -1.0, 1),
    (0.75, 0.50, 1, -1.0, 1),
    (0.90, 0.50, 2, -1.0, 1),
    (0.45, 0.20, 1, -1.0, 1),
    (1.00, 0.00, 2, -1.0, 1),
)

SCHEDULE_POLICY_MODE = os.environ.get("SAMA_SCHED_MODE", "adaptive")
FIXED_SCHEDULE_POLICY = os.environ.get("SAMA_SCHED_FIXED", "")
FIXED_SCHEDULE_PARAM_INDEX = int(os.environ.get("SAMA_SCHED_PARAM_IDX", "1"))

SCHEDULE_POLICY_POOL = tuple(
    s.strip() for s in os.environ.get("SAMA_SCHED_POOL", "realign").split(",")
    if s.strip())

POLISH_INITIAL = os.environ.get("SAMA_POLISH_INITIAL", "0") not in ("0", "no", "false")

LAMARCKIAN_DECODE = os.environ.get("SAMA_LAMARCKIAN", "1") not in ("0", "no", "false")

DECODER_REKEY = os.environ.get("SAMA_REKEY", "none").strip().lower()
if DECODER_REKEY not in ("none", "order", "schedule"):
    DECODER_REKEY = "none"

RENEW_ENABLED = os.environ.get("SAMA_RENEW", "1") not in ("0", "no", "false")

RENEW_DIVERSITY_FLOOR = float(os.environ.get("SAMA_RENEW_DIV", "0.02"))
RENEW_STAGNATION_MIN = int(os.environ.get("SAMA_RENEW_STAG", "8"))

RENEW_COOLDOWN = int(os.environ.get("SAMA_RENEW_COOL", "8"))

RENEW_BUDGET_FRACTION = float(os.environ.get("SAMA_RENEW_BUDGET", "0.50"))

RENEW_REPLACE_FRACTION = float(os.environ.get("SAMA_RENEW_REPLACE", "0.40"))

RENEW_MODE = int(os.environ.get("SAMA_RENEW_MODE", "1"))
RENEW_SIGMA = float(os.environ.get("SAMA_RENEW_SIGMA", "0.03"))

RENEW_PERTURB = (float(os.environ.get("SAMA_RENEW_PMIN", "0.03")),
                 float(os.environ.get("SAMA_RENEW_PMAX", "0.15")))

RENEW_ELITE_PROB = float(os.environ.get("SAMA_RENEW_ELITE", "0.60"))

INIT_RULE_MODE = os.environ.get("SAMA_INIT", "lft").strip().lower()
if INIT_RULE_MODE not in ("lft", "portfolio"):
    INIT_RULE_MODE = "lft"

PARENT_FRACTION = float(os.environ.get("SAMA_PARENT_FRACTION", "0.50"))

PARENT_FRACTION_MODE = os.environ.get("SAMA_PARENT_MODE", "fixed").strip().lower()
if PARENT_FRACTION_MODE not in ("fixed", "adaptive"):
    PARENT_FRACTION_MODE = "fixed"
PARENT_FRACTION_MIN = float(os.environ.get("SAMA_PARENT_MIN", "0.20"))
PARENT_FRACTION_MAX = float(os.environ.get("SAMA_PARENT_MAX", "0.60"))

PARENT_DIVERSITY_HI = float(os.environ.get("SAMA_PARENT_DIV_HI", "0.04"))
PARENT_DIVERSITY_LO = float(os.environ.get("SAMA_PARENT_DIV_LO", "0.01"))
INIT_RULE_COMBOS = (
    ("flat", "lft"), ("flat", "slk"), ("flat", "est"), ("flat", "work"),
    ("work", "lft"), ("work", "slk"), ("cnt", "lft"), ("dur", "lft"),
    ("rel", "lft"), ("rand", "work"), ("flat", "nsucc"), ("flat", "rand"),
)


MAX_WORKERS = 30
INSTANCES_PER_BATCH = 3
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def apply_thread_env() -> None:

    for key, value in SINGLE_THREAD_ENV.items():
        os.environ.setdefault(key, value)

def dev_instance_json_path(instance: str) -> Path:

    if instance not in DEVELOPMENT_INSTANCES:
        raise KeyError(
            f"{instance!r} is not one of the four development instances "
            f"{DEVELOPMENT_INSTANCES}")
    target = f"{instance}.json"
    for path in sorted(JSON_ROOT.rglob(target)):
        if path.is_file():
            return path
    raise FileNotFoundError(f"cannot locate {target} below {JSON_ROOT}")

def instance_json_path(instance: str) -> Path:

    if instance not in INSTANCE_GROUP:
        raise KeyError(f"{instance!r} is not part of the 24-instance comparison set")
    direct = JSON_ROOT / INSTANCE_GROUP[instance] / f"{instance}.json"
    if direct.is_file():
        return direct
    target = f"{instance}.json"
    for path in JSON_ROOT.rglob(target):
        if path.is_file():
            return path
    raise FileNotFoundError(f"cannot locate {target} below {JSON_ROOT}")

def iter_instance_json_files():
    for name in VALIDATION_INSTANCES:
        try:
            yield name, instance_json_path(name)
        except FileNotFoundError:
            continue

def raw_run_dir(algorithm_label: str, instance: str, seed: int) -> Path:
    return RAW_DIR / algorithm_label / instance / f"seed_{seed:03d}"

def baseline_algorithm_dir(algorithm_label: str) -> Path:
    return RAW_DIR / algorithm_label

def sama_version_dir(method: str, version: str) -> Path:
    return RAW_DIR / method / version

def version_tag(method: str, index: int) -> str:
    return f"{method}_r{index:03d}"

SAMA_EA = METHOD_SAMA
