"""Population search, verified action credit and selective evaluation."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluator import (
    CROSSOVER_TABLE, DecodePolicy, Evaluator, MUTATION_TABLE, SELECTION_TABLE,
    OperatorSet, RunOutcome, SurrogateWrapper, make_decode_policy,
    precedence_lists, profile_change_points, random_keys_to_order,
)
from experiment_config import (
    ABLATION_FIXED_DE_PARAM, ABLATION_FIXED_GA_PARAM,
    BUDGET_DECAY, BUDGET_K_MAX, BUDGET_K_MIN, BUDGET_PARITY_MODE,
    DECODER_REKEY, META_MODE,
    INIT_RULE_COMBOS, INIT_RULE_MODE,
    DE_CR_CAND, DE_CROSSOVER_OPERATORS, ENTROPY_COEF, EVALUATOR_USE_CACHE,
    EXPLOIT_MODE, EXPLOIT_STRENGTH, FIXED_SCHEDULE_PARAM_INDEX,
    FIXED_SCHEDULE_POLICY,
    ALG_PRIOR_BIAS, LAMARCKIAN_DECODE, LR_INNER, META_ACTION_SLOTS,
    OPERATOR_HEAD_WIDTH,
    PARENT_FRACTION,
    PARENT_DIVERSITY_HI, PARENT_DIVERSITY_LO, PARENT_FRACTION_MAX,
    PARENT_FRACTION_MIN, PARENT_FRACTION_MODE,
    REWARD_W_BEST, REWARD_W_MEAN, SAMA_EXACT_FRACTION, SURROGATE_ARCHIVE_CAP,
    SURROGATE_DIAG, SURROGATE_EXPLORE, SURROGATE_FEATURE_MODE,
    SURROGATE_POOL_FACTOR, SURROGATE_RENEW_POOL,
    DE_F_CAND, DE_JITTER, DE_MUT_P, DE_MUTATION_STRATEGIES,
    DE_SELECTION_OPERATORS,
    GA_CR_CAND, GA_CROSSOVER_OPERATORS, GA_MUTATION_OPERATORS, GA_PM_CAND,
    GA_SELECTION_OPERATORS, N_GEN, POLICY_HIDDEN, POLICY_STATE_DIM,
    POLICY_TEMPERATURE, POLISH_INITIAL, POP_SIZE, SCHEDULE_POLICY_MODE,
    SCHEDULE_POLICY_OPERATORS, SCHEDULE_POLICY_PARAMS, SCHEDULE_POLICY_POOL, STAGNATION_SCALE, SURROGATE_PARAMS,
    SURROGATE_RETRAIN_EVERY, SURROGATE_TYPE,
    RENEW_BUDGET_FRACTION, RENEW_COOLDOWN, RENEW_DIVERSITY_FLOOR, RENEW_ELITE_PROB,
    RENEW_ENABLED, RENEW_MODE, RENEW_PERTURB, RENEW_REPLACE_FRACTION,
    RENEW_SIGMA, RENEW_STAGNATION_MIN,
)
from meta_policy import MetaPolicyNet, build_policy
from problem_loader import Instance, validate_schedule
from seeding import set_seed

CONVERGENCE_FIELDS = [
    "generation", "elapsed_wall_s", "elapsed_cpu_s", "best_verified_tpm",
    "generation_best_verified_tpm", "mean_verified_tpm", "population_diversity",
    "n_candidates_cum", "n_exact_sgs_cum", "n_surrogate_predictions_cum",
    "surrogate_error_or_rank_metric", "selected_algorithm",
    "selected_selection_operator", "selected_crossover_operator",
    "selected_mutation_operator", "selected_local_search_operator",
    "selected_schedule_operator", "schedule_policy_cost",
    "selected_operator_parameters", "algorithm_operator_reward",
    "algorithm_operator_success", "exact_budget_k", "stagnation_length",
    "surrogate_rank_corr", "surrogate_recall_at_k", "surrogate_batch_regret",
    "surrogate_rho_all", "surrogate_k",
]

@dataclass(frozen=True)
class DecoderConfig:

    mode: str = SCHEDULE_POLICY_MODE
    fixed_name: str = FIXED_SCHEDULE_POLICY or "serial"
    param_index: int = FIXED_SCHEDULE_PARAM_INDEX
    lamarckian: bool = LAMARCKIAN_DECODE

    rekey: str = DECODER_REKEY

    pool: Tuple[str, ...] = SCHEDULE_POLICY_POOL

    polish_initial: bool = POLISH_INITIAL

    override_params: Optional[Dict[str, float]] = None

    init: str = INIT_RULE_MODE
    init_combos: Tuple[Tuple[str, str], ...] = INIT_RULE_COMBOS

    meta_mode: str = META_MODE

    exploit_mode: str = EXPLOIT_MODE
    lr: float = LR_INNER

    temperature: float = POLICY_TEMPERATURE
    entropy_coef: float = ENTROPY_COEF

    exploit: float = EXPLOIT_STRENGTH

    alg_prior: float = ALG_PRIOR_BIAS

    de_jitter: float = DE_JITTER

    de_mut_p: float = DE_MUT_P

    slots: int = META_ACTION_SLOTS

    fixed_alg: str = "GA"

    @property
    def operators(self) -> Tuple[str, ...]:
        return tuple(self.pool) if self.pool else tuple(SCHEDULE_POLICY_OPERATORS)

    @classmethod
    def from_env(cls) -> "DecoderConfig":
        return cls()

    @classmethod
    def adaptive(cls, lamarckian: bool = True, rekey: str = "order",
                 pool: Tuple[str, ...] = (), polish_initial: bool = False,
                 override_params: Optional[Dict[str, float]] = None,
                 init: str = "", init_combos: Tuple[Tuple[str, str], ...] = (),
                 meta_mode: str = "", exploit_mode: str = "", lr: float = None,
                 temperature: float = None, exploit: float = None,
                 fixed_alg: str = "", slots: int = 0, alg_prior: float = None,
                 de_jitter: float = None, de_mut_p: float = None,
                 entropy_coef: float = None
                 ) -> "DecoderConfig":
        return cls(mode="adaptive", lamarckian=lamarckian, rekey=rekey,
                   pool=pool, polish_initial=polish_initial,
                   override_params=override_params,
                   init=init or INIT_RULE_MODE,
                   init_combos=init_combos or INIT_RULE_COMBOS,
                   meta_mode=meta_mode or META_MODE,
                   exploit_mode=exploit_mode or EXPLOIT_MODE,

                   lr=LR_INNER if lr is None else lr,
                   temperature=(POLICY_TEMPERATURE if temperature is None
                                else temperature),
                   exploit=EXPLOIT_STRENGTH if exploit is None else exploit,
                   fixed_alg=(fixed_alg or "GA").upper(),
                   slots=slots or META_ACTION_SLOTS,
                   alg_prior=ALG_PRIOR_BIAS if alg_prior is None else alg_prior,
                   de_jitter=DE_JITTER if de_jitter is None else de_jitter,
                   de_mut_p=DE_MUT_P if de_mut_p is None else de_mut_p,
                   entropy_coef=ENTROPY_COEF if entropy_coef is None else entropy_coef)

    @classmethod
    def pinned(cls, name: str, param_index: int = 1,
               lamarckian: bool = True, rekey: str = "order",
               pool: Tuple[str, ...] = (), polish_initial: bool = False,
               override_params: Optional[Dict[str, float]] = None,
                 init: str = "", init_combos: Tuple[Tuple[str, str], ...] = (),
                 meta_mode: str = "", exploit_mode: str = "", lr: float = None,
                 temperature: float = None, exploit: float = None,
                 fixed_alg: str = "", slots: int = 0, alg_prior: float = None,
                 de_jitter: float = None, de_mut_p: float = None,
                 entropy_coef: float = None
                 ) -> "DecoderConfig":
        return cls(mode="fixed", fixed_name=name, param_index=param_index,
                   lamarckian=lamarckian, rekey=rekey, pool=pool,
                   polish_initial=polish_initial,
                   override_params=override_params,
                   init=init or INIT_RULE_MODE,
                   init_combos=init_combos or INIT_RULE_COMBOS,
                   meta_mode=meta_mode or META_MODE,
                   exploit_mode=exploit_mode or EXPLOIT_MODE,

                   lr=LR_INNER if lr is None else lr,
                   temperature=(POLICY_TEMPERATURE if temperature is None
                                else temperature),
                   exploit=EXPLOIT_STRENGTH if exploit is None else exploit,
                   fixed_alg=(fixed_alg or "GA").upper(),
                   slots=slots or META_ACTION_SLOTS,
                   alg_prior=ALG_PRIOR_BIAS if alg_prior is None else alg_prior,
                   de_jitter=DE_JITTER if de_jitter is None else de_jitter,
                   de_mut_p=DE_MUT_P if de_mut_p is None else de_mut_p,
                   entropy_coef=ENTROPY_COEF if entropy_coef is None else entropy_coef)

    def policy_values(self) -> Dict[str, float]:

        beta_ref, tau_ref, rounds_ref, gate_ref, bw_ref = SCHEDULE_POLICY_PARAMS[
            self.param_index % len(SCHEDULE_POLICY_PARAMS)]
        vals = {"beta": beta_ref, "tau": tau_ref, "rounds": rounds_ref,
                "gate": gate_ref, "bw": bw_ref}
        if self.override_params:
            vals.update({k: float(v) for k, v in self.override_params.items()})
        return vals

GA_OP_COUNT = (len(GA_SELECTION_OPERATORS) * len(GA_CROSSOVER_OPERATORS)
               * len(GA_MUTATION_OPERATORS))
DE_OP_COUNT = (len(DE_MUTATION_STRATEGIES) * len(DE_CROSSOVER_OPERATORS)
               * len(DE_SELECTION_OPERATORS))

class MetaController:

    def __init__(self, seed: int, temperature: float = POLICY_TEMPERATURE,
                 decoder: Optional[DecoderConfig] = None):
        import torch

        self.torch = torch
        set_seed(seed)
        self.decoder = decoder or DecoderConfig.from_env()

        self.meta_mode = getattr(self.decoder, "meta_mode", META_MODE) or META_MODE
        self.adapt_alg = self.meta_mode in ("algorithm", "joint")
        self.adapt_op = self.meta_mode in ("operator", "joint")

        self.alg_prior = float(getattr(self.decoder, "alg_prior", 0.0) or 0.0) \
            if self.adapt_alg else 0.0

        self.exploit_mode = getattr(self.decoder, "exploit_mode",
                                    EXPLOIT_MODE) or EXPLOIT_MODE
        _lr = getattr(self.decoder, "lr", None)
        self.lr = float(LR_INNER if _lr is None else _lr)
        self.entropy_coef = float(self.decoder.entropy_coef)
        if not np.isfinite(self.entropy_coef) or self.entropy_coef < 0:
            raise ValueError("entropy_coef must be finite and nonnegative")
        self.net = build_policy(
            "cpu",
            global_dim=POLICY_STATE_DIM,
            hidden=POLICY_HIDDEN,
            n_op=OPERATOR_HEAD_WIDTH,
            n_dec=len(self.decoder.operators),
            n_dec_par=len(SCHEDULE_POLICY_PARAMS),
        )
        assert OPERATOR_HEAD_WIDTH >= max(GA_OP_COUNT, DE_OP_COUNT), \
            "operator head must be wide enough for every legal triple"
        _temp = getattr(self.decoder, "temperature", None)
        self.temperature = float(POLICY_TEMPERATURE if _temp is None else _temp)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.select_count: Dict[str, int] = {}
        self.combo_count: Dict[str, int] = {}
        self.combo_success: Dict[str, int] = {}
        self.combo_reward: Dict[str, float] = {}
        self.logps: List = []

        self._reward_ema = 0.0
        self._reward_ema_alpha = 0.05
        self._update_count = 0

    def _laplace_rate(self, combo: str) -> float:
        calls = self.combo_count.get(combo, 0)
        succ = self.combo_success.get(combo, 0)

        return (succ + 1.0) / (calls + 2.0)

    def _mean_reward(self, combo: str) -> float:

        calls = self.combo_count.get(combo, 0)
        total = self.combo_reward.get(combo, 0.0)
        return total / (calls + 1.0)

    def _credit(self, combo: str) -> float:
        if self.exploit_mode == "reward":
            return self._mean_reward(combo)
        return self._laplace_rate(combo)

    def _factor_keys(self, algorithm: str, op_idx: int) -> Tuple[str, str]:
        if algorithm == "GA":
            nm = len(GA_MUTATION_OPERATORS)
            nc = len(GA_CROSSOVER_OPERATORS)
            cro = GA_CROSSOVER_OPERATORS[(op_idx // nm) % nc]
            mut = GA_MUTATION_OPERATORS[op_idx % nm]
        else:
            nm = len(DE_MUTATION_STRATEGIES)
            nc = len(DE_CROSSOVER_OPERATORS)
            mut = DE_MUTATION_STRATEGIES[op_idx % nm]
            cro = DE_CROSSOVER_OPERATORS[(op_idx // nm) % nc]
        return f"F|{algorithm}|CROSS|{cro}", f"F|{algorithm}|MUT|{mut}"

    def _factor_credit(self, algorithm: str, op_idx: int) -> float:
        kc, km = self._factor_keys(algorithm, op_idx)
        return self._mean_reward(kc) + self._mean_reward(km)

    def _exploit_logit_bias(self, algorithm: str, width: int):

        torch = self.torch
        _st = getattr(self.decoder, "exploit", None)
        strength = float(EXPLOIT_STRENGTH if _st is None else _st)
        if strength <= 0.0:
            return None

        rates = []
        if algorithm == "GA":
            for op_idx in range(GA_OP_COUNT):
                if self.exploit_mode == "factor":
                    rates.append(self._factor_credit(algorithm, op_idx))
                else:
                    cro = GA_CROSSOVER_OPERATORS[(op_idx // len(GA_MUTATION_OPERATORS))
                                                 % len(GA_CROSSOVER_OPERATORS)]
                    mut = GA_MUTATION_OPERATORS[op_idx % len(GA_MUTATION_OPERATORS)]
                    rates.append(self._credit(f"{algorithm}|{cro}|{mut}"))
        else:
            for op_idx in range(DE_OP_COUNT):
                if self.exploit_mode == "factor":
                    rates.append(self._factor_credit(algorithm, op_idx))
                else:
                    mut = DE_MUTATION_STRATEGIES[op_idx % len(DE_MUTATION_STRATEGIES)]
                    cro = DE_CROSSOVER_OPERATORS[
                        (op_idx // len(DE_MUTATION_STRATEGIES)) % len(DE_CROSSOVER_OPERATORS)]
                    rates.append(self._credit(f"{algorithm}|{cro}|{mut}"))

        bias = [0.0] * width
        for idx, value in enumerate(rates):
            if idx < width:
                bias[idx] = value
        b = torch.tensor(bias, dtype=torch.float32)

        mean = sum(bias) / len(rates)
        b = b - mean
        return b * strength

    def _exploit_decoder_bias(self):

        torch = self.torch
        _st = getattr(self.decoder, "exploit", None)
        strength = float(EXPLOIT_STRENGTH if _st is None else _st)
        if strength <= 0.0 or self.decoder.mode != "adaptive":
            return None
        rates = [self._credit(f"DEC|{name}") for name in self.decoder.operators]
        b = torch.tensor(rates, dtype=torch.float32)
        b = b - b.mean()
        return b * strength

    def _resolve_schedule_policy(self, dec_idx: int, dec_par_idx: int):

        dec = self.decoder
        if dec.mode == "fixed" and dec.fixed_name:
            name = dec.fixed_name
            vals = dec.policy_values()
        else:
            pool = dec.operators
            name = pool[dec_idx % len(pool)]
            b0, t0, r0, g0, w0 = SCHEDULE_POLICY_PARAMS[
                dec_par_idx % len(SCHEDULE_POLICY_PARAMS)]
            vals = {"beta": b0, "tau": t0, "rounds": r0, "gate": g0, "bw": w0}
        return (name, vals)

    def _forward(self, state):
        torch = self.torch
        h, logits_alg = self.net.forward(state)

        if self.alg_prior:
            logits_alg = logits_alg + torch.tensor(
                [self.alg_prior, 0.0], dtype=logits_alg.dtype)
        alg_dist = torch.distributions.Categorical(logits=logits_alg / self.temperature)
        return torch, h, alg_dist

    def sample(self, state) -> Tuple[OperatorSet, object]:
        torch, h, alg_dist = self._forward(state)
        alg_idx_t = alg_dist.sample()

        if not self.adapt_alg:
            alg_idx_t = torch.tensor(0)
        return self._action(torch, h, alg_dist, alg_idx_t)

    def sample_multi(self, state, k: int):

        torch, h, alg_dist = self._forward(state)
        if self.adapt_alg:
            idxs = alg_dist.sample([int(k)])
        else:
            idxs = torch.zeros(int(k), dtype=torch.long)
        return [self._action(torch, h, alg_dist, idxs[j]) for j in range(int(k))]

    def _action(self, torch, h, alg_dist, alg_idx_t):
        alg_idx = int(alg_idx_t.item())

        logits_op, logits_par, logits_dec, logits_dec_par = self.net.conditioned(h, alg_idx)
        algorithm_name = "GA" if alg_idx == 0 else "DE"

        legal = GA_OP_COUNT if alg_idx == 0 else DE_OP_COUNT
        width = logits_op.shape[-1]
        if legal < width:
            keep = torch.arange(width) < legal
            logits_op = logits_op.masked_fill(~keep, -1e9)

        exploit_bias = self._exploit_logit_bias(algorithm_name, width)
        if exploit_bias is not None:
            logits_op = logits_op + exploit_bias
        dec_bias = self._exploit_decoder_bias()
        if dec_bias is not None and dec_bias.numel() == logits_dec.numel():
            logits_dec = logits_dec + dec_bias

        op_dist = torch.distributions.Categorical(logits=logits_op / self.temperature)
        par_dist = torch.distributions.Categorical(logits=logits_par / self.temperature)
        dec_dist = torch.distributions.Categorical(logits=logits_dec / self.temperature)
        dec_par_dist = torch.distributions.Categorical(
            logits=logits_dec_par / self.temperature)

        op_idx = int(op_dist.sample().item()) if self.adapt_op else 0
        par_idx = int(par_dist.sample().item()) if self.adapt_op else 0
        dec_idx = int(dec_dist.sample().item())
        dec_par_idx = int(dec_par_dist.sample().item())

        if alg_idx == 0:
            algorithm = "GA"

            nm = len(GA_MUTATION_OPERATORS)
            nc = len(GA_CROSSOVER_OPERATORS)
            sel_i = op_idx // (nm * nc)
            cro_i = (op_idx // nm) % nc
            mut_i = op_idx % nm
            operators = {
                "selection": GA_SELECTION_OPERATORS[sel_i],
                "crossover": GA_CROSSOVER_OPERATORS[cro_i],
                "mutation": GA_MUTATION_OPERATORS[mut_i],
                "local_search": "none",
            }
            parameters = dict(ABLATION_FIXED_GA_PARAM) if not self.adapt_op else {
                "CR": GA_CR_CAND[par_idx],
                "PM": GA_PM_CAND[(par_idx * 2 + alg_idx) % len(GA_PM_CAND)],
            }
        else:
            algorithm = "DE"
            nm = len(DE_MUTATION_STRATEGIES)
            nc = len(DE_CROSSOVER_OPERATORS)
            mut_i = op_idx % nm
            cro_i = (op_idx // nm) % nc
            sel_i = (op_idx // (nm * nc)) % len(DE_SELECTION_OPERATORS)
            operators = {
                "selection": DE_SELECTION_OPERATORS[sel_i],
                "crossover": DE_CROSSOVER_OPERATORS[cro_i],
                "mutation": DE_MUTATION_STRATEGIES[mut_i],
                "local_search": "none",
            }
            parameters = dict(ABLATION_FIXED_DE_PARAM) if not self.adapt_op else {
                "F": DE_F_CAND[(par_idx * 3 + alg_idx) % len(DE_F_CAND)],
                "CR": DE_CR_CAND[par_idx],
            }
            parameters["JITTER"] = self.decoder.de_jitter
            parameters["MUTP"] = self.decoder.de_mut_p

        dec_name, dec_params = self._resolve_schedule_policy(dec_idx, dec_par_idx)
        operators["schedule"] = dec_name
        parameters.update({f"dec_{k}": v for k, v in dec_params.items()})

        logp = None
        if self.adapt_alg:
            logp = alg_dist.log_prob(alg_idx_t)
        if self.adapt_op:
            term = op_dist.log_prob(torch.tensor(op_idx)) \
                + par_dist.log_prob(torch.tensor(par_idx))
            logp = term if logp is None else logp + term
        if self.decoder.mode != "fixed":

            term = dec_dist.log_prob(torch.tensor(dec_idx)) \
                + dec_par_dist.log_prob(torch.tensor(dec_par_idx))
            logp = term if logp is None else logp + term
        if logp is None:
            logp = alg_dist.log_prob(alg_idx_t) * 0.0
        prob_alg = alg_dist.probs

        prob_hyp = torch.cat([op_dist.probs, par_dist.probs,
                              dec_dist.probs, dec_par_dist.probs], dim=-1)
        return OperatorSet(algorithm, operators, parameters), (logp, prob_alg, prob_hyp, alg_idx)

    def record(self, opset: OperatorSet, reward: float, improved: bool) -> None:
        combo = f"{opset.algorithm}|{opset.operators.get('crossover','')}|{opset.operators.get('mutation','')}"
        dec_combo = f"DEC|{opset.operators.get('schedule','serial')}"

        self.select_count[opset.algorithm] = self.select_count.get(opset.algorithm, 0) + 1
        factors = (f"F|{opset.algorithm}|CROSS|{opset.operators.get('crossover','')}",
                   f"F|{opset.algorithm}|MUT|{opset.operators.get('mutation','')}")
        for key in (combo, dec_combo) + factors:
            self.combo_count[key] = self.combo_count.get(key, 0) + 1
            self.combo_reward[key] = self.combo_reward.get(key, 0.0) + reward
            if improved:
                self.combo_success[key] = self.combo_success.get(key, 0) + 1

    def update(self, logp, reward: float, prob_alg, prob_hyp) -> None:

        torch = self.torch
        entropy = -(prob_alg * torch.log(prob_alg + 1e-8)).sum() \
            - (prob_hyp * torch.log(prob_hyp + 1e-8)).sum()

        advantage = float(reward) - self._reward_ema
        self._reward_ema = (1 - self._reward_ema_alpha) * self._reward_ema \
            + self._reward_ema_alpha * float(reward)

        loss = -logp * advantage - self.entropy_coef * entropy
        self.opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self._update_count += 1

    def update_multi(self, items) -> None:

        if not items:
            return
        torch = self.torch
        rewards = [float(r) for _, r, _, _ in items]
        base = sum(rewards) / len(rewards)
        loss = None
        for logp, reward, prob_alg, prob_hyp in items:
            entropy = -(prob_alg * torch.log(prob_alg + 1e-8)).sum() \
                - (prob_hyp * torch.log(prob_hyp + 1e-8)).sum()
            advantage = float(reward) - base
            term = -logp * advantage - self.entropy_coef * entropy / len(items)
            loss = term if loss is None else loss + term
        self._reward_ema = (1 - self._reward_ema_alpha) * self._reward_ema \
            + self._reward_ema_alpha * base
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self._update_count += 1

    def snapshot(self) -> dict:
        return {
            "algorithm_selection_counts": dict(self.select_count),
            "operator_selection_counts": dict(self.combo_count),
            "algorithm_operator_success_stats": {
                combo: {
                    "calls": self.combo_count.get(combo, 0),
                    "successes": self.combo_success.get(combo, 0),
                    "cumulative_reward": round(self.combo_reward.get(combo, 0.0), 4),
                }
                for combo in sorted(self.combo_count)
            },
        }

def _state_vector(evaluator: Evaluator, pop: Sequence[dict], initial_best: float, gen: int,
                  stagnation: int, surrogate_err: float,
                  n_gen: int = N_GEN) -> np.ndarray:
    fits = np.array([ind["fitness"] for ind in pop], dtype=np.float64)
    scale = 1000.0
    best, worst, mean = float(fits.min()), float(fits.max()), float(fits.mean())
    std = float(fits.std(ddof=1)) if len(fits) > 1 else 0.0
    rel = (best - initial_best) / initial_best if initial_best else 0.0
    n = len(fits)
    if n >= 4 and std > 1e-9:
        skew = float((n / ((n - 1) * (n - 2))) * (np.sum((fits - mean) ** 3) / std ** 3))
        kurt = float((n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * (np.sum((fits - mean) ** 4) / std ** 4)
                     - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3)))
    else:
        skew, kurt = 0.0, 3.0

    return np.array([
        best / scale, worst / scale, mean / scale, std / scale,
        rel, skew, kurt - 3.0,
        min(1.0, stagnation / STAGNATION_SCALE), gen / max(1, n_gen - 1),
    ], dtype=np.float32)

def _spearman(a: Sequence[float], b: Sequence[float]) -> float:

    if len(a) < 3:
        return float("nan")
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    if np.all(xa == xa[0]) or np.all(xb == xb[0]):
        return float("nan")
    ra = np.argsort(np.argsort(xa)).astype(float)
    rb = np.argsort(np.argsort(xb)).astype(float)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if den <= 1e-12:
        return float("nan")
    return float((ra * rb).sum() / den)

def _diversity(pop: Sequence[dict]) -> float:
    keys = np.array([ind["keys"] for ind in pop], dtype=np.float64)
    return float(keys.std(axis=0).mean())

def _n_elites(pop_size: int, fraction: float = 0.10) -> int:

    return max(1, int(fraction * pop_size))

def _write_back(evaluator, decoder, inst, order, keys, best_order, best_starts):

    chosen = list(order)
    if decoder.lamarckian:
        if decoder.rekey == "schedule":
            schedule_order = evaluator._topo_order_from_starts(best_starts)
            if schedule_order != list(order):
                chosen = schedule_order
                keys = _order_to_keys(chosen, inst)
        elif list(best_order) != list(order):
            chosen = list(best_order)
            keys = _order_to_keys(chosen, inst)
    return keys, chosen

def _decay_budget(gen: int, n_gen: int, stagnation: int,
                  k_max: Optional[int] = None,
                  k_min: Optional[int] = None,
                  spent: int = 0,
                  target: int = 0,
                  settings: Optional[dict] = None) -> int:

    if settings is not None:
        return _mechanism_budget(gen, n_gen, stagnation, k_max, k_min,
                                 spent, target, settings)
    real_max = BUDGET_K_MAX if k_max is None else int(k_max)
    real_min = BUDGET_K_MIN if k_min is None else int(k_min)
    if BUDGET_DECAY == "linear":
        k = int(real_max - (real_max - real_min) * gen / (n_gen - 1))
    elif BUDGET_DECAY == "exp":
        ratio = (gen / (n_gen - 1)) ** 2
        k = int(real_max - (real_max - real_min) * ratio)
    else:
        progress = gen / (n_gen - 1)
        base = 1 - np.exp(-progress)
        penalty = np.exp(-stagnation / STAGNATION_SCALE)
        k = int(real_max - (real_max - real_min) * base * penalty)
    k = int(max(real_min, min(real_max, k)))

    if target > 0:
        remaining = target - spent
        gens_left = n_gen - gen
        if remaining <= 0:
            return 0
        nominal_left = 0
        for g in range(gen, n_gen):
            if BUDGET_DECAY == "adaptive":
                p = g / (n_gen - 1)
                nominal_left += int(max(real_min, min(
                    real_max,
                    real_max - (real_max - real_min) * (1 - np.exp(-p)))))
            elif BUDGET_DECAY == "linear":
                nominal_left += int(max(real_min, min(
                    real_max, real_max - (real_max - real_min) * g / (n_gen - 1))))
            else:
                nominal_left += int(max(real_min, min(
                    real_max, real_max - (real_max - real_min) * (g / (n_gen - 1)) ** 2)))
        if nominal_left > 0:
            k = int(round(k * remaining / nominal_left))

        k = max(0, min(k, remaining))
        if gens_left > 1:
            k = max(0, min(k, remaining - (gens_left - 1)))
    return int(k)

def _mechanism_budget(gen, n_gen, stagnation, k_max, k_min, spent, target, settings):

    mode = settings.get("mode", "adaptive")
    tau = float(settings.get("stagnation_scale", STAGNATION_SCALE))
    rate = float(settings.get("decay_rate", 1.0))
    if mode not in ("adaptive", "progress", "uniform"):
        raise ValueError(f"unknown mechanism budget mode: {mode}")
    if not np.isfinite(tau) or tau <= 0 or not np.isfinite(rate) or rate <= 0:
        raise ValueError("budget scale and decay rate must be positive and finite")
    if n_gen < 2 or target <= 0:
        raise ValueError("mechanism validation requires >=2 generations and a fixed budget")

    if mode == "adaptive" and tau == STAGNATION_SCALE and rate == 1.0:
        return _decay_budget(gen, n_gen, stagnation, k_max, k_min, spent, target)
    remaining = max(0, target - spent)
    left = n_gen - gen
    if remaining == 0:
        return 0
    if left == 1:
        return remaining
    if mode == "uniform":
        k = int(round(remaining / left))
    else:
        hi = BUDGET_K_MAX if k_max is None else int(k_max)
        lo = BUDGET_K_MIN if k_min is None else int(k_min)
        def nominal(g, stall=0):
            progress = g / (n_gen - 1)
            penalty = np.exp(-stall / tau) if mode == "adaptive" else 1.0
            return int(max(lo, min(hi, hi - (hi - lo)
                                  * (1 - np.exp(-rate * progress)) * penalty)))
        k = nominal(gen, stagnation)
        nominal_left = sum(nominal(g) for g in range(gen, n_gen))
        k = int(round(k * remaining / nominal_left))
    return int(max(0, min(k, remaining - (left - 1))))

def _mechanism_k_bounds(pop_size, n_gen, target, settings):

    import math
    from experiment_config import calibrated_k_bounds, BUDGET_K_HARD_CAP
    mode = settings.get("mode", "adaptive")
    rate = float(settings.get("decay_rate", 1.0))
    if mode in ("adaptive", "progress") and rate == 1.0:
        return calibrated_k_bounds(pop_size, n_gen, target=target)
    if mode == "uniform":
        return BUDGET_K_MIN, int(math.ceil(target / n_gen))
    shapes = [1 - math.exp(-rate * g / (n_gen - 1)) for g in range(n_gen)]
    total = sum(shapes)
    hi = math.ceil((target - BUDGET_K_MIN * total) / (n_gen - total))
    return BUDGET_K_MIN, max(BUDGET_K_MIN, min(BUDGET_K_HARD_CAP, hi))

def run_variant(
    inst: Instance,
    *,
    seed: int,
    method: str,
    use_surrogate: bool,
    use_meta_adaptation: bool,
    n_gen: int = N_GEN,
    pop_size: int = POP_SIZE,
    surrogate_params: Optional[dict] = None,
    surrogate_type: Optional[str] = None,
    capture_validation: bool = True,
    verbose: bool = False,
    decode_config: Optional[DecoderConfig] = None,
    parent_fraction: float = PARENT_FRACTION,
    renew: Optional[Dict[str, float]] = None,
    parent: Optional[Dict[str, float]] = None,
    sa: Optional[Dict[str, float]] = None,
    budget: Optional[dict] = None,
    capture_mechanism: bool = False,
) -> RunOutcome:

    decoder = decode_config or DecoderConfig.from_env()

    RN = {
        "enabled": 1.0 if RENEW_ENABLED else 0.0,
        "div": RENEW_DIVERSITY_FLOOR,
        "stag": float(RENEW_STAGNATION_MIN),
        "cool": float(RENEW_COOLDOWN),
        "budget": RENEW_BUDGET_FRACTION,
        "replace": RENEW_REPLACE_FRACTION,
        "pmin": RENEW_PERTURB[0],
        "pmax": RENEW_PERTURB[1],
        "elite": RENEW_ELITE_PROB,
        "mode": float(RENEW_MODE),
        "sigma": RENEW_SIGMA,

        "cheap": 0.0,
    }
    if renew:
        RN.update(renew)

    PF = {
        "mode": 1.0 if PARENT_FRACTION_MODE == "adaptive" else 0.0,
        "base": parent_fraction,
        "min": PARENT_FRACTION_MIN,
        "max": PARENT_FRACTION_MAX,
        "div_hi": PARENT_DIVERSITY_HI,
        "div_lo": PARENT_DIVERSITY_LO,
    }
    if parent:
        PF.update(parent)

    SA = {
        "fraction": SAMA_EXACT_FRACTION,
        "pool": SURROGATE_POOL_FACTOR,
        "retrain": float(SURROGATE_RETRAIN_EVERY),

        "feat": float(SURROGATE_FEATURE_MODE),

        "arch": float(SURROGATE_ARCHIVE_CAP),

        "eps": float(SURROGATE_EXPLORE),

        "diag": float(SURROGATE_DIAG),

        "ren_pool": float(SURROGATE_RENEW_POOL) if use_surrogate else 1.0,
    }
    if sa:
        SA.update(sa)
    set_seed(seed)
    t_wall0 = time.perf_counter()
    t_cpu0 = time.process_time()

    if BUDGET_PARITY_MODE == "calibrate":
        from experiment_config import calibrated_k_bounds

        parity_target = int(round(pop_size * n_gen * (
            SA["fraction"] if use_surrogate else 1.0)))
        budget_k_min, budget_k_max = calibrated_k_bounds(pop_size, n_gen,
                                                         target=parity_target)
        if budget is not None:
            budget_k_min, budget_k_max = _mechanism_k_bounds(
                pop_size, n_gen, parity_target, budget)
    else:
        budget_k_min, budget_k_max = BUDGET_K_MIN, BUDGET_K_MAX
        parity_target = 0

    evaluator = Evaluator(inst, use_cache=EVALUATOR_USE_CACHE)
    pred = precedence_lists(inst)
    evaluator.bind_precedence(pred)
    evaluator.bind_profile_changes(profile_change_points(inst))

    rng = np.random.RandomState(seed % (2 ** 31))

    lower = np.array([0] * len(inst.durations), dtype=float)
    upper = np.array([1] * len(inst.durations), dtype=float)
    if decoder.init == "portfolio":
        keys_pop = np.array(_initial_keys(inst, rng, pop_size, decoder.init_combos),
                            dtype=float)
    else:
        keys_pop = rng.uniform(0.0, 1.0, size=(pop_size, len(inst.durations)))
        lft_order = _min_lft_order(inst)
        keys_pop[0] = _order_to_keys(lft_order, inst)
        for i in range(1, pop_size):
            keys_pop[i] = _order_to_keys(_sampled_order(inst, lft_order, rng), inst)

    init_policy = None
    if use_meta_adaptation and decoder.polish_initial:
        init_policy = make_decode_policy(decoder.fixed_name, decoder.policy_values())

    pop: List[dict] = []
    for i in range(pop_size):
        order = random_keys_to_order(inst, keys_pop[i])
        if init_policy is not None:
            rec, best_order, best_starts = evaluator.evaluate_policy(order, init_policy)
            order = list(best_order)
            keys_pop[i] = _order_to_keys(order, inst)
        else:
            rec = evaluator.evaluate(order)
            best_starts = list(evaluator.start_times_cache())
        pop.append({"keys": keys_pop[i], "order": order, "fitness": rec.fitness,
                    "starts": best_starts})
    pop.sort(key=lambda d: d["fitness"])
    incumbent = dict(pop[0])
    initial_best = incumbent["fitness"]

    surrogate = None

    archive: List[Tuple[List[int], float]] = []
    surrogate_archive_size = 0
    def _feature(ind: dict):

        return ind["keys"] if SA["feat"] >= 1.0 else ind["order"]

    if use_surrogate:
        surrogate = SurrogateWrapper(
            surrogate_type or SURROGATE_TYPE, len(inst.durations),
            surrogate_params or SURROGATE_PARAMS)
        X = np.array([_feature(ind) for ind in pop], dtype=np.float32)
        y = np.array([ind["fitness"] for ind in pop], dtype=np.float32)
        surrogate.fit(X, y)
        archive.extend((_feature(ind), ind["fitness"]) for ind in pop)
        surrogate_archive_size = len(archive)

    controller = MetaController(seed, decoder=decoder) if use_meta_adaptation else None

    fname = decoder.fixed_name or (decoder.operators[0] if decoder.operators
                                   else "serial")
    if (decoder.fixed_alg or "GA").upper() == "DE":
        fixed_ops = OperatorSet(
            "DE",
            {"selection": "tournament", "crossover": "binomial",
             "mutation": "rand1", "local_search": "none", "schedule": fname},
            dict(ABLATION_FIXED_DE_PARAM, JITTER=decoder.de_jitter,
                 MUTP=decoder.de_mut_p),
        )
    else:
        fixed_ops = OperatorSet(
            "GA",
            {"selection": "tournament", "crossover": "sbx", "mutation": "polynomial",
             "local_search": "none", "schedule": fname},
            dict(ABLATION_FIXED_GA_PARAM),
        )
    fixed_vals = decoder.policy_values()

    if fname == "serial":
        fixed_policy = DecodePolicy(0, 1.0, 0.0, -1.0)
    else:
        fixed_policy = make_decode_policy(fname, fixed_vals)

    cheap_policy = DecodePolicy(0, 1.0, 0.0, -1.0)

    convergence: List[dict] = []
    last_best = incumbent["fitness"]
    stagnation = 0
    surrogate_err = float("nan")
    policy_time = 0.0

    last_renew_gen = -int(RN["cool"])
    n_renewals = 0
    rho_ema = float("nan")

    recall_ema = float("nan")
    regret_ema = float("nan")
    rho_all_ema = float("nan")
    diag_k_ema = float("nan")

    pop_prev_fits = [d["fitness"] for d in pop]

    for gen in range(n_gen):
        gen_best = None

        div_now = _diversity(pop)
        if PF["mode"] > 0.0:
            span = PF["div_hi"] - PF["div_lo"]
            t = 0.0 if span <= 0 else (PF["div_hi"] - div_now) / span
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            pf_now = PF["min"] + (PF["max"] - PF["min"]) * t
        else:
            pf_now = PF["base"]

        action_meta: List[dict] = []
        if use_meta_adaptation:
            state = _state_vector(evaluator, pop, initial_best, gen, stagnation, surrogate_err)
            import torch
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            t0 = time.perf_counter()
            n_slots = max(1, int(decoder.slots or 1))
            sampled = (controller.sample_multi(state_t, n_slots) if n_slots > 1
                       else [controller.sample(state_t)])
            for opset_k, (logp_k, prob_alg_k, prob_hyp_k, alg_idx_k) in sampled:
                action_meta.append({"opset": opset_k, "logp": logp_k,
                                    "prob_alg": prob_alg_k,
                                    "prob_hyp": prob_hyp_k,
                                    "alg_idx": alg_idx_k})

            opset = action_meta[0]["opset"]
            logp = action_meta[0]["logp"]
            prob_alg = action_meta[0]["prob_alg"]
            prob_hyp = action_meta[0]["prob_hyp"]
            policy = make_decode_policy(
                opset.operators.get("schedule", "serial"),
                {k[4:]: v for k, v in opset.parameters.items() if k.startswith("dec_")},
            )
            policy_time += time.perf_counter() - t0
        else:
            opset = fixed_ops

            policy = fixed_policy
            logp = prob_alg = prob_hyp = None

        budget_spent_before = evaluator.n_exact_sgs
        stagnation_before = stagnation
        budget_k = _decay_budget(gen, n_gen, stagnation, budget_k_max, budget_k_min,
                                 spent=evaluator.n_exact_sgs,
                                 target=parity_target, settings=budget)
        generation_total_allowance = budget_k

        pool_size = max(pop_size, int(budget_k * (
            SA["pool"] if use_surrogate else 1.0)) + 4)

        renew_budget = 0
        renew_now = False

        if (RN["enabled"] > 0.0 and gen < n_gen - 1
                and stagnation >= RN["stag"]
                and gen - last_renew_gen >= RN["cool"]
                and div_now < RN["div"]):
            renew_now = True
            renew_budget = max(1, int(round(budget_k * RN["budget"])))

            budget_k = max(0, int(budget_k) - renew_budget)
            last_renew_gen = gen
            n_renewals += 1

        frozen_elites = [ind.copy() for ind in pop[:_n_elites(pop_size)]]
        fresh_needed = pool_size if use_surrogate else max(1, int(budget_k))

        gate_reference = float(pop[0]["fitness"])

        if len(action_meta) > 1:

            k = len(action_meta)
            base, extra = divmod(max(0, int(fresh_needed)), k)
            slices: List[List[np.ndarray]] = []
            for j in range(k):
                want = base + (1 if j < extra else 0)
                if want <= 0:
                    slices.append([])
                    continue
                got = _generate_offspring(
                    inst, pop, action_meta[j]["opset"], want + (1 if j else 0),
                    rng, elite_fraction=0.0,
                    selection_percentile=(0.0, pf_now),
                )
                slices.append(got[1:] if j and len(got) > 1 else got[:want])

            offspring_keys = []
            slice_of: List[int] = []
            depth = max((len(sl) for sl in slices), default=0)
            for i in range(depth):
                for j, sl in enumerate(slices):
                    if i < len(sl):
                        offspring_keys.append(sl[i])
                        slice_of.append(j)
        else:
            offspring_keys = _generate_offspring(
                inst, pop, opset, fresh_needed, rng,
                elite_fraction=0.0,
                selection_percentile=(0.0, pf_now),
            )
            slice_of = [0] * len(offspring_keys)

        if use_surrogate and surrogate is not None:
            if SA["feat"] >= 1.0:

                X = np.array(offspring_keys, dtype=np.float32)
                orders = None
            else:
                orders = [random_keys_to_order(inst, k) for k in offspring_keys]
                X = np.array(orders, dtype=np.float32)
            predicted = surrogate.predict(X)
            rank = np.argsort(predicted)

            if SA["eps"] > 0.0 and len(rank) > 1:
                seq = list(rank)
                for head in range(len(seq) - 1):
                    if rng.random() < SA["eps"]:
                        j = rng.randint(head, len(seq) - 1)
                        seq[head], seq[j] = seq[j], seq[head]
                rank = np.array(seq, dtype=int)
            predicted = predicted[rank]
            if orders is not None:
                orders = [orders[i] for i in rank]
            offspring_keys = [offspring_keys[i] for i in rank]
            slice_of = [slice_of[i] for i in rank] if slice_of else []
        else:

            orders = None
            predicted = None

        renew_ranked: List[dict] = []
        if renew_now:
            n_act = len(inst.durations)
            renew_spent = 0

            draw = max(1, int(SA["ren_pool"])) if use_surrogate else 1
            guard = 0

            def _perturb() -> np.ndarray:

                base = incumbent["keys"] if rng.uniform() < RN["elite"] \
                    else pop[rng.randint(0, len(pop))]["keys"]
                pert = np.array(base, dtype=float, copy=True)
                frac = rng.uniform(RN["pmin"], RN["pmax"])
                m = max(1, int(round(frac * n_act)))
                idx = rng.choice(n_act, size=m, replace=False)
                if RN["mode"] >= 1.0:

                    pert[idx] = np.clip(
                        pert[idx] + rng.normal(0.0, RN["sigma"], size=m), 0.0, 1.0)
                else:
                    pert[idx] = rng.uniform(0.0, 1.0, size=m)
                return pert

            while renew_spent < renew_budget and guard < 20 * renew_budget + 20:
                guard += 1
                if draw > 1:
                    batch = [_perturb() for _ in range(draw)]
                    pred_b = surrogate.predict(np.array(batch, dtype=np.float32))
                    pick = [batch[j] for j in np.argsort(pred_b)[:1]]
                else:
                    pick = [_perturb()]
                for pert in pick:
                    order = random_keys_to_order(inst, pert)

                    rec, best_order, best_starts = evaluator.evaluate_policy(
                        order, cheap_policy if RN["cheap"] > 0.0 else policy,
                        reference=gate_reference, room=renew_budget - renew_spent)
                    renew_spent += int(rec.n_units)
                    keys, chosen = _write_back(evaluator, decoder, inst, order,
                                               pert, best_order, best_starts)
                    renew_ranked.append({"keys": keys, "order": chosen,
                                         "fitness": rec.fitness,
                                         "starts": list(best_starts)})
            renew_ranked.sort(key=lambda d: d["fitness"])

        ranked: List[dict] = []
        true_errs: List[float] = []
        screen_pred: List[float] = []
        screen_true: List[float] = []
        gen_sgs = 0
        pooled_surrogate = use_surrogate and surrogate is not None

        diag_mode = pooled_surrogate and SA["diag"] > 0.0
        budget_k_diag = budget_k
        diag_pred: List[float] = []
        diag_true: List[float] = []
        diag_cost: List[int] = []
        for pos in range(len(offspring_keys)):
            order = orders[pos] if orders is not None else random_keys_to_order(
                inst, offspring_keys[pos])
            room = budget_k - gen_sgs
            if diag_mode:
                room = None
            elif room <= 0 and pooled_surrogate:

                ranked.append({"keys": offspring_keys[pos], "order": order,
                               "fitness": float(predicted[pos]), "estimated": True})
                continue
            elif room <= 0:

                break
            rec, best_order, best_starts = evaluator.evaluate_policy(
                order, policy, reference=gate_reference, room=room)
            gen_sgs += int(rec.n_units)
            if pooled_surrogate:
                true_errs.append(abs(float(predicted[pos]) - rec.fitness))
                screen_pred.append(float(predicted[pos]))
                screen_true.append(float(rec.fitness))
                if diag_mode:
                    diag_pred.append(float(predicted[pos]))
                    diag_true.append(float(rec.fitness))
                    diag_cost.append(int(rec.n_units))
            keys, chosen_order = _write_back(evaluator, decoder, inst, order,
                                             offspring_keys[pos], best_order,
                                             best_starts)
            ranked.append({"keys": keys, "order": chosen_order,
                           "fitness": rec.fitness, "starts": list(best_starts),
                           "slice": slice_of[pos] if pos < len(slice_of) else 0})

        if true_errs:
            surrogate_err = float(np.mean(true_errs))

        if len(screen_true) >= 3:
            rho = _spearman(screen_pred, screen_true)
            if rho == rho:
                rho_ema = rho if rho_ema != rho_ema else 0.9 * rho_ema + 0.1 * rho

        if diag_mode and len(diag_true) >= 3:
            pred_a = np.asarray(diag_pred, dtype=float)
            true_a = np.asarray(diag_true, dtype=float)
            cost_a = np.asarray(diag_cost, dtype=int)
            by_pred = np.argsort(pred_a, kind="stable")

            spent = 0
            k = 0
            for j in by_pred:
                spent += int(cost_a[j])
                k += 1
                if spent >= budget_k_diag:
                    break
            if k > 0:
                selected = set(int(j) for j in by_pred[:k])
                by_true = np.argsort(true_a, kind="stable")
                top_true = set(int(j) for j in by_true[:k])
                recall_now = len(selected & top_true) / float(k)
                regret_now = float(true_a[list(selected)].min() - true_a.min())
                rho_diag = _spearman(diag_pred, diag_true)
                if recall_now == recall_now:
                    recall_ema = recall_now if recall_ema != recall_ema \
                        else 0.9 * recall_ema + 0.1 * recall_now
                if regret_now == regret_now:
                    regret_ema = regret_now if regret_ema != regret_ema \
                        else 0.9 * regret_ema + 0.1 * regret_now
                if rho_diag == rho_diag:
                    rho_all_ema = rho_diag if rho_all_ema != rho_all_ema \
                        else 0.9 * rho_all_ema + 0.1 * rho_diag
                diag_k_ema = k if diag_k_ema != diag_k_ema else 0.9 * diag_k_ema + 0.1 * k

        merged = frozen_elites + [r for r in ranked if not r.get("estimated")] + pop
        merged.sort(key=lambda d: d["fitness"])
        pop = merged[:pop_size]

        if renew_now and renew_ranked:
            n_keep = max(1, pop_size - max(1, int(RN["replace"] * pop_size)))
            n_keep = min(n_keep, pop_size - 1)
            pop = pop[:n_keep] + renew_ranked[:pop_size - n_keep]
            pop.sort(key=lambda d: d["fitness"])

        if use_surrogate and surrogate is not None:
            archive.extend(
                (_feature(r), r["fitness"])
                for r in ranked
                if not r.get("estimated")
            )
            if gen > 0 and gen % max(1, int(SA["retrain"])) == 0 and len(archive) >= 3:
                recent = archive
                if SA["arch"] > 0 and len(archive) > int(SA["arch"]):

                    recent = archive[-int(SA["arch"]):]
                ax = np.array([a[0] for a in recent], dtype=np.float32)
                ay = np.array([a[1] for a in recent], dtype=np.float32)
                surrogate.fit(ax, ay)
                surrogate_archive_size = len(archive)

        gen_best = pop[0]
        improved = gen_best["fitness"] < incumbent["fitness"]
        if improved:
            incumbent = dict(gen_best)
            stagnation = 0
        else:
            stagnation += 1

        prev_mean = float(np.mean(pop_prev_fits)) if pop_prev_fits else gen_best["fitness"]
        cur_mean = float(np.mean([d["fitness"] for d in pop]))
        denom = max(1.0, float(incumbent["fitness"]))
        mean_gain = (prev_mean - cur_mean) / denom
        best_gain = (last_best - gen_best["fitness"]) / denom
        reward = float(np.clip(REWARD_W_BEST * best_gain * 10.0
                               + REWARD_W_MEAN * mean_gain * 10.0, -1.0, 1.0))
        last_best = min(last_best, gen_best["fitness"])
        pop_prev_fits = [d["fitness"] for d in pop]

        slot_counts, slot_means, slot_rewards = [], [], []
        if use_meta_adaptation:
            t0 = time.perf_counter()
            if len(action_meta) > 1:

                k = len(action_meta)
                sums = [0.0] * k
                cnts = [0] * k
                for r in ranked:
                    if r.get("estimated"):
                        continue
                    j = int(r.get("slice", 0))
                    if 0 <= j < k:
                        sums[j] += float(r["fitness"])
                        cnts[j] += 1
                means = [(sums[j] / cnts[j] if cnts[j] else None) for j in range(k)]
                if capture_mechanism:
                    slot_counts, slot_means = list(cnts), list(means)
                    slot_rewards = [None] * k
                vals = [m for m in means if m is not None]
                if len(vals) >= 2:
                    mu = float(np.mean(vals))
                    sd = float(np.std(vals))
                    best_m = min(vals)
                    items = []
                    for j, m in enumerate(means):
                        if m is None:
                            continue
                        rj = -(m - mu) / (sd + 1e-9)
                        if capture_mechanism:
                            slot_rewards[j] = float(rj)

                        items.append((action_meta[j]["logp"], rj,
                                      action_meta[j]["prob_alg"],
                                      action_meta[j]["prob_hyp"]))
                        controller.record(action_meta[j]["opset"], rj,
                                          m <= best_m + 1e-12)
                    controller.update_multi(items)
            else:
                controller.update(logp, reward, prob_alg, prob_hyp)
                controller.record(opset, reward, improved)
            policy_time += time.perf_counter() - t0

        alg_log, sel_log, cro_log, mut_log, ls_log, par_log = opset.as_log()
        fits = [ind["fitness"] for ind in pop]
        convergence.append({
            "generation": gen,
            "elapsed_wall_s": round(time.perf_counter() - t_wall0, 4),
            "elapsed_cpu_s": round(time.process_time() - t_cpu0, 4),
            "best_verified_tpm": incumbent["fitness"],
            "generation_best_verified_tpm": gen_best["fitness"],
            "mean_verified_tpm": round(float(np.mean(fits)), 4),
            "population_diversity": round(_diversity(pop), 6),
            "n_candidates_cum": evaluator.n_candidates,
            "n_exact_sgs_cum": evaluator.n_exact_sgs,
            "n_surrogate_predictions_cum": surrogate.n_predictions if surrogate else 0,
            "surrogate_error_or_rank_metric": round(surrogate_err, 4) if surrogate_err == surrogate_err else "",
            "surrogate_archive_size": surrogate_archive_size,
            "selected_algorithm": alg_log,
            "selected_selection_operator": sel_log,
            "selected_crossover_operator": cro_log,
            "selected_mutation_operator": mut_log,

            "selected_local_search_operator": ("renew" if renew_now else ls_log),
            "selected_operator_parameters": par_log,
            "selected_schedule_operator": opset.operators.get("schedule", "serial"),
            "schedule_policy_cost": int(policy.cost),
            "algorithm_operator_reward": round(reward, 4),
            "algorithm_operator_success": int(improved),
            "exact_budget_k": budget_k,
            "stagnation_length": stagnation,
            "surrogate_rank_corr": (round(rho_ema, 4)
                                    if rho_ema == rho_ema else ""),
            "surrogate_recall_at_k": (round(recall_ema, 4)
                                      if recall_ema == recall_ema else ""),
            "surrogate_batch_regret": (round(regret_ema, 4)
                                       if regret_ema == regret_ema else ""),
            "surrogate_rho_all": (round(rho_all_ema, 4)
                                  if rho_all_ema == rho_all_ema else ""),
            "surrogate_k": (round(diag_k_ema, 2)
                            if diag_k_ema == diag_k_ema else ""),
        })
        if capture_mechanism:
            sampled = []
            for j, action in enumerate(action_meta):
                op = action["opset"]
                pa = action["prob_alg"].detach()
                ph = action["prob_hyp"].detach()
                def ent(p):
                    return float(-(p * controller.torch.log(p + 1e-8)).sum().item())
                sampled.append({
                    "slot": j, "algorithm": op.algorithm,
                    "operators": dict(op.operators), "parameters": dict(op.parameters),
                    "branch_entropy": ent(pa), "loss_head_entropy": ent(pa) + ent(ph),
                    "verified_count": slot_counts[j] if j < len(slot_counts) else 0,
                    "verified_mean": slot_means[j] if j < len(slot_means) else None,
                    "slot_reward": slot_rewards[j] if j < len(slot_rewards) else None,
                })
            convergence[-1].update({
                "budget_spent_before": budget_spent_before,
                "budget_spent_generation": evaluator.n_exact_sgs - budget_spent_before,
                "generation_total_allowance": generation_total_allowance,
                "offspring_allowance": budget_k,
                "renewal_spent_generation": renew_spent if renew_now else 0,
                "budget_remaining_before": parity_target - budget_spent_before,
                "budget_target": parity_target, "stagnation_before": stagnation_before,
                "budget_mode": (budget or {}).get("mode", BUDGET_DECAY),
                "entropy_coef": decoder.entropy_coef, "sampled_actions": sampled,
                "policy_updates_cum": controller._update_count,
            })
        if verbose and gen % 20 == 0:
            print(f"    gen {gen:3d} best={incumbent['fitness']:.0f} K={budget_k} "
                  f"stagn={stagnation} algo={alg_log}", flush=True)

    order = list(incumbent["order"])
    full_order = [0] + [i + 1 for i in order] + [len(inst.durations) + 1]
    evaluator2 = Evaluator(inst)
    evaluator2.bind_precedence(pred)
    evaluator2.bind_profile_changes(profile_change_points(inst))

    cached_starts = incumbent.get("starts")
    if cached_starts is not None and len(cached_starts) == len(inst.durations):
        start = [0] + [int(v) for v in cached_starts]
        start.append(int(max(start)))
    else:
        start = _start_times(inst, order)
    finish = [s + d for s, d in zip(start, [0] + list(inst.durations) + [0])]
    validation = validate_schedule(inst, start, finish) if capture_validation else {}

    return RunOutcome(
        best_tpm=int(incumbent["fitness"]),
        cmax=int(validation.get("cmax", incumbent["fitness"])),
        best_order=order,
        start=list(start),
        finish=list(finish),
        n_exact_sgs=evaluator.n_exact_sgs,
        n_candidates=evaluator.n_candidates,
        n_cache_hits=evaluator.n_cache_hits,
        wall_time_s=round(time.perf_counter() - t_wall0, 6),
        cpu_time_s=round(time.process_time() - t_cpu0, 6),
        decode_time_s=round(evaluator.decode_time_s, 6),
        surrogate_fit_time_s=round(surrogate.fit_time_s, 6) if surrogate else 0.0,
        surrogate_predict_time_s=round(surrogate.predict_time_s, 6) if surrogate else 0.0,
        policy_time_s=round(policy_time, 6),
        io_time_s=0.0,
        n_surrogate_retrains=surrogate.n_fits if surrogate else 0,
        n_surrogate_predictions=surrogate.n_predictions if surrogate else 0,
        algorithm_selection_counts=controller.select_count if controller else {},
        operator_selection_counts=controller.combo_count if controller else {},
        algorithm_operator_success_stats=(
            controller.snapshot()["algorithm_operator_success_stats"] if controller else {}),
        best_found_generation=_first_improvement_gen(convergence, incumbent["fitness"]),
        best_found_exact_eval=evaluator.n_exact_sgs,
        valid_schedule=bool(validation.get("valid", False)),
        validation=validation,
        convergence=convergence,
        use_surrogate=use_surrogate,
        use_meta_adaptation=use_meta_adaptation,
        budget_k_min=budget_k_min,
        budget_k_max=budget_k_max,
        budget_target_exact=parity_target if parity_target else pop_size * n_gen,
    )

def _first_improvement_gen(conv: Sequence[dict], best: int) -> int:
    for row in conv:
        if row["best_verified_tpm"] == best:
            return int(row["generation"])
    return -1

def _min_lft_order(inst: Instance) -> List[int]:
    n = len(inst.durations)
    pred = precedence_lists(inst)

    es = [0] * n
    for i in range(n):
        if pred[i]:
            es[i] = max(es[p] + inst.durations[p] for p in pred[i])
        es[i] = max(es[i], inst.release[i])
    horizon = max(es[i] + inst.durations[i] for i in range(n)) if n else 0
    ef = [es[i] + inst.durations[i] for i in range(n)]
    lf = [horizon] * n
    for i in reversed(range(n)):
        succs = inst.successors[i]
        if succs:
            lf[i] = min(lf[j] - inst.durations[j] for j in succs)
    lft = lf
    indeg = [len(pred[i]) for i in range(n)]
    ready = [i for i in range(n) if indeg[i] == 0]
    order: List[int] = []
    while ready:
        ready.sort(key=lambda i: (lft[i], i))
        node = ready.pop(0)
        order.append(node)
        for j in inst.successors[node]:
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
    return order

def _sampled_order(inst: Instance, lft_order: List[int], rng) -> List[int]:
    n = len(inst.durations)
    pred = precedence_lists(inst)
    es = [0] * n
    for i in _topo(inst):
        if pred[i]:
            es[i] = max(es[p] + inst.durations[p] for p in pred[i])
    horizon = max(es[i] + inst.durations[i] for i in range(n)) if n else 0
    lf = [horizon] * n
    for i in reversed(_topo(inst)):
        succs = inst.successors[i]
        if succs:
            lf[i] = min(lf[j] - inst.durations[j] for j in succs)
    indeg = [len(pred[i]) for i in range(n)]
    ready = [i for i in range(n) if indeg[i] == 0]
    order: List[int] = []
    while ready:
        max_lft = max(lf[i] for i in ready)
        weights = [max_lft - lf[i] + 1 for i in ready]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        pick = 0
        for pos, w in enumerate(weights):
            acc += w
            if acc >= r:
                pick = pos
                break
        node = ready.pop(pick)
        order.append(node)
        for j in inst.successors[node]:
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
    return order

def _topo(inst: Instance) -> List[int]:
    n = len(inst.durations)
    pred = precedence_lists(inst)
    indeg = [len(pred[i]) for i in range(n)]
    ready = [i for i in range(n) if indeg[i] == 0]
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for j in inst.successors[node]:
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
    return order

def _rule_priority(inst: Instance, activity_rule: str, project_rule: str,
                   rng) -> List[float]:

    n = len(inst.durations)
    dur = list(inst.durations)
    pred = precedence_lists(inst)
    topo = _topo(inst)

    es = [0.0] * n
    for i in topo:
        base = float(inst.release[i])
        if pred[i]:
            base = max(base, max(es[p] + dur[p] for p in pred[i]))
        es[i] = base
    horizon = max((es[i] + dur[i] for i in range(n)), default=0) or 1.0
    lf = [float(horizon)] * n
    for i in reversed(topo):
        s = inst.successors[i]
        if s:
            lf[i] = min(lf[j] - dur[j] for j in s)
    lft = lf
    slk = [lf[i] - es[i] - dur[i] for i in range(n)]

    nsucc = [0] * n
    for i in reversed(topo):
        for j in inst.successors[i]:
            nsucc[i] += 1 + nsucc[j]

    work = [float(sum(inst.demands[i])) * dur[i] for i in range(n)]

    if activity_rule == "lft":
        act = [float(v) for v in lft]
    elif activity_rule == "slk":
        act = [float(v) for v in slk]
    elif activity_rule == "est":
        act = [float(v) for v in es]
    elif activity_rule == "lst":
        act = [float(lft[i] - dur[i]) for i in range(n)]
    elif activity_rule == "work":
        act = [-v for v in work]
    elif activity_rule == "dur":
        act = [-float(v) for v in dur]
    elif activity_rule == "nsucc":
        act = [-float(v) for v in nsucc]
    else:
        act = [rng.random() for _ in range(n)]

    nproj = max(1, inst.num_projects)
    if project_rule == "flat":
        return act
    p_work = [0.0] * nproj
    p_cnt = [0.0] * nproj
    p_dur = [0.0] * nproj
    p_rel = [0.0] * nproj
    for i in range(n):
        pj = inst.project_of[i] if inst.project_of[i] < nproj else 0
        p_work[pj] += work[i]
        p_cnt[pj] += 1.0
        p_dur[pj] += dur[i]
        p_rel[pj] = min(p_rel[pj], inst.release[i]) if p_cnt[pj] > 1.0 \
            else float(inst.release[i])
    if project_rule == "work":
        key = [-v for v in p_work]
    elif project_rule == "cnt":
        key = [-v for v in p_cnt]
    elif project_rule == "dur":
        key = [-v for v in p_dur]
    elif project_rule == "rel":
        key = list(p_rel)
    else:
        key = [rng.random() for _ in range(nproj)]
    rank = [0] * nproj
    for pos, pj in enumerate(sorted(range(nproj), key=lambda p: (key[p], p))):
        rank[pj] = pos

    lo = min(act)
    hi = max(act)
    span = (hi - lo) or 1.0
    return [float(rank[inst.project_of[i] if inst.project_of[i] < nproj else 0])
            + 0.999 * (act[i] - lo) / span for i in range(n)]

def _priority_order(inst: Instance, pri: Sequence[float], rng) -> List[int]:

    n = len(inst.durations)
    pred = precedence_lists(inst)
    indeg = [len(pred[i]) for i in range(n)]
    ready = [i for i in range(n) if indeg[i] == 0]
    order: List[int] = []
    while ready:
        hi = max(pri[i] for i in ready)
        weights = [hi - pri[i] + 1.0 for i in ready]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        pick = ready[-1]
        for idx, i in enumerate(ready):
            acc += weights[idx]
            if r <= acc:
                pick = i
                break
        ready.remove(pick)
        order.append(pick)
        for j in inst.successors[pick]:
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
    return order

def _initial_keys(inst: Instance, rng, pop_size: int,
                  combos: Sequence[Tuple[str, str]]) -> List[np.ndarray]:

    keys: List[np.ndarray] = []

    det = _rule_priority(inst, "lft", "flat", rng)
    base = [det[i] for i in range(len(det))]
    order = sorted(range(len(base)), key=lambda i: (base[i], i))
    keys.append(_order_to_keys(_priority_fixed(inst, base), inst))
    for k in range(1, pop_size):
        p_rule, a_rule = combos[(k - 1) % len(combos)]
        pri = _rule_priority(inst, a_rule, p_rule, rng)
        keys.append(_order_to_keys(_priority_order(inst, pri, rng), inst))
    return keys

def _priority_fixed(inst: Instance, pri: Sequence[float]) -> List[int]:

    n = len(inst.durations)
    pred = precedence_lists(inst)
    indeg = [len(pred[i]) for i in range(n)]
    ready = [i for i in range(n) if indeg[i] == 0]
    order: List[int] = []
    while ready:
        ready.sort(key=lambda i: (pri[i], i))
        node = ready.pop(0)
        order.append(node)
        for j in inst.successors[node]:
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
    return order

def _order_to_keys(order: Sequence[int], inst: Instance) -> np.ndarray:
    n = len(order)
    ranks = np.linspace(1.0, 0.0, n, endpoint=False)
    keys = np.zeros(n, dtype=float)
    for pos, idx in enumerate(order):
        keys[idx] = ranks[pos]
    return keys

def _generate_offspring(inst: Instance, pop: List[dict], opset: OperatorSet,
                        pool_size: int, rng, *, elite_fraction: float,
                        selection_percentile: Tuple[float, float]) -> List[np.ndarray]:

    import random as _random

    n_pop = len(pop)
    pool = list(range(n_pop))
    fitness = [ind["fitness"] for ind in pop]
    n_elite = max(1, int(elite_fraction * n_pop))
    ranked = sorted(pool, key=lambda i: fitness[i])
    children: List[np.ndarray] = [pop[i]["keys"].copy() for i in ranked[:n_elite]]

    lo = int(selection_percentile[0] * n_pop)
    hi = max(lo + 1, int(selection_percentile[1] * n_pop))
    candidates = ranked[lo:hi]

    if opset.algorithm == "GA":
        cross = CROSSOVER_TABLE[opset.operators["crossover"]]
        mutate = MUTATION_TABLE[opset.operators["mutation"]]
        cr = opset.parameters["CR"]
        pm = opset.parameters["PM"]
        while len(children) < pool_size:
            i1 = _pick_selection(opset.operators["selection"], candidates, fitness, rng)
            i2 = _pick_selection(opset.operators["selection"], candidates, fitness, rng)
            attempts = 0
            while i2 == i1 and attempts < 10:
                i2 = _pick_selection(opset.operators["selection"], candidates, fitness, rng)
                attempts += 1
            c1, c2 = cross(pop[i1]["keys"], pop[i2]["keys"], cr, rng)
            c1 = mutate(c1, pm, rng)
            c2 = mutate(c2, pm, rng)
            children.append(c1)
            if len(children) < pool_size:
                children.append(c2)
    else:
        strategy = opset.operators["mutation"]
        cross = opset.operators["crossover"]
        F = opset.parameters["F"]
        CR = opset.parameters["CR"]
        best_keys = pop[ranked[0]]["keys"]

        off = rng.randint(0, len(pool))
        order_pool = pool[off:] + pool[:off]
        i_iter = iter(order_pool)
        while len(children) < pool_size:
            try:
                i = next(i_iter)
            except StopIteration:
                i_iter = iter(order_pool)
                i = next(i_iter)
            r1, r2 = _two_distinct(pool, i, rng)
            if strategy == "rand1":
                base = pop[r1]["keys"]
            elif strategy == "best1":
                base = best_keys
            else:
                base = None
            mutant = _de_mutant(pop[i]["keys"], pop[r1]["keys"], pop[r2]["keys"],
                                best_keys, F, strategy)

            jitter = float(opset.parameters.get("JITTER", DE_JITTER) or 0.0)
            if jitter > 0.0:
                mutant = np.clip(
                    mutant + rng.normal(0.0, jitter, size=mutant.shape),
                    0.0, 1.0)
            trial = _de_crossover(pop[i]["keys"], mutant, CR, cross, rng)

            mut_p = float(opset.parameters.get("MUTP", DE_MUT_P) or 0.0)
            if mut_p > 0.0:
                trial = MUTATION_TABLE[opset.operators.get(
                    "polymorph", "polynomial")](trial, mut_p, rng)
            children.append(trial)
    return children[:pool_size]

def _pick_selection(name: str, candidates: Sequence[int], fitness: Sequence[float], rng) -> int:
    if name == "roulette":
        best = min(fitness[i] for i in candidates)
        weights = [best / max(1e-9, fitness[i]) for i in candidates]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for i, w in zip(candidates, weights):
            acc += w
            if acc >= r:
                return i
        return candidates[-1]

    best = None
    for _ in range(3):
        cand = candidates[rng.randint(0, len(candidates) - 1)]
        if best is None or fitness[cand] < fitness[best]:
            best = cand
    return best

def _two_distinct(pool: Sequence[int], avoid: int, rng) -> Tuple[int, int]:
    r1 = avoid
    while r1 == avoid:
        r1 = pool[rng.randint(0, len(pool) - 1)]
    r2 = avoid
    while r2 == avoid or r2 == r1:
        r2 = pool[rng.randint(0, len(pool) - 1)]
    return r1, r2

def _de_mutant(current: np.ndarray, a: np.ndarray, b: np.ndarray,
               best: np.ndarray, F: float, strategy: str) -> np.ndarray:
    if strategy == "rand1":
        mutant = a + F * (b - a)
    elif strategy == "best1":
        mutant = best + F * (a - b)
    else:
        mutant = current + F * (best - current) + F * (a - b)
    return np.clip(mutant, 0.0, 1.0)

def _de_crossover(target: np.ndarray, mutant: np.ndarray, CR: float, mode: str, rng) -> np.ndarray:
    n = len(target)
    if mode == "binomial":
        mask = rng.random(n) <= CR
        if not mask.any():
            mask[rng.randint(0, n - 1)] = True
        trial = target.copy()
        trial[mask] = mutant[mask]
        return trial

    trial = target.copy()
    start = rng.randint(0, n - 1)
    length = 0
    while length < n and rng.random() <= CR:
        length += 1
    if length == 0:
        trial[start] = mutant[start]
    else:
        for off in range(length):
            pos = (start + off) % n
            trial[pos] = mutant[pos]
    return trial

def _start_times(inst: Instance, order: Sequence[int]) -> List[int]:

    evaluator = Evaluator(inst)
    evaluator.bind_precedence(precedence_lists(inst))
    evaluator.bind_profile_changes(profile_change_points(inst))
    return evaluator.start_times(order)
