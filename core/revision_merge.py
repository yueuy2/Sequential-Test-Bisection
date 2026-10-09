"""Audit every predeclared formal task and normalize its complete raw records.

No simulation is launched. Invalid/nonfinite results remain in raw outputs and
the audit; an invalid campaign cannot be marked complete or used by the report.
ASTB rate summaries use only base replicates; extra linear CLT runs remain in
the all-replicates file. Retained legacy views have their own provenance/audit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ("SA small", "SA selected", "SA large", "ASA", "PJ", "STB", "ASTB")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default) + "\n")


def log_summary(log_values):
    """Arithmetic mean/MCSE and interpolated quantiles from logarithmic errors."""
    z = np.asarray(log_values, dtype=float)
    keys = ("mean", "median", "q10", "q90", "mcse")
    if np.isnan(z).any() or np.isposinf(z).any():
        invalid = math.inf if np.isposinf(z).any() else math.nan
        return {"log10_" + key: invalid if key == "mean" else math.nan for key in keys}
    if np.isneginf(z).all():
        return {"log10_" + key: -math.inf for key in keys}
    maximum = float(z.max())
    scaled = 10. ** (z - maximum)
    result = {}
    for key, value in (("mean", scaled.mean()), ("mcse", scaled.std(ddof=1) / math.sqrt(len(z)))):
        result["log10_" + key] = maximum + math.log10(value) if value > 0 else -math.inf
    ordered = np.sort(z)
    for key, probability in (("median", .5), ("q10", .1), ("q90", .9)):
        position = (len(z) - 1) * probability
        left = int(math.floor(position))
        weight = position - left
        result["log10_" + key] = float(ordered[left]) if weight == 0 else float(
            np.logaddexp(ordered[left] * math.log(10) + math.log1p(-weight),
                         ordered[left + 1] * math.log(10) + math.log(weight)) / math.log(10))
    return result


def pow10(value):
    if math.isnan(value):
        return math.nan
    if value == math.inf:
        return math.inf
    if value == -math.inf:
        return 0.
    try:
        return 10. ** value
    except OverflowError:
        return math.inf


def wilson(count, repetitions):
    z = stats.norm.ppf(.975)
    proportion = count / repetitions
    denominator = 1 + z*z / repetitions
    center = (proportion + z*z / (2*repetitions)) / denominator
    radius = z * math.sqrt(proportion*(1-proportion)/repetitions + z*z/(4*repetitions**2)) / denominator
    return center-radius, center+radius


def fail(audit, reason, **context):
    audit["failures"].append(dict(reason=reason, **context))


def common(cfg):
    return {key: cfg.get(key) for key in ("setting", "gamma", "theta", "theta_exact", "sigma", "A", "left", "right", "delta")}


def check_numeric(audit, row, keys, context, log_keys=()):
    for key in keys:
        value = row.get(key)
        if value is None or not math.isfinite(float(value)):
            fail(audit, "nonfinite_or_missing", field=key, value=value, **context)
    for key in log_keys:
        value = row.get(key)
        if value is None or math.isnan(float(value)) or value == math.inf:
            fail(audit, "invalid_logarithmic_value", field=key, value=value, **context)


def expected_tasks(campaign, plan, root, audit):
    selection = json.loads((root / "setting_selection.json").read_text())
    chosen = selection["selected"]["configuration"]
    if chosen not in plan["candidate_nuisance_settings"]:
        fail(audit, "Selected setting is absent from the predeclared candidate list")
    if selection.get("execution_fingerprint") != campaign.execution_fingerprint():
        fail(audit, "Selection execution fingerprint does not match current scientific sources")
    configs = campaign.cfgs_for(chosen)
    choices = json.loads((root / "formal_tuning/selected_steps.json").read_text())
    if set(choices) != {str(cfg["setting"]) for cfg in configs}:
        fail(audit, "Selected coefficient settings do not match frozen configuration")
    formal = plan["formal"]
    for setting, choice in choices.items():
        if choice.get("execution_fingerprint") != campaign.execution_fingerprint():
            fail(audit, "Tuning choice execution fingerprint mismatch", setting=setting)
        if choice.get("repetitions") != formal["tuning_repetitions"]:
            fail(audit, "Tuning repetitions mismatch", setting=setting)
    tuning = campaign.build_tuning(configs, formal["tuning_n"], formal["tuning_repetitions"], formal["tuning_phase"])
    evaluation = campaign.build_evaluation(configs, plan["grid"], formal["rate_repetitions"], formal["evaluation_phase"], choices)
    linear = next(cfg for cfg in configs if cfg["gamma"] == 1.)
    for n, repetitions in formal["linear_clt_repetitions"].items():
        evaluation.extend(dict(kind="astb", cfg=linear, n=int(n), rep=rep, phase=formal["evaluation_phase"])
                          for rep in range(formal["rate_repetitions"], repetitions))
    return configs, choices, {"formal_tuning": tuning, "formal_results": evaluation}


def audit_records(campaign, root, groups, audit):
    records, manifest = [], []
    expected_counts, actual_counts = Counter(), Counter()
    for folder, tasks in groups.items():
        directory = root / folder
        expected = {campaign.task_id(task): task for task in tasks}
        expected_counts.update(task["kind"] for task in tasks)
        if len(expected) != len(tasks):
            fail(audit, "Duplicate expected task IDs", folder=folder)
        recorded_manifest = directory / "expected_tasks.json"
        if not recorded_manifest.exists() or json.loads(recorded_manifest.read_text()) != tasks:
            fail(audit, "Saved task manifest does not match independently reconstructed workload", folder=folder)
        contract = directory / "execution_fingerprint.json"
        if not contract.exists() or json.loads(contract.read_text()).get("execution_fingerprint") != campaign.execution_fingerprint():
            fail(audit, "Output execution fingerprint mismatch", folder=folder)
        files = {path.stem: path for path in (directory / "tasks").glob("*.pkl")}
        for identifier in sorted(set(expected)-set(files)):
            audit["missing"].append(str(directory / "tasks" / (identifier + ".pkl")))
        for identifier in sorted(set(files)-set(expected)):
            fail(audit, "Unexpected task record retained in source directory", path=str(files[identifier]))
        for identifier, task in expected.items():
            if identifier not in files:
                continue
            path = files[identifier]
            manifest.append(dict(path=str(path.resolve()), sha256=digest(path), task=identifier, kind=task["kind"]))
            try:
                with path.open("rb") as handle:
                    record = pickle.load(handle)
                if record["task"] != task or record["task_hash"] != campaign.task_hash(task):
                    fail(audit, "Task/configuration fingerprint mismatch", path=str(path))
                if record.get("execution_fingerprint") != campaign.execution_fingerprint():
                    fail(audit, "Raw record source fingerprint mismatch", path=str(path))
                result = record["result"]
                context = dict(task=identifier, path=str(path))
                for key, expected_value in (("setting", task["cfg"]["setting"]), ("replicate", task["rep"])):
                    if key in result and result[key] != expected_value:
                        fail(audit, "Result identity does not match task", field=key, actual=result[key], expected=expected_value, **context)
                target = task.get("n", task.get("grid", [None])[-1])
                if result.get("q") != target:
                    fail(audit, "Observation budget mismatch", expected=target, actual=result.get("q"), **context)
                if result.get("status", "complete") != "complete":
                    fail(audit, "Terminal simulation failure", status=result.get("status"), result=result, **context)
                if task["kind"] in ("stb", "comparators"):
                    if [row.get("n") for row in result.get("rows", [])] != task["grid"]:
                        fail(audit, "Continuing-run reporting grid mismatch", **context)
                    for row in result.get("rows", []):
                        if row.get("q") != row.get("n"):
                            fail(audit, "Per-budget observation count mismatch", n=row.get("n"), q=row.get("q"), **context)
                        for key, expected_value in (("setting", task["cfg"]["setting"]), ("replicate", task["rep"])):
                            if key in row and row[key] != expected_value:
                                fail(audit, "Per-budget result identity mismatch", field=key, n=row.get("n"), **context)
                if task["kind"] == "astb" and result.get("tests", 0) + result.get("extra", 0) != target:
                    fail(audit, "ASTB test/extra total mismatch", **context)
                records.append((task, result, context))
                actual_counts[task["kind"]] += 1
            except Exception as error:
                fail(audit, "Unreadable or malformed task record", path=str(path), detail=repr(error))
    audit["counts"] = dict(expected=dict(expected_counts), observed=dict(actual_counts),
                           expected_total=sum(expected_counts.values()), observed_total=sum(actual_counts.values()))
    return records, manifest


def normalize(records, audit):
    rates, stbs, astbs, tuning = [], [], [], []
    for task, result, context in records:
        cfg, kind = task["cfg"], task["kind"]
        meta = dict(common(cfg), replicate=task["rep"], source_task=context["task"], source_path=context["path"])
        if kind == "tuning":
            tuning.append((task, result, context))
            continue
        if kind == "comparators":
            for row in result.get("rows", []):
                for index, method in enumerate(METHODS[:5]):
                    item = dict(meta, n=row["n"], q=row["q"], method=method,
                                estimate=row["estimates"][index], error=row["errors"][index],
                                log10_error=row["log10_errors"][index], exact_error=row["exact_errors"][index],
                                status=result.get("status", "complete"), fallback=0)
                    if method == "ASA":
                        item.update(clip_lower=row["clips"][0], clip_upper=row["clips"][1])
                    check_numeric(audit, item, ("estimate", "error"), dict(context, method=method, n=row["n"]), ("log10_error",))
                    rates.append(item)
        elif kind == "stb":
            for row in result.get("rows", []):
                item = dict(meta, **row)
                check_numeric(audit, item, ("estimate", "error", "length"), dict(context, n=row["n"]), ("log10_error", "log10_length"))
                for key in ("pointwise", "simultaneous"):
                    if item.get(key) not in (0, 1):
                        fail(audit, "Invalid coverage indicator", field=key, **context)
                stbs.append(item)
                rates.append(dict(item, method="STB"))
        elif kind == "astb":
            item = dict(meta, **result)
            item["model"] = "linear" if cfg["gamma"] == 1 else "pure_cubic" if cfg["gamma"] == 3 else "jump"
            check_numeric(audit, item, ("estimate", "error", "noise_sum", "slope", "D", "variance_estimate"), context, ("log10_error",))
            if cfg["gamma"] == 1:
                check_numeric(audit, item, ("beta", "normalized", "standardized", "linear_remainder"), context)
            astbs.append(item)
            rates.append(dict(item, method="ASTB"))
    return pd.DataFrame(rates), pd.DataFrame(stbs), pd.DataFrame(astbs), tuning


def audit_tuning(tuning, choices, plan, audit):
    rows = []
    for task, result, context in tuning:
        cfg, rep = task["cfg"], task["rep"]
        expected_cs = np.asarray(plan["sa_grid"])
        expected_pc = np.asarray(plan["pj_c_grid"])
        expected_ps = np.asarray(plan["pj_s_grid"][str(cfg["gamma"])])
        for key, expected in (("sa_c_grid", expected_cs), ("pj_c_grid", expected_pc), ("pj_s_grid", expected_ps)):
            if not np.array_equal(np.asarray(result[key]), expected):
                fail(audit, "Tuning grid mismatch", field=key, **context)
        for method, values in (("SA", result.get("sa_errors")), ("PJ", result.get("pj_errors"))):
            if values is None:
                if method == "SA" and cfg["gamma"] == 1:
                    continue
                fail(audit, "Missing tuning errors", method=method, **context)
                continue
            values = np.asarray(values)
            shape = (len(expected_cs),) if method == "SA" else (len(expected_ps), len(expected_pc))
            if values.shape != shape:
                fail(audit, "Tuning error-array shape mismatch", method=method, **context)
                continue
            for index in np.ndindex(values.shape):
                s = None if method == "SA" else expected_ps[index[0]]
                c = expected_cs[index[0]] if method == "SA" else expected_pc[index[1]]
                error = float(values[index])
                rows.append(dict(common(cfg), replicate=rep, n=task["n"], method=method, c=c, s=s, error=error, nonfinite=not math.isfinite(error), source_path=context["path"]))
                if not math.isfinite(error):
                    fail(audit, "Nonfinite tuning candidate retained", method=method, c=c, s=s, replicate=rep, **context)
    frame = pd.DataFrame(rows)
    for setting, part in frame.groupby("setting", sort=False):
        choice = choices[str(setting)]
        if part.iloc[0].gamma == 1. and choice["sa_c"] != 1.:
            fail(audit, "Linear SA must use oracle c=1", setting=int(setting))
        for method, candidates in part.groupby("method", sort=False):
            key = ["c"] if method == "SA" else ["s", "c"]
            means = candidates.groupby(key, sort=False, dropna=False).error.agg(
                lambda x: float(np.mean(np.where(np.isfinite(x), x, np.inf))))
            if not np.isfinite(means).any():
                fail(audit, "All tuning candidates failed", setting=int(setting), method=method)
                continue
            selected = means.idxmin()
            actual = choice["sa_c"] if method == "SA" else (choice["pj_s"], choice["pj_c"])
            if actual != selected:
                fail(audit, "Selected coefficient does not minimize complete independent tuning errors", setting=int(setting), method=method, expected=selected, actual=actual)
    return frame


def repetition_audit(rates, stbs, astbs, configs, plan, audit):
    base = plan["formal"]["rate_repetitions"]
    for cfg in configs:
        for n in plan["grid"]:
            for method in METHODS:
                part = rates[(rates.setting == cfg["setting"]) & (rates.n == n) & (rates.method == method) & (rates.replicate < base)]
                if sorted(part.replicate.tolist()) != list(range(base)):
                    fail(audit, "Rate replicate identities mismatch", setting=cfg["setting"], n=n, method=method)
            part = astbs[(astbs.setting == cfg["setting"]) & (astbs.n == n)]
            expected = plan["formal"]["linear_clt_repetitions"].get(str(n), base) if cfg["gamma"] == 1 else base
            if sorted(part.replicate.tolist()) != list(range(expected)):
                fail(audit, "ASTB replicate identities mismatch", setting=cfg["setting"], n=n, expected=expected)
    audit["repetitions"] = dict(rate_per_method=base, rate_replica_ids=f"0..{base-1}",
                               linear_clt_by_n=plan["formal"]["linear_clt_repetitions"],
                               astb_rate_excludes_extra_clt_replicates=True)


def summaries(rates, stbs, astbs, plan):
    base = plan["formal"]["rate_repetitions"]
    error_rows, coverage_rows, clt_rows = [], [], []
    rate_only = rates[rates.replicate < base]
    for (_, n, method), part in rate_only.groupby(["setting", "n", "method"], sort=False):
        first = part.iloc[0]
        values = log_summary(part.log10_error)
        error_rows.append(dict({key: first[key] for key in ("setting", "gamma", "theta", "sigma", "A", "left", "right")},
            n=int(n), method=method, repetitions=len(part), **values,
            **{key: pow10(values["log10_" + key]) for key in ("mean", "median", "q10", "q90", "mcse")},
            zeros=int(np.isneginf(part.log10_error).sum()),
            float64_underflow=int(((part.error == 0) & np.isfinite(part.log10_error)).sum()),
            nonfinite=int((~np.isfinite(part.error)).sum())))
    for (_, n), part in stbs.groupby(["setting", "n"], sort=False):
        first, count = part.iloc[0], len(part)
        length = log_summary(part.log10_length)
        point, simultaneous = int(part.pointwise.sum()), int(part.simultaneous.sum())
        pl, ph = wilson(point, count)
        sl, sh = wilson(simultaneous, count)
        coverage_rows.append(dict({key: first[key] for key in ("setting", "gamma", "theta", "sigma", "A", "left", "right")},
            n=int(n), repetitions=count, target=1-float(first.delta),
            pointwise=point/count, point_low=pl, point_high=ph, simultaneous=simultaneous/count,
            simultaneous_low=sl, simultaneous_high=sh,
            **{key + "_length": value for key, value in length.items()},
            **{key + "_length": pow10(length["log10_" + key]) for key in ("mean", "median", "q10", "q90", "mcse")},
            length_mcse=pow10(length["log10_mcse"]),
            median_stages=float(part.stages.median()),
            zeros=int(np.isneginf(part.log10_length).sum()),
            float64_underflow=int(((part.length == 0) & np.isfinite(part.log10_length)).sum())))
    linear = astbs[(astbs.gamma == 1) & astbs.n.isin(map(int, plan["formal"]["linear_clt_repetitions"]))]
    for n, part in linear.groupby("n", sort=True):
        first, count = part.iloc[0], len(part)
        z = part.normalized.to_numpy(float)
        target = (float(first.sigma) / float(first.beta))**2
        gen = np.random.default_rng(np.random.SeedSequence([42, 20261009, 250, 50, int(first.setting), int(n)]))
        variances = z[gen.integers(0, count, size=(1000, count))].var(axis=1, ddof=1)
        empirical = np.sort(part.standardized.to_numpy(float))
        theoretical = stats.norm.ppf((np.arange(count) + .5) / count)
        clt_rows.append(dict(model="linear", setting=int(first.setting), gamma=1., theta=first.theta,
            sigma=first.sigma, beta=first.beta, n=int(n), repetitions=count, mean=z.mean(),
            mean_mcse=z.std(ddof=1)/math.sqrt(count), variance=z.var(ddof=1),
            variance_low=np.quantile(variances, .025), variance_high=np.quantile(variances, .975),
            target_variance=target, variance_ratio=z.var(ddof=1)/target,
            median_slope=part.slope.median(), mean_extra_fraction=(part.extra/n).mean(),
            mean_variance_estimate=part.variance_estimate.mean(), fallbacks=int(part.fallback.sum()),
            interval_failures=int((1-part.covered).sum()),
            remainder_rmse=float(np.sqrt(np.mean(part.linear_remainder.to_numpy(float)**2))),
            qq_outside=int(((abs(empirical)>3.5) | (abs(theoretical)>3.5)).sum()),
            max_abs_standardized=float(np.max(np.abs(empirical)))))
    return pd.DataFrame(error_rows), pd.DataFrame(coverage_rows), pd.DataFrame(clt_rows)


def legacy_inputs(directory, output, audit):
    """Keep legacy settings/replicates separate and verify their original tasks."""
    directory = Path(directory).resolve()
    coverage_path = directory / "coverage_length_summary.csv"
    cubic_path = directory / "clt_all_replicates.csv"
    legacy_audit = dict(status="pending", missing=[], failures=[], counts={})
    provenance = dict(coverage=dict(source=str(coverage_path), sha256=digest(coverage_path)),
                      cubic_qq=dict(source=str(cubic_path), sha256=digest(cubic_path)))
    coverage = pd.read_csv(coverage_path)
    coverage = coverage[coverage.gamma == 2].copy()
    cubic = pd.read_csv(cubic_path)
    cubic = cubic[cubic.model == "cubic"].copy()
    coverage.to_csv(output / "legacy_gamma2_coverage.csv", index=False)
    cubic.to_csv(output / "legacy_cubic_qq_replicates.csv", index=False)
    grid = [10**k for k in range(2, 10)]
    expected_cs = {(setting, n) for setting in (4, 5, 6, 7) for n in grid}
    if set(zip(coverage.setting, coverage.n)) != expected_cs or len(coverage) != len(expected_cs) or not (coverage.repetitions == 200).all():
        fail(legacy_audit, "Legacy gamma2 coverage rows/repetitions mismatch")
    for key in ("pointwise", "point_low", "point_high", "simultaneous", "simultaneous_low", "simultaneous_high", "log10_median_length", "log10_q10_length", "log10_q90_length", "median_stages"):
        if not np.isfinite(coverage[key]).all():
            fail(legacy_audit, "Legacy gamma2 nonfinite summary retained", field=key)
    for n in [10**k for k in range(3, 10)]:
        repetitions = 400 if n <= 10**6 else 200
        part = cubic[cubic.n == n]
        if sorted(part.replicate.tolist()) != list(range(repetitions)):
            fail(legacy_audit, "Legacy cubic replicate identities mismatch", n=n)
    if not ((cubic.tests + cubic.extra == cubic.n) & (cubic.q == cubic.n)).all():
        fail(legacy_audit, "Legacy cubic observation counts mismatch")
    for key in ("estimate", "normalized", "standardized", "noise_sum", "slope", "D", "variance_estimate"):
        if not np.isfinite(cubic[key]).all():
            fail(legacy_audit, "Legacy cubic nonfinite values retained", field=key)
    task_manifest = []
    specs = [("stb", setting, 0, rep) for setting in (4, 5, 6, 7) for rep in range(200)]
    specs += [("astb", 1, 10**k, rep) for k in range(3, 10) for rep in range(400 if k <= 6 else 200)]
    for kind, setting, n, rep in specs:
        name = f"{kind}_{setting:02d}_{n:010d}_{rep:04d}"
        path = directory / "tasks" / (name + ".pkl")
        if not path.exists():
            legacy_audit["missing"].append(str(path))
            continue
        task_manifest.append(dict(path=str(path), sha256=digest(path)))
        with path.open("rb") as handle:
            record = pickle.load(handle)
        result = record["result"]
        if list(record["task"][:4]) != [kind, setting, rep, n]:
            fail(legacy_audit, "Legacy task identity mismatch", path=str(path))
        target = n if kind == "astb" else 10**9
        if result.get("q") != target:
            fail(legacy_audit, "Legacy task observation total mismatch", path=str(path))
        if kind == "stb":
            if [row["n"] for row in result["rows"]] != grid or any(row["q"] != row["n"] for row in result["rows"]):
                fail(legacy_audit, "Legacy STB reporting grid/count mismatch", path=str(path))
        elif result["tests"] + result["extra"] != n:
            fail(legacy_audit, "Legacy ASTB test/extra count mismatch", path=str(path))
    legacy_audit["counts"] = dict(expected_stb_tasks=800, expected_cubic_astb_tasks=2200,
                                  observed_tasks=len(task_manifest), coverage_rows=len(coverage), cubic_replicates=len(cubic))
    legacy_audit["status"] = "complete" if not legacy_audit["missing"] and not legacy_audit["failures"] else "failed"
    write_json(output / "legacy_task_manifest.json", task_manifest)
    provenance["task_manifest_sha256"] = digest(output / "legacy_task_manifest.json")
    provenance["separate_scope"] = "Original four gamma2 confidence-sequence settings and positive-slope e+2e^3 cubic QQ only; never mixed into new comparisons."
    provenance["audit"] = legacy_audit
    write_json(output / "legacy_provenance.json", provenance)
    if legacy_audit["status"] != "complete":
        fail(audit, "Legacy input audit failed", audit=legacy_audit)
    return provenance


def merge(root, output, legacy_data):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    import revision_campaign as campaign
    campaign.ROOT = root
    campaign.execution_fingerprint.cache_clear()
    audit = dict(status="running", missing=[], failures=[], started=time.time(),
                 scientific_execution_fingerprint=campaign.execution_fingerprint(),
                 checkpoint_policy="Completed-task result records are authoritative; unfinished checkpoints remain resumable.")
    write_json(output / "formal_completion_audit.json", audit)
    try:
        plan = json.loads((root / "revision_design.json").read_text())
        configs, choices, groups = expected_tasks(campaign, plan, root, audit)
        records, manifest = audit_records(campaign, root, groups, audit)
        write_json(output / "formal_raw_task_manifest.json", manifest)
        rates, stbs, astbs, tuning = normalize(records, audit)
        # Write raw rows before summaries or completeness rejection.
        rates.to_csv(output / "all_error_replicates.csv", index=False)
        stbs.to_csv(output / "stb_all_replicates.csv", index=False)
        astbs.to_csv(output / "astb_all_replicates.csv", index=False)
        tuning_frame = audit_tuning(tuning, choices, plan, audit)
        tuning_frame.to_csv(output / "tuning_all_replicates.csv", index=False)
        if not rates.empty and not astbs.empty:
            repetition_audit(rates, stbs, astbs, configs, plan, audit)
        else:
            fail(audit, "No formal data available")
        legacy = legacy_inputs(legacy_data, output, audit)
        if not audit["missing"] and not audit["failures"]:
            errors, coverage, clt = summaries(rates, stbs, astbs, plan)
            errors.to_csv(output / "error_summary.csv", index=False)
            coverage.to_csv(output / "coverage_length_summary.csv", index=False)
            clt.to_csv(output / "clt_summary.csv", index=False)
            audit["summary_rows"] = dict(errors=len(errors), coverage=len(coverage), linear_clt=len(clt))
        audit["input_provenance"] = dict(design=dict(path=str(root / "revision_design.json"), sha256=digest(root / "revision_design.json")),
            selection=dict(path=str(root / "setting_selection.json"), sha256=digest(root / "setting_selection.json")),
            selected_steps=dict(path=str(root / "formal_tuning/selected_steps.json"), sha256=digest(root / "formal_tuning/selected_steps.json")),
            raw_manifest_sha256=digest(output / "formal_raw_task_manifest.json"),
            legacy_provenance_sha256=digest(output / "legacy_provenance.json"))
        audit["output_sha256"] = {path.name: digest(path) for path in sorted(output.glob("*.csv"))}
    except Exception as error:
        fail(audit, "Merge exception; existing raw files remain preserved", detail=repr(error))
        raise
    finally:
        audit["finished"] = time.time()
        audit["status"] = "complete" if not audit["missing"] and not audit["failures"] else "failed"
        write_json(output / "formal_completion_audit.json", audit)
    if audit["status"] != "complete":
        raise RuntimeError(f"Formal audit failed: {len(audit['missing'])} missing, {len(audit['failures'])} failures; see {output / 'formal_completion_audit.json'}")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-directory", type=Path)
    parser.add_argument("--legacy-data", type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.run_directory, args.data_directory or args.run_directory / "data", args.legacy_data)
    print(json.dumps(result, indent=2, default=json_default))
