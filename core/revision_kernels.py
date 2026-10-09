"""Generalized observation-by-observation kernels for the new simulation.

No fast-math, skipped observations, root-centered estimator, or altered test
boundary. Coefficients c are dimensionless: SA and PJ use c / b_ref, where
b_ref = A * R**(gamma - 1), R = right - left. For gamma=-1, the retained
positive-slope cubic model uses b_ref=A.

Comparator state (13 float64 entries):
    0:3  SA small / selected / large positions
    3    ASA position
    4    PJ position
    5:9  ASA mean x, mean y, centered Sxx, centered Sxy
    9    PJ compensated sum of pre-update positions minus initial midpoint
    10:12 ASA lower/upper slope clipping counts
    12   PJ Kahan compensation
PJ estimate is midpoint + state[9] / n. The midpoint is fixed and never theta.

PJ tuning arrays have shape (number of exponents, number of coefficients).
This computes k**(-s) once for each exponent and observation, sharing it among
coefficient candidates. Each candidate still receives every observation.
"""
from fractions import Fraction
import math
from pathlib import Path

import numpy as np
from numba import njit

STATE_SIZE = 13
METHODS = ("SA small", "SA selected", "SA large", "ASA", "PJ")


@njit(cache=True)
def signal(x, theta, gamma, A, R, rounded_sign):
    """Power core and its tangent tails; gamma=0 is a crossing jump.

    gamma=-1 denotes A*(e+2*e**3) on the interval-sized core, followed
    by its matching tangent outside that core. All other supported gamma
    values are nonnegative. R is the interval width.
    """
    e = x - theta
    if gamma == 0.:
        if e == 0.:
            return A * rounded_sign
        return A if e > 0. else -A
    z = abs(e)
    if gamma == -1.:
        if z <= R:
            value = z + 2. * z * z * z
        else:
            value = R + 2. * R * R * R + (1. + 6. * R * R) * (z - R)
    elif z <= R:
        value = z if gamma == 1. else z * z if gamma == 2. else z * z * z if gamma == 3. else z**gamma
    else:
        value = R**gamma + gamma * R**(gamma - 1.) * (z - R)
    value *= A
    return value if e > 0. else -value


@njit(cache=True)
def tuning_sa_block(noise, start, x, c_values, theta, gamma, A, R,
                    rounded_sign, b_ref):
    """Advance every SA candidate using the same observation noise."""
    for j in range(len(noise)):
        k = start + j
        eps = noise[j]
        for m in range(len(c_values)):
            y = signal(x[m], theta, gamma, A, R, rounded_sign) + eps
            x[m] -= (c_values[m] / b_ref) * y / k


@njit(cache=True)
def tuning_pj_block(noise, start, x, sums, compensation, c_values, s_values,
                    theta, gamma, A, R, rounded_sign, b_ref, left, right, center):
    """Advance the exponent-by-coefficient PJ grid, with pre-update averaging."""
    for j in range(len(noise)):
        k = start + j
        eps = noise[j]
        for row in range(len(s_values)):
            decay = k**(-s_values[row])
            for column in range(len(c_values)):
                centered = (x[row, column] - center) - compensation[row, column]
                updated_sum = sums[row, column] + centered
                compensation[row, column] = (updated_sum - sums[row, column]) - centered
                sums[row, column] = updated_sum
                y = signal(x[row, column], theta, gamma, A, R, rounded_sign) + eps
                x[row, column] = min(right, max(left,
                    x[row, column] - (c_values[column] / b_ref) * decay * y))


@njit(cache=True)
def comparator_block(noise, start, state, sa_c, pj_c, pj_s, theta, gamma,
                     A, R, rounded_sign, b_ref, left, right, center):
    """Three SA coefficients, running-OLS ASA, and projected averaged PJ."""
    for j in range(len(noise)):
        k = start + j
        eps = noise[j]
        ya = signal(state[3], theta, gamma, A, R, rounded_sign) + eps

        # Pre-update PJ position; center is the fixed initial midpoint.
        centered = (state[4] - center) - state[12]
        updated_sum = state[9] + centered
        state[12] = (updated_sum - state[9]) - centered
        state[9] = updated_sum

        # Running centered OLS uses every ASA (position, observation) pair.
        dx, dy = state[3] - state[5], ya - state[6]
        state[5] += dx / k
        state[6] += dy / k
        state[7] += dx * (state[3] - state[5])
        state[8] += dx * (ya - state[6])
        slope = state[8] / state[7] if state[7] > 0. else b_ref
        if slope < .1 * b_ref:
            state[10] += 1
        if slope > 10. * b_ref:
            state[11] += 1
        b = min(10. * b_ref, max(.1 * b_ref, slope))
        for m in range(3):
            multiplier = .25 if m == 0 else 1. if m == 1 else 4.
            y = signal(state[m], theta, gamma, A, R, rounded_sign) + eps
            state[m] -= ((sa_c / b_ref) * multiplier) * y / k
        state[3] -= ya / (k * b)
        yp = signal(state[4], theta, gamma, A, R, rounded_sign) + eps
        state[4] = min(right, max(left, state[4] - (pj_c / b_ref) * k**(-pj_s) * yp))


@njit(cache=True)
def scan_test(noise, signal_value, previous_sum, j, stage, K, v):
    """First strict boundary crossing, preserving the original cumsum blocks."""
    partial = 0.
    for k in range(len(noise)):
        partial += signal_value + noise[k]
        current = previous_sum + partial
        index = j + k + 1
        threshold = v * math.sqrt(2. * index * (K + stage) * math.log1p(index))
        if abs(current) > threshold:
            return k + 1, current, True
    return len(noise), previous_sum + partial, False


def kernel_parameters(cfg):
    """Normalize configuration types once, outside observation loops."""
    left = float(cfg["left"]) if "left" in cfg else float(Fraction(cfg.get("left_fraction", "0")))
    right = float(cfg["right"]) if "right" in cfg else float(Fraction(cfg.get("right_fraction", "1")))
    theta = float(cfg["theta"]) if "theta" in cfg else float(Fraction(cfg["theta_fraction"]))
    exact_theta = Fraction(cfg["theta_fraction"]) if "theta_fraction" in cfg else Fraction.from_float(theta)
    gamma = float(cfg["gamma"])
    A = float(cfg.get("A", cfg.get("amplitude", cfg.get("signal", 1.))))
    R = right - left
    if not (math.isfinite(left) and math.isfinite(right) and R > 0.):
        raise ValueError("The interval must have finite endpoints and positive width.")
    if not left < theta < right:
        raise ValueError("The root must lie strictly inside the interval.")
    if not math.isfinite(A) or A <= 0.:
        raise ValueError("A must be finite and positive.")
    if gamma not in (-1., 0., 1., 2., 3.):
        raise ValueError("Supported gamma values are -1, 0, 1, 2, and 3.")
    if "R" in cfg and not math.isclose(float(cfg["R"]), R, rel_tol=1e-14, abs_tol=0.):
        raise ValueError("R must equal right-left.")
    offset = Fraction.from_float(theta) - exact_theta
    rounded_sign = 1. if offset > 0 else -1. if offset < 0 else 0.
    b_ref = A if gamma == -1. else A * R**(gamma - 1.)
    if not math.isfinite(b_ref) or b_ref <= 0.:
        raise ValueError("The reference slope must be finite and positive.")
    return dict(theta=theta, gamma=gamma, A=A, R=R, rounded_sign=rounded_sign,
                b_ref=b_ref, left=left, right=right, center=left + .5 * R)


def reference_slope(cfg):
    return kernel_parameters(cfg)["b_ref"]


def initial_comparator_state(cfg):
    state = np.zeros(STATE_SIZE, dtype=np.float64)
    state[:5] = kernel_parameters(cfg)["center"]
    return state


def comparator_estimates(state, n, center):
    if n <= 0:
        raise ValueError("n must be positive.")
    return np.r_[state[:4], center + state[9] / n]


def _scalar_signal(x, p):
    """Independent direct mathematical signal for small validation cases."""
    error = x - p["theta"]
    gamma, width, amplitude = p["gamma"], p["R"], p["A"]
    if gamma == 0:
        return amplitude * (p["rounded_sign"] if error == 0 else math.copysign(1., error))
    distance = abs(error)
    if gamma == -1:
        value = (distance + 2 * distance**3 if distance <= width
                 else width + 2 * width**3 + (1 + 6 * width**2) * (distance - width))
    else:
        value = (distance**gamma if distance <= width else
                 width**gamma + gamma * width**(gamma - 1) * (distance - width))
    return amplitude * math.copysign(value, error)


def _scalar_reference(noise, p, sa_c, pj_c, pj_s, grid):
    """Direct scalar updates; recompute ASA OLS from complete stored histories."""
    x = [p["center"]] * 5
    asa_x, asa_y, pj_positions = [], [], []
    clips = [0, 0]
    rows = {}
    for k, eps in enumerate(noise, 1):
        ys = [_scalar_signal(value, p) + float(eps) for value in x]
        asa_x.append(x[3])
        asa_y.append(ys[3])
        mx, my = math.fsum(asa_x) / k, math.fsum(asa_y) / k
        sxx = math.fsum((value - mx)**2 for value in asa_x)
        sxy = math.fsum((xx - mx) * (yy - my) for xx, yy in zip(asa_x, asa_y))
        slope = sxy / sxx if sxx > 0 else p["b_ref"]
        clips[0] += int(slope < .1 * p["b_ref"])
        clips[1] += int(slope > 10 * p["b_ref"])
        b = min(10 * p["b_ref"], max(.1 * p["b_ref"], slope))
        pj_positions.append(x[4] - p["center"])
        for m, multiplier in enumerate((.25, 1., 4.)):
            x[m] -= ((sa_c / p["b_ref"]) * multiplier) * ys[m] / k
        x[3] -= ys[3] / (k * b)
        x[4] = min(p["right"], max(p["left"],
                   x[4] - (pj_c / p["b_ref"]) * k**(-pj_s) * ys[4]))
        if k in grid:
            rows[k] = dict(estimates=np.array(x[:4] + [p["center"] + math.fsum(pj_positions) / k]),
                           clips=np.array(clips), pj_position=x[4])
    return rows


def validate_kernels(old_kernel_path=None):
    """Small independent checks only; never launches a formal experiment.

    Pass the read-only original hpc_kernels.py path for backwards-comparison
    checks. Its decorators are made uncached only in memory, so validation
    cannot write compiler caches beside the reference file.
    """
    checks = []
    grid = (1, 2, 7, 31, 127, 257)
    generator = np.random.default_rng(np.random.SeedSequence([42, 0, 910]))
    configs = [
        dict(theta=.37, theta_fraction="37/100", gamma=gamma, A=amplitude, left=left, right=right)
        for gamma in (0., 1., 2., 3., -1.)
        for amplitude, left, right in ((1., 0., 1.), (2.5, -.4, 1.6))
    ]
    for cfg in configs:
        p = kernel_parameters(cfg)
        noise = np.ascontiguousarray(generator.normal(0., .17 * p["A"], grid[-1]))
        expected = _scalar_reference(noise, p, 1., .7, .75, grid)
        state = initial_comparator_state(cfg)
        q = 0
        for budget in grid:
            comparator_block(noise[q:budget], q + 1, state, 1., .7, .75, **p)
            np.testing.assert_allclose(comparator_estimates(state, budget, p["center"]),
                                       expected[budget]["estimates"], rtol=2e-11, atol=3e-13)
            np.testing.assert_array_equal(state[10:12], expected[budget]["clips"])
            np.testing.assert_allclose(state[4], expected[budget]["pj_position"], rtol=2e-12, atol=2e-13)
            q = budget
        checks.append(f"independent scalar updates/OLS: gamma={p['gamma']}, A={p['A']}, R={p['R']}")

    # Candidate grids match individually run comparator methods.
    cfg = dict(theta=.37, theta_fraction="37/100", gamma=3., A=2., left=-.2, right=1.3)
    p = kernel_parameters(cfg)
    c_values = np.array([.25, 1., 4.])
    s_values = np.array([2/3, .75, 1.])
    noise = np.ascontiguousarray(generator.normal(0., .2, 513))
    sa = np.full(len(c_values), p["center"])
    shape = (len(s_values), len(c_values))
    pj = np.full(shape, p["center"])
    sums, compensation = np.zeros(shape), np.zeros(shape)
    sa_parameters = {key: p[key] for key in ("theta", "gamma", "A", "R", "rounded_sign", "b_ref")}
    tuning_sa_block(noise, 1, sa, c_values, **sa_parameters)
    tuning_pj_block(noise, 1, pj, sums, compensation, c_values, s_values, **p)
    for row, exponent in enumerate(s_values):
        for column, coefficient in enumerate(c_values):
            state = initial_comparator_state(cfg)
            comparator_block(noise, 1, state, coefficient, coefficient, exponent, **p)
            np.testing.assert_array_equal(sa[column], state[1])
            np.testing.assert_array_equal(pj[row, column], state[4])
            np.testing.assert_array_equal(sums[row, column], state[9])
            np.testing.assert_array_equal(compensation[row, column], state[12])
    checks.append("SA/PJ tuning candidates equal standalone comparator trajectories")

    # Coordinate transform x'=offset+L*x and response transform y'=M*y.
    # For power/jump models A'=M*A/L**gamma and b_ref'=M*b_ref/L.
    L, M, offset = 2., 4., -.5
    for gamma in (0., 1., 2., 3.):
        cfg = dict(theta=.375, theta_fraction="3/8", gamma=gamma, A=2., left=0., right=1.)
        transformed = dict(theta=offset + L * cfg["theta"], gamma=gamma,
                           A=M * cfg["A"] / L**gamma, left=offset, right=offset + L)
        p, transformed_p = kernel_parameters(cfg), kernel_parameters(transformed)
        noise = np.ascontiguousarray(generator.normal(0., .1, 1000))
        base, changed = initial_comparator_state(cfg), initial_comparator_state(transformed)
        comparator_block(noise, 1, base, 1., .7, .75, **p)
        comparator_block(M * noise, 1, changed, 1., .7, .75, **transformed_p)
        np.testing.assert_allclose(comparator_estimates(changed, len(noise), transformed_p["center"]),
                                   offset + L * comparator_estimates(base, len(noise), p["center"]),
                                   rtol=5e-10, atol=3e-11)
        np.testing.assert_array_equal(base[10:12], changed[10:12])
        checks.append(f"coordinate/response scale equivariance: gamma={gamma}")

    # Linear SA with c=1 is theta - sample_mean(noise)/A after the first update.
    cfg = dict(theta=.375, theta_fraction="3/8", gamma=1., A=3., left=-.5, right=1.)
    p = kernel_parameters(cfg)
    noise = np.ascontiguousarray(generator.normal(0., .4, 10007))
    state = initial_comparator_state(cfg)
    comparator_block(noise, 1, state, 1., 1., .75, **p)
    expected = p["theta"] - math.fsum(map(float, noise)) / (p["A"] * len(noise))
    np.testing.assert_allclose(state[1], expected, rtol=0., atol=3e-14)
    checks.append("linear SA exact sample-mean identity with c=1")

    # PJ summation adds offsets to the initial midpoint, never the root.
    cfg = dict(theta=.125, theta_fraction="1/8", gamma=1., A=1., left=-2., right=3.)
    p = kernel_parameters(cfg)
    state = initial_comparator_state(cfg)
    comparator_block(np.array([0.]), 1, state, 1., 1., .75, **p)
    assert comparator_estimates(state, 1, p["center"])[4] == p["center"]
    checks.append("PJ first estimate is the initial midpoint (pre-update average)")

    # Independent exact first-crossing/cumsum check, including strict equality.
    for stage in (1, 7):
        noise = np.ascontiguousarray(generator.normal(0., .3, 8192))
        f, previous, j, K, v = .08, .2, 11, 7, .3
        path = previous + np.cumsum(f + noise)
        indexes = np.arange(j + 1, j + len(noise) + 1)
        hits = np.flatnonzero(np.abs(path) > v * np.sqrt(2. * indexes * (K + stage) * np.log1p(indexes)))
        used = int(hits[0] + 1) if len(hits) else len(noise)
        actual = scan_test(noise, f, previous, j, stage, K, v)
        assert actual[0] == used and actual[2] == bool(len(hits))
        np.testing.assert_array_equal(actual[1], path[used - 1])
    boundary = .3 * math.sqrt(2. * 1 * (7 + 1) * math.log1p(1))
    assert scan_test(np.array([boundary]), 0., 0., 0, 1, 7, .3)[2] is False
    checks.append("unchanged strict first-crossing and block-cumsum convention")

    backwards = []
    if old_kernel_path is not None:
        text = Path(old_kernel_path).read_text()
        if "@njit(cache=True)" not in text:
            raise ValueError("Unrecognized original kernel decorators; inspect the reference.")
        namespace = {"__name__": "_readonly_old_kernels", "__file__": str(old_kernel_path)}
        exec(compile(text.replace("@njit(cache=True)", "@njit(cache=False)"),
                     str(old_kernel_path), "exec"), namespace)
        for gamma in (0., 1., 2., 3.):
            cfg = dict(theta=.37, theta_fraction="37/100", gamma=gamma, A=1., left=0., right=1.)
            p = kernel_parameters(cfg)
            for coefficient in (1., 32.):
                noise = np.ascontiguousarray(generator.normal(0., .2, 10000))
                new = initial_comparator_state(cfg)
                old = np.zeros(12)
                old[:5] = .5
                comparator_block(noise, 1, new, coefficient, 1., .75, **p)
                namespace["comparator_block"](noise, 1, old, coefficient, p["theta"], gamma, p["rounded_sign"])
                np.testing.assert_array_equal(new[:9], old[:9])
                np.testing.assert_array_equal(new[10:12], old[10:12])
                pj_old, pj_new = old[9] / len(noise), p["center"] + new[9] / len(noise)
                np.testing.assert_allclose(pj_new, pj_old, rtol=5e-13, atol=5e-14)
                backwards.append(dict(gamma=gamma, sa_c=coefficient,
                                      pj_average_roundoff_difference=pj_new - pj_old))
        checks.append("old A=R=1 SA/ASA/PJ positions bitwise equal; PJ mean agrees to roundoff")
    return dict(passed=len(checks), checks=checks, backwards_comparison=backwards,
                seed=42, maximum_check_n=10007, formal_experiment_run=False)
