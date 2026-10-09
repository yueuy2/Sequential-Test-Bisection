"""Numerical display helpers copied from the verified report generator."""

import os, math
from fractions import Fraction
from pathlib import Path

METHODS = (
    "SA small", "SA selected", "SA large", "ASA", "PJ", "STB", "ASTB",
)

COLORS = ("#0072B2", "#333333", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")

LINE_STYLES = ("--", "-", "-.", ":", "--", "-", "-.")

GAMMAS = (1, 3, 0)

RATE_GRID = tuple(10**k for k in range(2, 10))

CLT_GRID = tuple(10**k for k in range(3, 10))

CLT_REPETITIONS = {n: 400 if n <= 10**6 else 200 for n in CLT_GRID}

QQ_WINDOW = (-3.5, 3.5)

SCHEMA = {
    "error_summary.csv": {
        "key": ["setting", "n", "method"],
        "required": [
            "setting", "gamma", "theta", "sigma", "n", "method",
            "repetitions", "log10_mean", "log10_mcse",
        ],
        "optional": ["mean", "mcse", "zeros", "float64_underflow", "nonfinite"],
        "meaning": (
            "Three selected families with one common parameter configuration. "
            "All seven methods, all rate budgets, 200 replicates per row. "
            "ASTB rate summaries always use base replicates 0..199, including "
            "at budgets with 200 additional linear CLT replicates."
        ),
    },
    "coverage_length_summary.csv": {
        "key": ["setting", "n"],
        "required": [
            "setting", "gamma", "theta", "sigma", "n", "repetitions",
            "pointwise", "point_low", "point_high", "simultaneous",
            "simultaneous_low", "simultaneous_high", "log10_median_length",
            "log10_q10_length", "log10_q90_length", "median_stages",
        ],
        "meaning": "Fresh STB runs for the three selected families only.",
    },
    "astb_all_replicates.csv": {
        "key": ["setting", "n", "replicate"],
        "required": [
            "setting", "gamma", "theta", "sigma", "n", "replicate",
            "estimate", "fallback", "tests", "extra", "noise_sum",
        ],
        "linear_clt_required": ["beta", "normalized", "standardized", "linear_remainder"],
        "optional": ["model", "q", "slope", "D", "variance_estimate", "covered"],
        "meaning": (
            "Fresh ASTB rate runs at every rate budget; 200 base runs per family. "
            "Linear runs add replicates 200..399 at 1e3..1e6 for the CLT. "
            "Other families have no positive-slope CLT claim."
        ),
    },
    "legacy_inputs": {
        "coverage": "Explicit path to original coverage_length_summary.csv, filtered gamma=2.",
        "cubic_qq": "Explicit path to original clt_all_replicates.csv, filtered model=cubic.",
        "provenance": "Record source paths and SHA-256 hashes in the new report manifest.",
    },
    "design_requirements": {
        "seed": 42, "max_n": 10**9, "rate_repetitions": 200,
        "gammas": list(GAMMAS), "rate_grid": list(RATE_GRID),
        "clt_repetitions": CLT_REPETITIONS,
        "selection": "Persist common configuration and independent selection stream namespace.",
        "sa_tuning": "Persist 60 independent repetitions per candidate and selected coefficients.",
        "output_directory": "simulation_revision_2026_10_09",
        "report_path": "simulation_2026_10_07/simulation_notes.tex",
    },
}

def latex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))

def log_sci(value, negative=False):
    """Render a magnitude from its logarithm without underflow or a fake floor."""
    if value == -math.inf:
        return "$0$"
    if not math.isfinite(value):
        raise ValueError("Nonfinite summary requires review before report generation")
    exponent = math.floor(value)
    mantissa = 10.0 ** (value - exponent)
    if round(mantissa, 2) >= 10:
        mantissa /= 10
        exponent += 1
    sign = "-" if negative else ""
    return rf"${sign}{mantissa:.2f}\times10^{{{exponent}}}$"

def sci(value):
    if value == 0:
        return "$0$"
    return log_sci(math.log10(abs(value)), negative=value < 0)

def power(n):
    exponent = round(math.log10(n))
    if n != 10**exponent:
        raise ValueError(f"Budget {n} is not a decimal power")
    return rf"$10^{{{exponent}}}$"

def require_columns(frame, filename):
    missing = set(SCHEMA[filename]["required"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{filename}: missing columns {sorted(missing)}")
    key = SCHEMA[filename]["key"]
    if frame.duplicated(key).any():
        raise ValueError(f"{filename}: duplicate keys {key}")

def validate_error_summary(frame):
    """Reject partial formal output before it can become a table or plot."""
    import numpy as np

    require_columns(frame, "error_summary.csv")
    if set(frame.gamma) != set(GAMMAS):
        raise ValueError("The new comparison requires gamma=1, 3, 0")
    if frame.theta.nunique() != 1 or frame.sigma.nunique() != 1:
        raise ValueError("The three new families must share theta and sigma")
    for gamma, family in frame.groupby("gamma"):
        if family.setting.nunique() != 1:
            raise ValueError(f"gamma={gamma}: expected one selected setting")
        observed = set(zip(family.n, family.method))
        expected = {(n, method) for n in RATE_GRID for method in METHODS}
        if observed != expected or len(family) != len(expected):
            raise ValueError(f"gamma={gamma}: incomplete or unexpected method/budget rows")
    if not (frame.repetitions == 200).all():
        raise ValueError("Every comparison summary must contain 200 repetitions")
    for name in ("log10_mean", "log10_mcse"):
        values = frame[name].to_numpy()
        if np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError(f"Invalid {name}; exact zero may use negative infinity")

def full_sample_qq(values):
    """Return all QQ pairs and the display-only outside count."""
    import numpy as np
    from scipy import stats

    sample = np.asarray(values, dtype=float)
    if not len(sample) or not np.isfinite(sample).all():
        raise ValueError("QQ diagnostics require every finite replicate")
    empirical = np.sort(sample)
    theoretical = stats.norm.ppf((np.arange(len(sample)) + 0.5) / len(sample))
    low, high = QQ_WINDOW
    outside = ((empirical < low) | (empirical > high) |
               (theoretical < low) | (theoretical > high))
    return theoretical, empirical, int(outside.sum())

def configure_publication_style():
    """Use real LaTeX for text, matching the prior publication figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.serif": ["Times"],
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
        "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12,
        "legend.fontsize": 10, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.17,
        "figure.dpi": 150,
    })
    return plt

def error_table(frame, gamma):
    """Compact method rows avoid a wide nine-column comparison table."""
    part = frame[(frame.gamma == gamma) & frame.n.isin((10**8, 10**9))]
    rows = []
    for method in METHODS:
        values = part[part.method == method].set_index("n")
        if set(values.index) != {10**8, 10**9}:
            raise ValueError(f"Incomplete table data for gamma={gamma}, {method}")
        rows.append(" & ".join([
            latex_escape(method_label(method)), log_sci(values.loc[10**8, "log10_mean"]),
            log_sci(values.loc[10**9, "log10_mean"]),
        ]) + r" \\")
    return "\n".join([
        r"\begin{tabular}{lcc}\toprule",
        r"Method & $n=10^8$ & $n=10^9$ \\ \midrule",
        *rows, r"\bottomrule\end{tabular}",
    ])

def method_label(method):
    """Display name; immutable raw task identifiers are unchanged."""
    return "SA optimal" if method == "SA selected" else method

def theta_tex(value):
    if abs(float(value) - 0.37) < 1e-12:
        return "$0.37$"
    fraction = Fraction(str(value)).limit_denominator(10000)
    if abs(float(fraction) - float(value)) < 1e-12 and fraction.denominator != 1:
        return rf"${fraction.numerator}/{fraction.denominator}$"
    return f"${float(value):g}$"

def save_figure(plt, fig, directory, name):
    for extension in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{extension}", bbox_inches="tight")
    plt.close(fig)

def logarithmic_axes(ax, ylabel):
    from matplotlib.ticker import FixedLocator, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator([2, 3, 4, 5, 6, 7, 8, 9]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: rf"$10^{{{x:g}}}$"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: rf"$10^{{{y:g}}}$"))
    ax.set(xlabel=r"Total observations $n$", ylabel=ylabel)

def make_figures(bundle, directory):
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    plt = configure_publication_style()
    error = bundle["error"]
    jump = error[(error.gamma == 0) & (error.method == "STB")].sort_values("n")
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(np.sqrt(jump.n / np.log(jump.n)), jump.log10_mean, "o-", color=COLORS[5])
    ax.set(xlabel=r"$\sqrt{n/\log n}$", ylabel=r"$\log_{10}$ mean absolute error")
    fig.tight_layout()
    save_figure(plt, fig, directory, "jump_transformed")
    for gamma in (1, 2, 3, 0):
        frame = bundle["legacy_cs"] if gamma == 2 else bundle["cs"][bundle["cs"].gamma == gamma]
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
        for (setting, part), color in zip(frame.groupby("setting"), COLORS):
            part = part.sort_values("n")
            head = part.iloc[0]
            x = np.log10(part.n.to_numpy())
            label = rf"$\theta={theta_tex(head.theta)[1:-1]},\ \sigma={head.sigma:g}$"
            axes[0].plot(x, part.pointwise, color=color, label=label)
            axes[0].plot(x, part.simultaneous, "o", color=color, ms=3, fillstyle="none")
            axes[1].plot(x, part.log10_median_length, "o-", color=color, ms=3, label=label)
            axes[1].fill_between(x, part.log10_q10_length.to_numpy(), part.log10_q90_length.to_numpy(), color=color, alpha=.1)
        target = float(frame.target.iloc[0]) if "target" in frame else 1-float(bundle["cfg"]["delta"])
        axes[0].axhline(target, color="black", ls="--", lw=1)
        axes[0].set(ylim=(max(0., min(target-.025, float(frame.simultaneous.min())-.015)), 1.005), ylabel="Coverage probability", xlabel=r"Total observations $n$")
        from matplotlib.ticker import FixedLocator, FuncFormatter
        axes[0].xaxis.set_major_locator(FixedLocator([2, 4, 6, 8, 9]))
        axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, p: rf"$10^{{{x:g}}}$"))
        logarithmic_axes(axes[1], "Median interval length")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.10))
        fig.tight_layout(rect=(0, .05, 1, 1))
        save_figure(plt, fig, directory, f"cs_gamma{gamma}")

