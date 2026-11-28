"""Data processing for the Fake Reviews Detector (Student 1: Data Processing).

Workflow: Load -> Validate -> Clean -> Explore -> Visualize -> Transform -> Export -> Summary

Dataset: Fake Reviews Dataset by mexwell (Kaggle), CC BY 4.0
         https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset
         Download it and save it as  data/fake_reviews_dataset.csv

Run the whole workflow:
    python data_processing.py
    python data_processing.py --input data/sample_reviews.csv      (small example file)
    python data_processing.py --config config.json                 (optional JSON overrides)

Use it from other files (for example train_model.py):
    from data_processing import load_clean_data
    df = load_clean_data()   # columns: category, rating, label, label_name, is_fake, text, char_count, word_count

Use it from the Streamlit app:
    from data_processing import load_data, validate_data, clean_text_series, clean_data
"""
import argparse
import copy
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- configuration
DEFAULT_CONFIG = {
    "paths": {
        "raw_data": "data/fake_reviews_dataset.csv",
        "processed_data": "data/reviews_clean.csv",
        "figures_dir": "outputs/figures",
        "reports_dir": "outputs/reports",
        "log_file": "logs/pipeline.log",
    },
    "data": {
        "required_columns": ["category", "rating", "label", "text_"],
        "text_column": "text_",
        "label_column": "label",
        "rating_column": "rating",
        "category_column": "category",
        # CG = computer-generated (fake), OR = original (real)
        "valid_labels": {"CG": "Fake", "OR": "Real"},
        "fake_label": "CG",
        "rating_range": [1, 5],
    },
    "cleaning": {
        "strip_html": True,
        "normalize_whitespace": True,
        "min_text_chars": 10,
        "drop_duplicate_texts": True,
    },
    "eda": {"top_words": 15},
    "logging": {"level": "INFO"},
    "source": {
        "name": "Fake Reviews Dataset",
        "author": "mexwell (Kaggle)",
        "url": "https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset",
        "license": "Attribution 4.0 International (CC BY 4.0)",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "description": (
            "Generated fake reviews dataset containing about 20k computer-generated (CG) "
            "reviews and 20k original (OR) real product reviews across 10 product categories."
        ),
    },
}

CLASS_ORDER = ["Real", "Fake"]
OUTPUT_COLUMNS = [
    "category", "rating", "label", "label_name", "is_fake",
    "text", "char_count", "word_count",
]
HTML_TAG_PATTERN = r"<[^>]+>"


class ConfigError(Exception):
    """Raised when a configuration file or value is invalid."""


class DataValidationError(ValueError):
    """Raised when the data has problems that make processing impossible."""


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def get_config(config=None) -> dict:
    """Return the full configuration.

    ``config`` can be None (defaults), a dict, or the path of a JSON file. Its values
    override the defaults. Relative paths are resolved against this file's folder.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config is not None:
        if isinstance(config, (str, Path)):
            path = Path(config)
            if not path.is_file():
                raise ConfigError(f"Config file not found: {path}")
            try:
                updates = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ConfigError(f"Config file is not valid JSON: {exc}") from exc
        else:
            updates = config
        if not isinstance(updates, dict):
            raise ConfigError("Configuration must be a JSON object / dict.")
        _deep_update(cfg, updates)

    for key, value in cfg["paths"].items():
        candidate = Path(value)
        cfg["paths"][key] = candidate if candidate.is_absolute() else BASE_DIR / candidate
    cfg["resolved"] = True
    return cfg


def _ensure_config(cfg) -> dict:
    if isinstance(cfg, dict) and cfg.get("resolved"):
        return cfg
    return get_config(cfg)


def setup_logging(level="INFO", log_file=None) -> logging.Logger:
    """Console + optional file logging. Called by the command line entry point only."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    return logger


# --------------------------------------------------------------------------- 1. load
def load_data(source, encoding="utf-8") -> pd.DataFrame:
    """Load a CSV from a file path or a file-like object (e.g. a Streamlit upload).

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the file is not a CSV, is empty, or cannot be parsed.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Only .csv files are supported, got '{path.suffix}'.")

    try:
        try:
            df = pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError:
            logger.warning("Could not decode file as %s, retrying with latin-1.", encoding)
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, encoding="latin-1")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"The CSV file could not be parsed: {exc}") from exc

    logger.info("Loaded %d rows and %d columns.", len(df), df.shape[1])
    return df


# --------------------------------------------------------------------------- 2. validate
@dataclass
class ValidationReport:
    """Result of :func:`validate_data`.

    ``errors`` stop the pipeline; ``warnings`` are problems the cleaning step fixes.
    """

    passed: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_data(df, cfg=None, required_columns=None, check_labels=True) -> ValidationReport:
    """Validate a DataFrame without changing it.

    Args:
        df: the loaded data.
        cfg: configuration (defaults are used when omitted).
        required_columns: columns that must exist (defaults to the config list). Use a
            shorter list to validate an uploaded file that has no label column.
        check_labels: also check the label column values (when the column exists).
    """
    cfg = _ensure_config(cfg)
    data_cfg = cfg["data"]
    required = (
        list(required_columns) if required_columns is not None
        else list(data_cfg["required_columns"])
    )
    text_col = data_cfg["text_column"]
    label_col = data_cfg["label_column"]
    rating_col = data_cfg["rating_column"]

    report = ValidationReport()

    if not isinstance(df, pd.DataFrame):
        report.errors.append("Input is not a table (DataFrame).")
        report.passed = False
        return report

    report.stats["rows"] = int(len(df))
    report.stats["columns"] = [str(c) for c in df.columns]

    if df.empty:
        report.errors.append("The dataset has no rows.")

    missing = [c for c in required if c not in df.columns]
    if missing:
        report.errors.append(f"Missing required columns: {', '.join(missing)}")

    if df.empty:
        report.passed = False
        return report

    present = [c for c in required if c in df.columns]
    null_counts = {c: int(n) for c, n in df[present].isna().sum().items()}
    report.stats["null_counts"] = null_counts
    for column, count in null_counts.items():
        if count:
            report.warnings.append(f"Column '{column}' has {count} missing values.")

    report.stats["duplicate_rows"] = int(df.duplicated().sum())
    if report.stats["duplicate_rows"]:
        report.warnings.append(f"{report.stats['duplicate_rows']} fully duplicated rows.")

    if text_col in df.columns:
        texts = df[text_col].dropna().astype(str)
        blank = int((texts.str.strip() == "").sum())
        duplicated_texts = int(df[text_col].dropna().duplicated().sum())
        report.stats["blank_texts"] = blank
        report.stats["duplicate_texts"] = duplicated_texts
        if blank:
            report.warnings.append(f"{blank} blank review texts.")
        if duplicated_texts:
            report.warnings.append(f"{duplicated_texts} repeated review texts.")

    if check_labels and label_col in df.columns:
        valid = {code.upper() for code in data_cfg["valid_labels"]}
        labels = df[label_col].dropna().astype(str).str.strip().str.upper()
        report.stats["label_counts"] = {str(k): int(v) for k, v in labels.value_counts().items()}
        invalid = int((~labels.isin(valid)).sum())
        report.stats["invalid_labels"] = invalid
        if invalid:
            report.warnings.append(f"{invalid} rows have labels outside {sorted(valid)}.")
        if labels.isin(valid).sum() == 0:
            report.errors.append("No row has a valid label value.")

    if rating_col in df.columns:
        ratings = pd.to_numeric(df[rating_col], errors="coerce")
        non_numeric = int((ratings.isna() & df[rating_col].notna()).sum())
        low, high = data_cfg["rating_range"]
        out_of_range = int(((ratings < low) | (ratings > high)).sum())
        report.stats["non_numeric_ratings"] = non_numeric
        report.stats["out_of_range_ratings"] = out_of_range
        if non_numeric:
            report.warnings.append(f"{non_numeric} ratings are not numeric.")
        if out_of_range:
            report.warnings.append(f"{out_of_range} ratings are outside {low}-{high}.")

    report.passed = not report.errors
    logger.info(
        "Validation %s: %d errors, %d warnings.",
        "passed" if report.passed else "FAILED",
        len(report.errors),
        len(report.warnings),
    )
    return report


def require_valid(report: ValidationReport) -> None:
    """Raise :class:`DataValidationError` if the report contains errors."""
    if not report.passed:
        raise DataValidationError("; ".join(report.errors))


# --------------------------------------------------------------------------- 3. clean
def clean_text_series(series: pd.Series, strip_html=True, normalize_whitespace=True) -> pd.Series:
    """Remove HTML tags and collapse whitespace in a text column (no missing values allowed)."""
    cleaned = series.astype(str)
    if strip_html:
        cleaned = cleaned.str.replace(HTML_TAG_PATTERN, " ", regex=True)
    if normalize_whitespace:
        cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    return cleaned.str.strip()


def clean_data(df: pd.DataFrame, cfg=None):
    """Clean the dataset and return ``(clean_df, report)``.

    Steps, in order: drop missing text/label, normalise and validate labels, validate
    ratings, clean text, drop very short texts, drop texts that appear with conflicting
    labels, drop duplicate texts. ``report`` counts the rows removed by each step.
    """
    cfg = _ensure_config(cfg)
    data_cfg = cfg["data"]
    clean_cfg = cfg["cleaning"]
    text_col = data_cfg["text_column"]
    label_col = data_cfg["label_column"]
    rating_col = data_cfg["rating_column"]

    out = df.copy()
    report = {"rows_before": int(len(out))}

    # 1. Missing text or label.
    n = len(out)
    out = out.dropna(subset=[text_col, label_col])
    report["removed_missing_text_or_label"] = n - len(out)

    # 2. Labels: trim, upper-case, keep only known codes.
    out[label_col] = out[label_col].astype(str).str.strip().str.upper()
    valid = {code.upper() for code in data_cfg["valid_labels"]}
    n = len(out)
    out = out[out[label_col].isin(valid)]
    report["removed_invalid_label"] = n - len(out)

    # 3. Ratings (only when the column exists, so uploaded files without ratings still work).
    if rating_col in out.columns:
        out[rating_col] = pd.to_numeric(out[rating_col], errors="coerce")
        low, high = data_cfg["rating_range"]
        n = len(out)
        out = out[out[rating_col].between(low, high)]
        report["removed_invalid_rating"] = n - len(out)
    else:
        report["removed_invalid_rating"] = 0

    # 4. Text cleaning.
    before = out[text_col].astype(str)
    cleaned = clean_text_series(
        out[text_col],
        strip_html=clean_cfg.get("strip_html", True),
        normalize_whitespace=clean_cfg.get("normalize_whitespace", True),
    )
    report["rows_with_html"] = int(before.str.contains(HTML_TAG_PATTERN, regex=True).sum())
    report["rows_text_modified"] = int((cleaned != before).sum())
    out[text_col] = cleaned

    # 5. Very short texts (also catches texts that were only HTML).
    n = len(out)
    out = out[out[text_col].str.len() >= clean_cfg.get("min_text_chars", 1)]
    report["removed_too_short"] = n - len(out)

    # 6. Identical texts with different labels are ambiguous: remove all copies.
    labels_per_text = out.groupby(text_col)[label_col].nunique()
    conflicting = set(labels_per_text[labels_per_text > 1].index)
    n = len(out)
    out = out[~out[text_col].isin(conflicting)]
    report["removed_conflicting_labels"] = n - len(out)

    # 7. Duplicates.
    n = len(out)
    if clean_cfg.get("drop_duplicate_texts", True):
        out = out.drop_duplicates(subset=[text_col], keep="first")
    else:
        out = out.drop_duplicates(keep="first")
    report["removed_duplicates"] = n - len(out)

    out = out.reset_index(drop=True)
    report["rows_after"] = int(len(out))
    report["rows_removed_total"] = report["rows_before"] - report["rows_after"]

    logger.info(
        "Cleaning finished: %d -> %d rows (%d removed).",
        report["rows_before"], report["rows_after"], report["rows_removed_total"],
    )
    return out, report


# --------------------------------------------------------------------------- 4. explore
_TOKEN_PATTERN = re.compile(r"[a-z]{3,}")
_EXTRA_STOP_WORDS = {
    "don", "didn", "doesn", "isn", "wasn", "aren", "couldn", "wouldn",
    "shouldn", "won", "ve", "ll",
}
_STOP_WORDS = set(ENGLISH_STOP_WORDS) | _EXTRA_STOP_WORDS


def label_name_series(df: pd.DataFrame, cfg=None) -> pd.Series:
    """Map label codes (CG/OR) to readable names (Fake/Real)."""
    cfg = _ensure_config(cfg)
    data_cfg = cfg["data"]
    mapping = {code.upper(): name for code, name in data_cfg["valid_labels"].items()}
    return df[data_cfg["label_column"]].astype(str).str.upper().map(mapping)


def top_words(texts, n=15):
    """Return the ``n`` most common non-stop-words as ``[[word, count], ...]``."""
    counter = Counter()
    for text in texts:
        counter.update(
            word for word in _TOKEN_PATTERN.findall(str(text).lower())
            if word not in _STOP_WORDS
        )
    return [[word, int(count)] for word, count in counter.most_common(n)]


def compute_eda_summary(df: pd.DataFrame, cfg=None) -> dict:
    """Compute class balance, rating, length, category and word statistics."""
    cfg = _ensure_config(cfg)
    data_cfg = cfg["data"]
    text_col = data_cfg["text_column"]
    rating_col = data_cfg["rating_column"]
    category_col = data_cfg.get("category_column", "category")
    n_top = cfg.get("eda", {}).get("top_words", 15)

    names = label_name_series(df, cfg)
    classes = [c for c in CLASS_ORDER if c in set(names)]
    words = df[text_col].str.split().str.len()
    chars = df[text_col].str.len()

    summary = {"rows": int(len(df)), "columns": [str(c) for c in df.columns]}

    counts = names.value_counts()
    summary["class_counts"] = {c: int(counts.get(c, 0)) for c in classes}
    summary["class_share_pct"] = {
        c: round(100 * summary["class_counts"][c] / len(df), 2) for c in classes
    }

    if rating_col in df.columns:
        share = (pd.crosstab(names, df[rating_col].astype(int), normalize="index") * 100).round(2)
        summary["rating_share_pct"] = {
            c: {int(r): float(share.loc[c, r]) for r in share.columns} for c in classes
        }
        summary["rating_mean"] = {
            c: round(float(df.loc[names == c, rating_col].mean()), 3) for c in classes
        }

    summary["length"] = {
        c: {
            "words_mean": round(float(words[names == c].mean()), 2),
            "words_median": float(words[names == c].median()),
            "words_p95": float(words[names == c].quantile(0.95)),
            "chars_mean": round(float(chars[names == c].mean()), 2),
            "chars_median": float(chars[names == c].median()),
        }
        for c in classes
    }

    if category_col in df.columns:
        table = pd.crosstab(df[category_col], names)
        summary["category_counts"] = {
            str(cat): {c: int(table.loc[cat, c]) if c in table.columns else 0 for c in classes}
            for cat in table.index
        }

    summary["top_words"] = {c: top_words(df.loc[names == c, text_col], n_top) for c in classes}

    logger.info("EDA summary computed for %d rows.", len(df))
    return summary


# --------------------------------------------------------------------------- 5. visualize
# Real = blue, Fake = orange (colour-blind-safe pair). Charts show a legend or panel
# titles, so colour is never the only cue. Figures use matplotlib's Figure class directly,
# so they work in scripts, tests and Streamlit (st.pyplot(fig)) without a display.
COLORS = {"Real": "#257de0", "Fake": "#f47230"}
INK = "#191919"
INK_2 = "#666666"
GRID = "#e7e7e7"
AXIS = "#c2c2c2"


def _style_axes(ax, grid_axis="y"):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _title(ax, text):
    ax.set_title(text, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)


def _legend(ax, **kwargs):
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, **kwargs)


def _thousands(ax, axis="y"):
    formatter = FuncFormatter(lambda value, _: f"{value:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def _pretty_category(name: str) -> str:
    return re.sub(r"_5$", "", str(name)).replace("_", " ")


def save_figure(fig: Figure, path) -> Path:
    """Save a figure as PNG, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info("Saved figure %s", path.name)
    return path


def plot_class_balance(eda: dict) -> Figure:
    classes = list(eda["class_counts"])
    counts = [eda["class_counts"][c] for c in classes]
    shares = [eda["class_share_pct"][c] for c in classes]

    fig = Figure(figsize=(6, 4.2))
    ax = fig.subplots()
    bars = ax.bar(classes, counts, width=0.5, color=[COLORS[c] for c in classes],
                  edgecolor="white", linewidth=1)
    for bar, count, share in zip(bars, counts, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{count:,}\n({share:.1f}%)", ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_ylim(0, max(counts) * 1.2)
    ax.set_ylabel("Number of reviews")
    _style_axes(ax)
    _thousands(ax)
    _title(ax, "Class balance: real vs. fake reviews")
    return fig


def plot_rating_distribution(eda: dict) -> Figure:
    share = eda["rating_share_pct"]
    classes = list(share)
    ratings = sorted(share[classes[0]])
    x = np.arange(len(ratings))
    width = 0.38

    fig = Figure(figsize=(7, 4.4))
    ax = fig.subplots()
    for i, cls in enumerate(classes):
        values = [share[cls][r] for r in ratings]
        bars = ax.bar(x + (i - 0.5) * width, values, width, color=COLORS[cls],
                      edgecolor="white", linewidth=1, label=cls)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:.0f}", ha="center", va="bottom", color=INK_2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ratings])
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Share of the class's reviews (%)")
    ax.set_ylim(0, max(max(v.values()) for v in share.values()) * 1.15)
    _style_axes(ax)
    _legend(ax, loc="upper left")
    _title(ax, "Rating distribution by class")
    return fig


def plot_text_length(df: pd.DataFrame, cfg=None) -> Figure:
    cfg = _ensure_config(cfg)
    text_col = cfg["data"]["text_column"]
    names = label_name_series(df, cfg)
    words = df[text_col].str.split().str.len()
    cap = words.quantile(0.99)
    bins = np.linspace(0, cap, 41)

    fig = Figure(figsize=(7, 4.4))
    ax = fig.subplots()
    for cls in [c for c in CLASS_ORDER if c in set(names)]:
        class_words = words[names == cls]
        median = class_words.median()
        shown = class_words[class_words <= cap]
        ax.hist(shown, bins=bins, density=True, histtype="stepfilled",
                alpha=0.25, color=COLORS[cls])
        ax.hist(shown, bins=bins, density=True, histtype="step", linewidth=2,
                color=COLORS[cls], label=f"{cls} (median {median:.0f} words)")
        ax.axvline(median, color=COLORS[cls], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Words per review (reviews above the 99th percentile not shown)")
    ax.set_ylabel("Density")
    _style_axes(ax)
    _legend(ax, loc="upper right")
    _title(ax, "Review length by class")
    return fig


def plot_category_distribution(eda: dict) -> Figure:
    categories = eda["category_counts"]
    classes = list(eda["class_counts"])
    order = sorted(categories, key=lambda c: sum(categories[c].values()))  # largest on top
    y = np.arange(len(order))
    height = 0.38

    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    for i, cls in enumerate(classes):
        values = [categories[cat][cls] for cat in order]
        ax.barh(y + (i - 0.5) * height, values, height, color=COLORS[cls],
                edgecolor="white", linewidth=1, label=cls)
    ax.set_yticks(y)
    ax.set_yticklabels([_pretty_category(c) for c in order])
    ax.set_xlabel("Number of reviews")
    _style_axes(ax, grid_axis="x")
    _thousands(ax, axis="x")
    _legend(ax, loc="lower right")
    _title(ax, "Reviews per product category")
    return fig


def plot_top_words(eda: dict) -> Figure:
    classes = list(eda["top_words"])
    fig = Figure(figsize=(10, 5.4))
    axes = np.atleast_1d(fig.subplots(1, len(classes), sharex=True))
    for ax, cls in zip(axes, classes):
        pairs = eda["top_words"][cls][::-1]
        ax.barh([w for w, _ in pairs], [n for _, n in pairs], color=COLORS[cls],
                edgecolor="white", linewidth=1)
        ax.set_xlabel("Occurrences")
        _style_axes(ax, grid_axis="x")
        _thousands(ax, axis="x")
        ax.set_title(f"{cls} reviews", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.suptitle("Most frequent words (stop-words removed)", x=0.01, ha="left",
                 color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def create_all_figures(df: pd.DataFrame, eda: dict, cfg=None) -> dict:
    """Create and save every chart. Returns ``{name: path}``."""
    cfg = _ensure_config(cfg)
    folder = Path(cfg["paths"]["figures_dir"])

    figures = {"class_balance": save_figure(plot_class_balance(eda), folder / "class_balance.png")}
    if "rating_share_pct" in eda:
        figures["rating_distribution"] = save_figure(
            plot_rating_distribution(eda), folder / "rating_distribution.png")
    figures["text_length"] = save_figure(plot_text_length(df, cfg), folder / "text_length.png")
    if "category_counts" in eda:
        figures["category_distribution"] = save_figure(
            plot_category_distribution(eda), folder / "category_distribution.png")
    figures["top_words"] = save_figure(plot_top_words(eda), folder / "top_words.png")
    return figures


# --------------------------------------------------------------------------- 6. transform
def transform_data(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """Return a model-ready table.

    - renames the text column to ``text``
    - adds ``label_name`` (Real/Fake) and ``is_fake`` (1 = fake, 0 = real)
    - adds ``char_count`` and ``word_count``
    """
    cfg = _ensure_config(cfg)
    data_cfg = cfg["data"]
    text_col = data_cfg["text_column"]
    label_col = data_cfg["label_column"]
    fake_code = str(data_cfg.get("fake_label", "CG")).upper()

    out = df.copy()
    out["label_name"] = label_name_series(out, cfg)
    out["is_fake"] = (out[label_col].astype(str).str.upper() == fake_code).astype(int)
    out = out.rename(columns={text_col: "text"})
    out["char_count"] = out["text"].str.len()
    out["word_count"] = out["text"].str.split().str.len()

    ordered = [c for c in OUTPUT_COLUMNS if c in out.columns]
    extra = [c for c in out.columns if c not in ordered]
    out = out[ordered + extra]

    logger.info("Transformed data: columns = %s", list(out.columns))
    return out


# --------------------------------------------------------------------------- 7-8. export + summary
def export_dataframe(df: pd.DataFrame, path) -> Path:
    """Save a DataFrame as UTF-8 CSV, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Exported %d rows to %s", len(df), path)
    return path


def _to_builtin(obj):
    """JSON fallback for numpy numbers, paths and sets."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Cannot serialise {type(obj).__name__}")


def export_json(data: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=_to_builtin)
    logger.info("Wrote %s", path)
    return path


def export_text(text: str, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def build_markdown_summary(summary: dict) -> str:
    """Turn the summary dictionary into a readable Markdown report."""
    src = summary["source"]
    val = summary["validation"]
    clean = summary["cleaning"]
    eda = summary["eda"]

    lines = [
        "# Data Processing Summary",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Data source",
        f"- **Name:** {src['name']}",
        f"- **Author:** {src['author']}",
        f"- **URL:** {src['url']}",
        f"- **License:** {src['license']} ({src['license_url']})",
        f"- **Description:** {' '.join(str(src['description']).split())}",
        "",
        "## Validation",
        f"- Passed: {'yes' if val['passed'] else 'no'}",
        f"- Rows loaded: {val['stats'].get('rows')}",
    ]
    for warning in val["warnings"]:
        lines.append(f"- Warning: {warning}")

    lines += [
        "",
        "## Cleaning",
        "| Step | Rows affected |",
        "|---|---|",
        f"| Rows before cleaning | {clean['rows_before']:,} |",
        f"| Removed: missing text or label | {clean['removed_missing_text_or_label']:,} |",
        f"| Removed: invalid label | {clean['removed_invalid_label']:,} |",
        f"| Removed: invalid rating | {clean['removed_invalid_rating']:,} |",
        f"| Text modified (HTML / whitespace) | {clean['rows_text_modified']:,} |",
        f"| Of which contained HTML tags | {clean['rows_with_html']:,} |",
        f"| Removed: too short | {clean['removed_too_short']:,} |",
        f"| Removed: same text, conflicting labels | {clean['removed_conflicting_labels']:,} |",
        f"| Removed: duplicates | {clean['removed_duplicates']:,} |",
        f"| **Rows after cleaning** | **{clean['rows_after']:,}** |",
        "",
        "## Exploration",
        "| Class | Reviews | Share (%) | Mean rating | Median words | Mean words |",
        "|---|---|---|---|---|---|",
    ]
    for cls, count in eda["class_counts"].items():
        length = eda["length"][cls]
        rating = eda.get("rating_mean", {}).get(cls, "-")
        lines.append(
            f"| {cls} | {count:,} | {eda['class_share_pct'][cls]} | {rating} | "
            f"{length['words_median']:.0f} | {length['words_mean']} |"
        )

    lines += ["", "## Figures"]
    for name, path in summary["figures"].items():
        lines.append(f"- {name}: `{path}`")

    out = summary["outputs"]
    lines += [
        "",
        "## Output",
        f"- Clean file: `{out['clean_csv']}` ({out['rows']:,} rows)",
        f"- Columns: {', '.join(out['columns'])}",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- pipeline
def _relative(path, root) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def run_pipeline(config=None, input_path=None) -> dict:
    """Run every step and return the summary dictionary.

    Args:
        config: None, a dict, or a JSON file path (see :func:`get_config`).
        input_path: optional CSV that replaces ``paths.raw_data``.

    Raises the underlying exception (after logging it) if a step fails.
    """
    cfg = get_config(config)
    source_path = Path(input_path) if input_path is not None else cfg["paths"]["raw_data"]

    try:
        logger.info("Step 1/8 - Load")
        df_raw = load_data(source_path)

        logger.info("Step 2/8 - Validate")
        report = validate_data(df_raw, cfg)
        require_valid(report)

        logger.info("Step 3/8 - Clean")
        df_clean, clean_report = clean_data(df_raw, cfg)

        logger.info("Step 4/8 - Explore")
        eda = compute_eda_summary(df_clean, cfg)

        logger.info("Step 5/8 - Visualize")
        figures = create_all_figures(df_clean, eda, cfg)

        logger.info("Step 6/8 - Transform")
        df_final = transform_data(df_clean, cfg)

        logger.info("Step 7/8 - Export")
        clean_path = export_dataframe(df_final, cfg["paths"]["processed_data"])

        logger.info("Step 8/8 - Summary")
        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": cfg["source"],
            "validation": report.to_dict(),
            "cleaning": clean_report,
            "eda": eda,
            "figures": {name: _relative(path, BASE_DIR) for name, path in figures.items()},
            "outputs": {
                "clean_csv": _relative(clean_path, BASE_DIR),
                "rows": int(len(df_final)),
                "columns": list(df_final.columns),
            },
        }
        reports_dir = Path(cfg["paths"]["reports_dir"])
        export_json(summary, reports_dir / "data_summary.json")
        export_text(build_markdown_summary(summary), reports_dir / "data_summary.md")
    except Exception:
        logger.exception("Pipeline failed.")
        raise

    logger.info("Pipeline finished successfully.")
    return summary


def load_clean_data(path=None, cfg=None) -> pd.DataFrame:
    """Convenience for other parts of the project: raw CSV -> validated, cleaned, model-ready table."""
    cfg = _ensure_config(cfg)
    df = load_data(path if path is not None else cfg["paths"]["raw_data"])
    require_valid(validate_data(df, cfg))
    clean_df, _ = clean_data(df, cfg)
    return transform_data(clean_df, cfg)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the fake-reviews data processing workflow.")
    parser.add_argument("--input", help="CSV file to process (default: data/fake_reviews_dataset.csv).")
    parser.add_argument("--config", help="Optional JSON file that overrides the default settings.")
    args = parser.parse_args(argv)

    try:
        cfg = get_config(args.config)
        setup_logging(cfg["logging"].get("level", "INFO"), cfg["paths"].get("log_file"))
        run_pipeline(cfg, input_path=args.input)
    except (ConfigError, DataValidationError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
