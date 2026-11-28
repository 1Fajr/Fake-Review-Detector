"""Tests for data_processing.py (run from the project folder with: pytest)."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_processing as dp  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    """Default settings, but every input/output path points inside a temporary folder."""
    return dp.get_config({
        "paths": {
            "raw_data": str(tmp_path / "raw.csv"),
            "processed_data": str(tmp_path / "out" / "clean.csv"),
            "figures_dir": str(tmp_path / "out" / "figures"),
            "reports_dir": str(tmp_path / "out" / "reports"),
            "log_file": str(tmp_path / "out" / "pipeline.log"),
        },
        "eda": {"top_words": 10},
    })


@pytest.fixture
def raw_df():
    """40 good rows + 8 edge-case rows (7 are removed by cleaning, 1 is kept after fixing)."""
    rows = [
        (
            "Books_5" if i % 2 == 0 else "Toys_and_Games_5",
            float(i % 5 + 1),
            "OR" if i % 2 == 0 else "CG",
            f"Review number {i} about the product, it works well and I like it",
        )
        for i in range(40)
    ]
    rows += [
        ("Books_5", 5.0, "OR", None),                                          # missing text
        ("Books_5", 5.0, "OR", "Review number 0 about the product, it works well and I like it"),  # duplicate
        ("Books_5", 4.0, "OR", "<div class='x'>Nice product   with\n spaces</div>  ok"),  # HTML (kept, fixed)
        ("Books_5", 4.0, "XX", "Row with an unknown label value here"),        # bad label
        ("Books_5", 9.0, "OR", "Row with an impossible rating value here"),    # bad rating
        ("Books_5", 3.0, "OR", "ok"),                                          # too short
        ("Books_5", 3.0, "OR", "Same text but different labels here"),         # conflict
        ("Books_5", 3.0, "CG", "Same text but different labels here"),         # conflict
    ]
    return pd.DataFrame(rows, columns=["category", "rating", "label", "text_"])


# ---------------------------------------------------------------- configuration
def test_get_config_resolves_relative_paths():
    cfg = dp.get_config()
    assert cfg["paths"]["raw_data"] == dp.BASE_DIR / "data" / "fake_reviews_dataset.csv"


def test_get_config_dict_overrides_defaults():
    cfg = dp.get_config({"cleaning": {"min_text_chars": 25}})
    assert cfg["cleaning"]["min_text_chars"] == 25
    assert cfg["cleaning"]["strip_html"] is True  # untouched default is kept


def test_get_config_from_json_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"eda": {"top_words": 5}}), encoding="utf-8")
    assert dp.get_config(path)["eda"]["top_words"] == 5


def test_get_config_missing_or_invalid_file(tmp_path):
    with pytest.raises(dp.ConfigError):
        dp.get_config(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(dp.ConfigError):
        dp.get_config(bad)


# ---------------------------------------------------------------- loading
def test_load_data_reads_csv(tmp_path, raw_df):
    path = tmp_path / "data.csv"
    raw_df.to_csv(path, index=False)
    loaded = dp.load_data(path)
    assert loaded.shape == raw_df.shape
    assert list(loaded.columns) == list(raw_df.columns)


def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        dp.load_data(tmp_path / "missing.csv")


def test_load_data_rejects_non_csv(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        dp.load_data(path)


def test_load_data_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        dp.load_data(path)


def test_load_data_accepts_file_object(raw_df):
    buffer = io.BytesIO(raw_df.to_csv(index=False).encode("utf-8"))
    assert len(dp.load_data(buffer)) == len(raw_df)


# ---------------------------------------------------------------- validation
def test_validate_reports_warnings_for_fixable_problems(cfg, raw_df):
    report = dp.validate_data(raw_df, cfg)
    assert report.passed
    assert report.stats["null_counts"]["text_"] == 1
    assert report.stats["invalid_labels"] == 1
    assert report.stats["out_of_range_ratings"] == 1
    assert report.warnings


def test_validate_missing_column_fails(cfg, raw_df):
    report = dp.validate_data(raw_df.drop(columns=["label"]), cfg)
    assert not report.passed
    with pytest.raises(dp.DataValidationError):
        dp.require_valid(report)


def test_validate_empty_dataframe_fails(cfg):
    report = dp.validate_data(pd.DataFrame(columns=["category", "rating", "label", "text_"]), cfg)
    assert not report.passed


def test_validate_upload_without_label_column(cfg):
    uploaded = pd.DataFrame({"text_": ["a perfectly fine review text"]})
    report = dp.validate_data(uploaded, cfg, required_columns=["text_"], check_labels=False)
    assert report.passed


def test_validate_all_labels_invalid_is_an_error(cfg, raw_df):
    raw_df["label"] = "XX"
    assert not dp.validate_data(raw_df, cfg).passed


def test_validate_works_without_explicit_config(raw_df):
    assert dp.validate_data(raw_df).passed


# ---------------------------------------------------------------- cleaning
def test_clean_text_series_removes_html_and_extra_spaces():
    series = pd.Series(["<b>Great</b>   product \n here  ", "plain text"])
    assert dp.clean_text_series(series).tolist() == ["Great product here", "plain text"]


def test_clean_data_removes_bad_rows_and_counts_them(cfg, raw_df):
    cleaned, report = dp.clean_data(raw_df, cfg)
    assert report["rows_before"] == 48
    assert report["removed_missing_text_or_label"] == 1
    assert report["removed_invalid_label"] == 1
    assert report["removed_invalid_rating"] == 1
    assert report["removed_too_short"] == 1
    assert report["removed_conflicting_labels"] == 2
    assert report["removed_duplicates"] == 1
    assert report["rows_with_html"] == 1
    assert report["rows_after"] == len(cleaned) == 41
    assert report["rows_removed_total"] == 7


def test_clean_data_output_is_consistent(cfg, raw_df):
    cleaned, _ = dp.clean_data(raw_df, cfg)
    assert cleaned["text_"].notna().all()
    assert set(cleaned["label"]) <= {"CG", "OR"}
    assert cleaned["rating"].between(1, 5).all()
    assert not cleaned["text_"].duplicated().any()
    assert not cleaned["text_"].str.contains("<|\n|  ", regex=True).any()
    assert "Nice product with spaces ok" in cleaned["text_"].tolist()


def test_clean_data_does_not_modify_input(cfg, raw_df):
    before = raw_df.copy()
    dp.clean_data(raw_df, cfg)
    pd.testing.assert_frame_equal(raw_df, before)


def test_clean_data_works_without_rating_column(cfg, raw_df):
    cleaned, report = dp.clean_data(raw_df.drop(columns=["rating"]), cfg)
    assert report["removed_invalid_rating"] == 0
    assert "rating" not in cleaned.columns


# ---------------------------------------------------------------- transform
def test_transform_adds_model_columns(cfg, raw_df):
    cleaned, _ = dp.clean_data(raw_df, cfg)
    final = dp.transform_data(cleaned, cfg)
    assert "text" in final.columns and "text_" not in final.columns
    assert set(final["is_fake"]) == {0, 1}
    assert (final.loc[final["label"] == "CG", "is_fake"] == 1).all()
    assert (final.loc[final["label"] == "CG", "label_name"] == "Fake").all()
    assert (final["word_count"] > 0).all()
    assert (final["char_count"] == final["text"].str.len()).all()


# ---------------------------------------------------------------- EDA and charts
def test_top_words_ignores_stop_words():
    words = [w for w, _ in dp.top_words(["The product is great and the product works"], n=5)]
    assert "the" not in words and "and" not in words
    assert words[0] == "product"


def test_eda_summary_is_consistent(cfg, raw_df):
    cleaned, _ = dp.clean_data(raw_df, cfg)
    eda = dp.compute_eda_summary(cleaned, cfg)
    assert sum(eda["class_counts"].values()) == len(cleaned)
    assert sum(eda["class_share_pct"].values()) == pytest.approx(100, abs=0.1)
    assert set(eda["class_counts"]) == {"Real", "Fake"}
    assert all(len(words) <= 10 for words in eda["top_words"].values())
    assert sum(sum(v.values()) for v in eda["category_counts"].values()) == len(cleaned)


def test_plot_functions_return_figures(cfg, raw_df):
    cleaned, _ = dp.clean_data(raw_df, cfg)
    eda = dp.compute_eda_summary(cleaned, cfg)
    figures = [
        dp.plot_class_balance(eda),
        dp.plot_rating_distribution(eda),
        dp.plot_text_length(cleaned, cfg),
        dp.plot_category_distribution(eda),
        dp.plot_top_words(eda),
    ]
    assert all(isinstance(fig, Figure) for fig in figures)


# ---------------------------------------------------------------- export
def test_export_dataframe_roundtrip(tmp_path, raw_df):
    path = dp.export_dataframe(raw_df, tmp_path / "nested" / "file.csv")
    assert pd.read_csv(path).shape == raw_df.shape


def test_export_json_handles_numpy_and_paths(tmp_path):
    path = dp.export_json(
        {"n": np.int64(3), "x": np.float64(1.5), "p": tmp_path}, tmp_path / "a.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["n"] == 3 and data["x"] == 1.5 and data["p"] == str(tmp_path)


# ---------------------------------------------------------------- end to end
def test_pipeline_end_to_end(cfg, raw_df):
    raw_df.to_csv(cfg["paths"]["raw_data"], index=False)
    summary = dp.run_pipeline(cfg)

    out = Path(cfg["paths"]["processed_data"]).parent
    assert Path(cfg["paths"]["processed_data"]).is_file()
    assert (out / "reports" / "data_summary.json").is_file()
    assert (out / "reports" / "data_summary.md").is_file()
    for name in ("class_balance", "rating_distribution", "text_length",
                 "category_distribution", "top_words"):
        assert (out / "figures" / f"{name}.png").stat().st_size > 0

    final = pd.read_csv(cfg["paths"]["processed_data"])
    assert len(final) == summary["cleaning"]["rows_after"] == 41
    assert {"text", "is_fake", "label_name", "word_count"} <= set(final.columns)


def test_pipeline_accepts_input_path(cfg, raw_df, tmp_path):
    other = tmp_path / "other.csv"
    raw_df.to_csv(other, index=False)
    summary = dp.run_pipeline(cfg, input_path=other)
    assert summary["cleaning"]["rows_before"] == 48


def test_pipeline_fails_clearly_when_data_file_is_missing(cfg):
    with pytest.raises(FileNotFoundError):
        dp.run_pipeline(cfg)


def test_load_clean_data_returns_model_ready_table(cfg, raw_df):
    raw_df.to_csv(cfg["paths"]["raw_data"], index=False)
    df = dp.load_clean_data(cfg=cfg)
    assert len(df) == 41
    assert {"text", "is_fake"} <= set(df.columns)


def test_main_returns_error_code_for_missing_file(tmp_path, capsys):
    code = dp.main(["--input", str(tmp_path / "missing.csv")])
    assert code == 1
    assert "ERROR" in capsys.readouterr().err
