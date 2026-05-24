"""Unit tests for the synthetic PHQ-9 / GAD-7 generator."""

from __future__ import annotations

import pytest

from spark_jobs.synthetic_screening import (
    GAD7_ITEMS,
    PHQ9_ITEMS,
    gad7_severity_band,
    generate,
    phq9_severity_band,
)


class TestSeverityBands:
    @pytest.mark.parametrize(
        "total,expected",
        [
            (0, "none"),
            (4, "none"),
            (5, "mild"),
            (9, "mild"),
            (10, "moderate"),
            (14, "moderate"),
            (15, "moderately_severe"),
            (19, "moderately_severe"),
            (20, "severe"),
            (27, "severe"),
        ],
    )
    def test_phq9_bands_at_canonical_cutoffs(self, total, expected):
        assert phq9_severity_band(total) == expected

    @pytest.mark.parametrize(
        "total,expected",
        [
            (0, "minimal"),
            (4, "minimal"),
            (5, "mild"),
            (9, "mild"),
            (10, "moderate"),
            (14, "moderate"),
            (15, "severe"),
            (21, "severe"),
        ],
    )
    def test_gad7_bands_at_canonical_cutoffs(self, total, expected):
        assert gad7_severity_band(total) == expected


class TestGenerator:
    def test_generator_produces_requested_row_count(self, spark):
        df = generate(spark, n=200, seed=7, partitions=2)
        assert df.count() == 200

    def test_phq9_items_are_in_valid_range(self, spark):
        df = generate(spark, n=500, seed=7, partitions=2).toPandas()
        for item in PHQ9_ITEMS:
            assert df[item].between(0, 3).all(), f"{item} out of [0,3] range"

    def test_gad7_items_are_in_valid_range(self, spark):
        df = generate(spark, n=500, seed=7, partitions=2).toPandas()
        for item in GAD7_ITEMS:
            assert df[item].between(0, 3).all(), f"{item} out of [0,3] range"

    def test_phq9_total_matches_sum_of_items(self, spark):
        df = generate(spark, n=500, seed=7, partitions=2).toPandas()
        recomputed = df[PHQ9_ITEMS].sum(axis=1)
        assert (df["phq9_total"] == recomputed).all()

    def test_phq9_band_assigned_consistently_with_total(self, spark):
        df = generate(spark, n=500, seed=7, partitions=2).toPandas()
        for _, row in df.iterrows():
            assert row["phq9_band"] == phq9_severity_band(int(row["phq9_total"]))

    def test_suicidal_ideation_prevalence_correlates_with_severity(self, spark):
        """Item 9 (SI) should be much rarer in the 'none' band than 'severe'."""
        df = generate(spark, n=5000, seed=7, partitions=4).toPandas()
        si_rate_none = (df.loc[df["phq9_band"] == "none", "phq9_suicidal_ideation"] > 0).mean()
        si_rate_severe = (
            df.loc[df["phq9_band"] == "severe", "phq9_suicidal_ideation"] > 0
        ).mean()
        # SI in 'severe' band should be much higher than in 'none' — usually >5x.
        assert si_rate_severe > si_rate_none * 3, (
            f"Expected SI prevalence to be much higher in severe band; "
            f"got none={si_rate_none:.3f} severe={si_rate_severe:.3f}"
        )

    def test_phq9_and_gad7_are_correlated(self, spark):
        """Calibrated correlation should be in the published range (~0.5-0.75)."""
        df = generate(spark, n=5000, seed=7, partitions=4).toPandas()
        corr = df["phq9_total"].corr(df["gad7_total"])
        assert 0.4 < corr < 0.8, f"PHQ-9 / GAD-7 correlation {corr:.3f} outside expected range"

    def test_overall_phq9_geq_10_prevalence_is_plausible(self, spark):
        """NHANES reports ~8% prevalence; our synthetic should be in [3%, 25%]."""
        df = generate(spark, n=5000, seed=7, partitions=4).toPandas()
        prev = (df["phq9_total"] >= 10).mean()
        assert 0.03 < prev < 0.25, f"PHQ-9 >= 10 prevalence {prev:.3f} implausible"

    def test_reproducible_with_same_seed(self, spark):
        df_a = generate(spark, n=200, seed=42, partitions=2).toPandas()
        df_b = generate(spark, n=200, seed=42, partitions=2).toPandas()
        assert (df_a["phq9_total"].values == df_b["phq9_total"].values).all()
