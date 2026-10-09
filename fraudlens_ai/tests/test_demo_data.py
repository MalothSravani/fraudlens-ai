import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import FEATURES, make_demo_data


def test_demo_data_schema_and_labels():
    df = make_demo_data(n=1200, seed=7)
    assert set(FEATURES).issubset(df.columns)
    assert "is_fraud" in df.columns
    assert set(df["is_fraud"].unique()).issubset({0, 1})
    assert df["is_fraud"].nunique() == 2
    assert df["amount"].min() > 0


def test_demo_data_is_reproducible():
    a = make_demo_data(n=300, seed=123)
    b = make_demo_data(n=300, seed=123)
    pd.testing.assert_frame_equal(a, b)
