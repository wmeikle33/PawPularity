import pandas as pd

from pawpularity.data import load_csv, save_csv


def test_save_and_load_csv_roundtrip(tmp_path):
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "score": [1, 2, 3],
        }
    )

    out_path = tmp_path / "sample.csv"
    save_csv(df, out_path)

    loaded = load_csv(out_path)

    pd.testing.assert_frame_equal(loaded, df)
