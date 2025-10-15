import sys
sys.path.append('./')
import pandas as pd
import numpy as np
from pyn_utils import DataFrameProcessor


def test_dataframe_processor():
    dfp = DataFrameProcessor()

    # --- Тест from_dict ---
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    df = dfp.from_dict(data)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['A', 'B']
    assert len(df) == 3

    # --- Тест filter_rows ---
    filtered = dfp.filter_rows(df, df['A'] > 1)
    assert len(filtered) == 2
    assert filtered['A'].min() == 2

    # --- Тест add_column ---
    df2 = dfp.add_column(df.copy(), 'C', [10, 20, 30])
    assert 'C' in df2.columns
    assert df2['C'].iloc[0] == 10

    # --- Тест group_and_aggregate ---
    df_grouped = pd.DataFrame({
        'group': ['x', 'x', 'y', 'y'],
        'value': [10, 20, 30, 40]
    })
    grouped = dfp.group_and_aggregate(df_grouped, by='group', agg_func={'value': 'mean'})
    assert 'value' in grouped.columns
    assert np.isclose(grouped.loc['x', 'value'], 15.0)
    assert np.isclose(grouped.loc['y', 'value'], 35.0)

    # --- Тест sort_by_column ---
    unsorted = pd.DataFrame({'A': [3, 1, 2]})
    sorted_df = dfp.sort_by_column(unsorted, 'A')
    assert sorted_df['A'].tolist() == [1, 2, 3]

    # --- Тест summarize ---
    summary = dfp.summarize(df)
    expected_stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    assert all(stat in summary.index for stat in expected_stats) or all(stat in summary.columns for stat in expected_stats)
    assert not summary.empty


if __name__ == "__main__":
    test_dataframe_processor()
    print("✅ Все тесты пройдены успешно!")
