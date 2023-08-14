import unittest
from context import *
import pandas as pd
import numpy as np
import math

class Test_Sync_Lib(unittest.TestCase):

    def setUp(self):
        self.desired_freq = 7
        self.mock_df = pd.DataFrame(data = np.random.rand(10,3), columns = ["A","B","C"])
        self.mock_df["Time"] = pd.date_range(start="2020-01-01 00:00:00", periods= 10 , freq= f"{self.desired_freq}S")
        self.mock_df.at[7,"A"] = np.nan

        # chat gpt fez isso talvez fassa sentido
        self.mock_df_wrong_freq = pd.DataFrame(data = np.random.rand(10,3), columns = ["A","B","C"])
        self.mock_df_wrong_freq["Time"] = pd.date_range(start="2020-01-01 00:00:00", periods= 10 , freq= "4S")

        self.mock_remove_nan_df = self.mock_df.copy()
        self.mock_remove_nan_df.at[6,"A"] = np.nan
        self.mock_remove_nan_df.at[5,"A"] = np.nan
        self.mock_remove_nan_df.at[4,"A"] = np.nan
        self.mock_remove_nan_df.at[3,"A"] = np.nan
        self.mock_remove_nan_df.at[2,"A"] = np.nan

    def test_substitutes_empty_values_by_ones_behind(self):
        df_without_empty_values = substitutes_empty_values_by_ones_behind(self.mock_df)
        self.assertEqual(df_without_empty_values.isna().sum().sum(),0)
        self.assertEqual(df_without_empty_values.shape, self.mock_df.shape)
        self.assertEqual(df_without_empty_values.at[7,"A"], self.mock_df.at[6,"A"])
        self.assertIsInstance(df_without_empty_values, pd.DataFrame)

    def test_ajust_df_frequency_by_mv_avg(self):
        # print(self.mock_df_wrong_freq)
        df_with_correct_freq = ajust_df_frequency_by_mv_avg(self.mock_df_wrong_freq, self.desired_freq)
        for row in df_with_correct_freq.index:
            self.assertEqual(df_with_correct_freq.at[row,"Time_Interval"].second % self.desired_freq, 0)
        for column in df_with_correct_freq.columns:
            if column == "Time_Interval": #it is very unlikely but this values can not be similar
                continue
            is_aproximate = math.isclose(df_with_correct_freq[column].mean(), self.mock_df_wrong_freq[column].mean(), rel_tol=0.1, abs_tol=0.1)

            self.assertTrue(is_aproximate)
        # print(df_with_correct_freq)

    def test_add_prefixes_in_columns_names(self):
        df_with_prefixes = add_prefixes_in_columns_names(self.mock_df, "prefix_", "Time")
        for column in df_with_prefixes.columns:
            if column == "Time":
                continue
            self.assertTrue(column.startswith("prefix_"))
        
    def test_merge_on_comon_col_adding_prefixes_to_col_names(self):
        df_with_prefixes = merge_on_comon_col_adding_prefixes_to_col_names(self.mock_df, self.mock_df, "prefix_1", "prefix_2", "Time")
        for column in df_with_prefixes.columns:
            if column == "Time":
                continue

            self.assertTrue(column.startswith("prefix_"))
        self.assertEqual(df_with_prefixes.shape[1], self.mock_df.shape[1] + self.mock_df.shape[1] - 1)
        
    def test_remove_nan_dfs(self):
        df_without_nan_1, df_without_nan_2 = remove_nan_dfs(self.mock_remove_nan_df.copy(), self.mock_remove_nan_df.copy())
        self.assertEqual(df_without_nan_1.isna().sum().sum(), 0)
        self.assertIsInstance(df_without_nan_1, pd.DataFrame)
        self.assertEqual(df_without_nan_2.isna().sum().sum(), 0)
        self.assertIsInstance(df_without_nan_2, pd.DataFrame)

    def test_generate_eigenvalue_named_columns(self):
        eigenvalue_column_names = generate_eigenvalue_named_columns(2)
        assert eigenvalue_column_names == ["eigenvalue_1", "eigenvalue_2"]

    def test_compensates_drift_with_PCA(self):
        df_compensated = compensates_drift_with_PCA(self.mock_df_wrong_freq.copy())
        self.assertIsInstance(df_compensated, pd.DataFrame)
        self.assertEqual(df_compensated.shape, self.mock_df_wrong_freq.shape)

if __name__ == '__main__':
    unittest.main()

# TODO fix substitutes_empty_values_by_ones_behind() cause it is completely wrong