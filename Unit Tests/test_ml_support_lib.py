import unittest
from context import *
import pandas as pd
import numpy as np
import math

class Test_ML_Support_Lib(unittest.TestCase):

    def setUp(self):
        self.ngram_generator = NGram_Generator(numerical_type=np.single)
        self.desired_freq = 7
        self.mock_df = pd.DataFrame(data = np.random.rand(10,3), columns = ["A","B","C"])
        self.mock_df_without_time = self.mock_df.copy()
        self.mock_df["Time"] = pd.date_range(start="2020-01-01 00:00:00", periods= 10 , freq= f"{self.desired_freq}S")
        self.mock_df.at[7,"A"] = np.nan

        # chat gpt fez isso talvez fassa sentido
        self.mock_df_wrong_freq = pd.DataFrame(data = np.random.rand(10,3), columns = ["A","B","C"])
        self.mock_df_wrong_freq["Time"] = pd.date_range(start="2020-01-01 00:00:00", periods= 10 , freq= "4S")

    def test_separates_non_numerical_columns(self):
        df_numerical, dict_non_numerical = separates_non_numerical_columns(self.mock_df.copy())
        self.assertEqual(df_numerical.shape[1], self.mock_df.shape[1] - 1)
        self.assertEqual(len(dict_non_numerical), 1)
        self.assertIsInstance(df_numerical, pd.DataFrame)
        self.assertIsInstance(dict_non_numerical, dict)

    def test_normalyzes_numerical_values_df(self): # this function normalizes by the formula z = (x - u) / s
        df_normalized = normalyzes_numerical_values_df(self.mock_df.copy())
        for column in df_normalized.columns:
            if column == "Time":
                continue
            self.assertTrue(math.isclose(df_normalized[column].mean(), 0, rel_tol=0.1, abs_tol=0.1))
            self.assertTrue(math.isclose(df_normalized[column].std(), 1, rel_tol=0.1, abs_tol=0.1))

        self.assertIsInstance(df_normalized, pd.DataFrame)
        self.assertEqual(df_normalized.shape, self.mock_df.shape)

    def test_reduces_dimensionality_by_PCA(self):
        df_reduced = reduces_dimensionality_by_PCA(self.mock_df_wrong_freq.copy(), 2)
        self.assertEqual(df_reduced.shape[1], 3)
        self.assertIsInstance(df_reduced, pd.DataFrame)

    def test_compensates_drift_with_PCA(self):
        df_compensated = compensates_drift_with_PCA(self.mock_df_wrong_freq.copy())
        self.assertIsInstance(df_compensated, pd.DataFrame)
        self.assertEqual(df_compensated.shape, self.mock_df_wrong_freq.shape)

    def test_convert_df_to_3d_np_array(self):
        three_d_ngrams_df = self.ngram_generator.create_3D_ngrams_df_from_2D_df(self.mock_df_without_time, 10)
        converted_df = convert_df_to_3d_np_array(three_d_ngrams_df, np.single)
        self.assertIsInstance(converted_df, np.ndarray)
        self.assertEqual(converted_df.shape, (three_d_ngrams_df.shape[0], three_d_ngrams_df.shape[1], three_d_ngrams_df.iloc[0][0].shape[0]))

    #this functions were not used in the final version of the code
    def test_shuffle_data_classification(self):
        pass
    
    def test_calculate_dtw_centroid(self):
        pass

    def test_dtw_distance_to_centroid(self):
        pass

    def test_detect_and_remove_outliers_positive_centroid(self):
        pass
 
    def test_detect_and_remove_outliers_negative_centroid(self):
        pass
