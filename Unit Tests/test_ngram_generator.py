import unittest
from context import NGram_Generator
import pandas as pd
import numpy as np

class Test_Ngram_Generator(unittest.TestCase):

    def setUp(self):
        self.ngram_generator = NGram_Generator(numerical_type=np.single)
        mock_list = [i for i in range(1,100)]
        self.input_series = pd.Series(mock_list[:])
        self.input_df = pd.DataFrame({'col1': mock_list[:], 'col2': mock_list[:]})

    def test_create_2D_ngrams_series_from_1D_series(self):
        two_d_ngrams_series = self.ngram_generator.create_2D_ngrams_series_from_1D_series(self.input_series)
        self.assertIsInstance(two_d_ngrams_series, pd.Series)
        self.assertEqual(two_d_ngrams_series.shape[0], 70)
        self.assertEqual(two_d_ngrams_series.iloc[0].shape[0], 30)

        self.assertIsInstance(two_d_ngrams_series.iloc[0], np.ndarray)
        self.assertEqual(two_d_ngrams_series.iloc[0][0], 1)
        self.assertEqual(two_d_ngrams_series.iloc[0][1], 2)

    def test_create_3D_ngrams_df_from_2D_df(self):
        three_d_ngrams_df = self.ngram_generator.create_3D_ngrams_df_from_2D_df(self.input_df, 30)
        self.assertIsInstance(three_d_ngrams_df, pd.DataFrame)

        self.assertEqual(three_d_ngrams_df.shape[0], 70)
        self.assertEqual(three_d_ngrams_df.shape[1], 2)
        
        self.assertEqual(three_d_ngrams_df.iloc[0][0].shape[0], 30)

    
if __name__ == '__main__':
    unittest.main()
    