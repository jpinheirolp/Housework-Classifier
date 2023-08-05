import unittest
from context import Execution_Activity
from class_instance_mocks import generate_execution_mock
import pandas as pd
import numpy as np

class Test_Execution_Activity(unittest.TestCase):

    def setUp(self):
        self.comment = 'This is a comment'
        self.execution_activity, self.data, self.begining_time, self.end_time = generate_execution_mock()

    def test_assert_argument_types(self):
        self.assertRaises(AssertionError, self.execution_activity.assert_argument_types, 1, self.begining_time, self.end_time)
        self.assertRaises(AssertionError, self.execution_activity.assert_argument_types, self.data, 1, self.end_time)
        self.assertRaises(AssertionError, self.execution_activity.assert_argument_types, self.data, self.begining_time, 1)

    def test_data(self):
        self.assertIsInstance(self.execution_activity.get_data(), pd.DataFrame)

    def test_begining_time(self):
        self.assertIsInstance(self.execution_activity.get_begining_time(), pd.Timestamp)

    def test_end_time(self):
        self.assertIsInstance(self.execution_activity.get_end_time(), pd.Timestamp)

    def test_comment(self):
        self.assertIsInstance(self.execution_activity.get_comment(), str)
    
if __name__ == '__main__':
    unittest.main()
    