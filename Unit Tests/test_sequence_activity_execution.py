import unittest
import pandas as pd
import numpy as np
import random

from context import Sequence_Activity_Execution
from context import Execution_Activity
from class_instance_mocks import generate_execution_mock

class Test_Sequence_Activity_Execution(unittest.TestCase):

    def setUp(self):
        # TODO make a mock of the execution activity in  the file class_instance_mocks.py
        self.name = "test"
        self.data_type = np.float64
        self.number_dimensions = 3
        self.sequence_activity_execution = Sequence_Activity_Execution(self.name, self.data_type, self.number_dimensions)

        execution_activity_1, data, begining_time, end_time = generate_execution_mock()
        execution_activity_2, data, begining_time, end_time = generate_execution_mock()
        execution_activity_3, data, begining_time, end_time = generate_execution_mock()
        execution_activity_4, data, begining_time, end_time = generate_execution_mock()
        execution_activity_5, data, begining_time, end_time = generate_execution_mock()
        execution_activity_6, data, begining_time, end_time = generate_execution_mock()
        execution_activity_7, data, begining_time, end_time = generate_execution_mock()
        execution_activity_8, data, begining_time, end_time = generate_execution_mock()
        execution_activity_9, data, begining_time, end_time = generate_execution_mock()
        execution_activity_10, data, begining_time, end_time = generate_execution_mock()    
    
        self.sequence_activity_execution.append_execution_sequence(execution_activity_1)
        self.sequence_activity_execution.append_execution_sequence(execution_activity_2)
        self.sequence_activity_execution.append_execution_sequence(execution_activity_3)
        self.sequence_activity_execution.append_execution_sequence(execution_activity_4)
        self.sequence_activity_execution.append_execution_sequence(execution_activity_5)
        self.sequence_activity_execution.append_execution_sequence(execution_activity_6)
        self.sequence_activity_execution.append_execution_sequence(execution_activity_7)
        self.sequence_activity_execution.append_execution_sequence(execution_activity_8)
        self.sequence_activity_execution.append_execution_sequence(execution_activity_9)
        self.sequence_activity_execution.append_execution_sequence(execution_activity_10)

    def test_assert_argument_types(self):
        self.assertRaises(AssertionError, self.sequence_activity_execution.assert_argument_types, 1, self.data_type)
        self.assertRaises(AssertionError, self.sequence_activity_execution.assert_argument_types, self.name, 1)

    def test_append_execution_sequence(self):
        mock_data_size_before = len(self.sequence_activity_execution.get_data())
        exection_to_append, data, begining_time, end_time = generate_execution_mock()
        self.sequence_activity_execution.append_execution_sequence(exection_to_append)
        mock_data_size_after = len(self.sequence_activity_execution.get_data())
        self.assertEqual(mock_data_size_before + 1, mock_data_size_after)
        last_execution = self.sequence_activity_execution.get_data()[-1]
        self.assertEqual(last_execution, exection_to_append)

    def test_create_concataneted_ngram_df_from_executions(self):
        ngram_size = 30
        concated_samples_df = self.sequence_activity_execution.create_concataneted_ngram_df_from_executions(self.sequence_activity_execution.get_data(), ngram_size)
        self.assertIsInstance(concated_samples_df, pd.DataFrame)
        self.assertEqual(concated_samples_df.iloc[0][0].shape[0], ngram_size)
        number_of_values_checked = 10
        for i in range(10):
            x = np.random.randint(0, concated_samples_df.shape[0])
            y = np.random.randint(0, concated_samples_df.shape[1])
            z = np.random.randint(0, ngram_size)
            self.assertEqual(concated_samples_df.iloc[x][y].dtype, np.float64)

    def test_prepare_activity_for_input_classification_by_percentages(self):
        pass
        # ...

    def test_generate_train_test_percentages(self):
        self.assertRaises(AssertionError, self.sequence_activity_execution.generate_train_test_percentages, 80,40)
        train_data_from_activity_with_test_percentage, test_data_from_activity_with_test_percentage = self.sequence_activity_execution.generate_train_test_percentages(60, 10)
        random_train_execution = train_data_from_activity_with_test_percentage[np.random.randint(0, len(train_data_from_activity_with_test_percentage))]
        random_test_execution = test_data_from_activity_with_test_percentage[np.random.randint(0, len(test_data_from_activity_with_test_percentage))]
        self.assertIsInstance(random_train_execution, Execution_Activity)
        self.assertIsInstance(random_test_execution, Execution_Activity)

        train_data_from_activity_without_test_percentage, test_data_from_activity_without_test_percentage = self.sequence_activity_execution.generate_train_test_percentages(60, 0)
        random_train_execution = train_data_from_activity_without_test_percentage[np.random.randint(0, len(train_data_from_activity_without_test_percentage))]
        random_test_execution = test_data_from_activity_without_test_percentage[np.random.randint(0, len(test_data_from_activity_without_test_percentage))]
        self.assertIsInstance(random_train_execution, Execution_Activity)
        self.assertIsInstance(random_test_execution, Execution_Activity)

        return random_train_execution, random_test_execution
        

    def test_prepare_activity_for_input_classification_by_days(self):    
        pass

    def pick_random_days_from_sequence_activity_execution(self, number_of_days: int):
        days = []
        for execution in self.sequence_activity_execution.get_data():
            df_execution = execution.get_data()
            number_of_samples = df_execution.shape[0]
            
            if(df_execution.shape[0]) == 0:
                continue               
            
            new_day = str(df_execution.iloc[0][0]).split(" ")[0]
            if new_day not in days:
                days.append(new_day)

        number_of_samples = len(days)
        random_ints = random.sample(range(0, number_of_samples), min(number_of_days, number_of_samples))
    
        map_random_days = map(lambda x: days[x], random_ints)
        random_days = list(map_random_days)

        return random_days

    def test_generate_train_test_days_lists(self):
        number_of_days = 10
        random_days = self.pick_random_days_from_sequence_activity_execution(number_of_days)
        
        
        train_data_from_activity_with_test_days, test_data_from_activity_with_test_days = self.sequence_activity_execution.generate_train_test_days_lists(random_days[0:5], random_days[5:10]) 
        #print(train_data_from_activity_with_test_days, test_data_from_activity_with_test_days)

        random_train_execution = train_data_from_activity_with_test_days[np.random.randint(0, len(train_data_from_activity_with_test_days))]
        random_test_execution = test_data_from_activity_with_test_days[np.random.randint(0, len(test_data_from_activity_with_test_days))]
        
        self.assertIsInstance(random_train_execution, Execution_Activity)
        self.assertIsInstance(random_test_execution, Execution_Activity)

    def test_creates_input_for_ml_model(self):
        random_train_execution, random_test_execution = self.test_generate_train_test_percentages()
        random_train_execution = [random_train_execution]
        random_test_execution = [random_test_execution]
        train_input_ml_model_df, train_input_series, test_input_ml_model_df, test_input_series = self.sequence_activity_execution.creates_input_for_ml_model(random_train_execution, random_test_execution, 30)    
        self.assertIsInstance(train_input_ml_model_df, pd.DataFrame)
        self.assertIsInstance(train_input_series, pd.Series)
        self.assertIsInstance(test_input_ml_model_df, pd.DataFrame)
        self.assertIsInstance(test_input_series, pd.Series)
        

    def test_join_activity_executions(self):
        concataneted_executions_df = self.sequence_activity_execution.join_activity_executions()
        self.assertIsInstance(concataneted_executions_df, pd.DataFrame)
        last_execution_data = self.sequence_activity_execution.get_data()[-1].get_data()
        self.assertEqual(concataneted_executions_df.shape[1], last_execution_data.shape[1])

    def test_get_data(self):
        self.assertIsInstance(self.sequence_activity_execution.get_data(), list)


if __name__ == '__main__':
    unittest.main()