import unittest
import pandas as pd
import numpy as np
import random

from context import House_Activitys_Learning_Base
from context import Sequence_Activity_Execution
from context import Execution_Activity
from class_instance_mocks import generate_execution_mock

class Test_House_Activitys_Learning_Base(unittest.TestCase):

    def setUp(self):
        self.activitys_df = pd.DataFrame(data = np.random.rand(100,3), columns = ["A","B","C"])
        self.activitys_df["Time"] = pd.date_range(start="2020-01-01 00:00:00", periods= 100 , freq= "4S")


        list_of_activitys = ["A","B","C","D"]
        list_of_comeents = ["comment1","comment2",""]
        self.mock_time_schedule_df = pd.DataFrame(columns = ["activity", "Started", "Ended", "Comments"] )
        self.mock_time_schedule_df["Started"] = pd.date_range(start="2020-01-01 00:00:00", periods= 10 , freq= "1Min")
        for i in self.mock_time_schedule_df.index:
            self.mock_time_schedule_df.at[i,"activity"] = random.choice(list_of_activitys)
            self.mock_time_schedule_df.at[i,"Comments"] = random.choice(list_of_comeents)
            random_time = random.randint(1, 30)
            self.mock_time_schedule_df.at[i,"Ended"] = self.mock_time_schedule_df.at[i,"Started"] + pd.Timedelta(minutes= random_time)


        self.house_activitys_learning_base = House_Activitys_Learning_Base(
            desired_freq = 4 ,
            synchronized_activitys_df = self.activitys_df, 
            time_schedule_df = self.mock_time_schedule_df,
            number_dimensions = 3)

    def test_prepare_data_input_classification(self):
        np_train_input_ml_model_df, np_train_input_series, np_test_input_ml_model_df, np_test_input_series = self.house_activitys_learning_base.prepare_data_input_classification(percentage_for_train = 80, percentage_for_test = 20)
        self.assertEqual(np_train_input_ml_model_df.shape[0], np_train_input_series.shape[0])
        self.assertEqual(np_test_input_ml_model_df.shape[0], np_test_input_series.shape[0])
        self.assertEqual(np_train_input_ml_model_df.shape[1], np_test_input_ml_model_df.shape[1])
        self.assertIsInstance(np_train_input_ml_model_df, np.ndarray)
        self.assertIsInstance(np_train_input_series, np.ndarray)
        self.assertIsInstance(np_test_input_ml_model_df, np.ndarray)
        self.assertIsInstance(np_test_input_series, np.ndarray)
        
    def test_save_data_to_csv(self):
        pass