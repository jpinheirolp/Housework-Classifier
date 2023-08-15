import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Code.sequence_activity_execution import Sequence_Activity_Execution
from Code.execution_activity import Execution_Activity
from Code.ml_support_lib import convert_df_to_3d_np_array
from tqdm import tqdm

class House_Activitys_Learning_Base: 
    def __init__(self, desired_freq: int, synchronized_activitys_df: pd.DataFrame ,time_schedule_df: pd.DataFrame, number_dimensions: int, numerical_type: np.object = np.single) -> None:
        self.data = {}
        self.numerical_type = numerical_type
        for activity in time_schedule_df["activity"].unique():
            self.data[activity] = Sequence_Activity_Execution(activity,numerical_type= self.numerical_type, number_dimensions=number_dimensions)

        self.data["no_activity"] = Sequence_Activity_Execution("no_activity",numerical_type= self.numerical_type, number_dimensions=number_dimensions)
        synchronized_activitys_df_time_stamps_set = set(synchronized_activitys_df["Time"][:])
        
        all_activitys_timestamp_set = set()

        number_no_activity_executions = 0

        for index, row in time_schedule_df.iterrows():
            time_begining = row["Started"]
            time_begining = pd.to_datetime(time_begining, format='%d/%m/%Y %H:%M')
            time_end = row["Ended"]
            time_end = pd.to_datetime(time_end, format='%d/%m/%Y %H:%M')
            activity_name = row["activity"]
            if activity_name == "Aera":
                number_no_activity_executions += 1
            activity_comment = row["Comments"]
            activitys_timenstamp_sequence = pd.date_range(start= time_begining, end= time_end, freq= f"{desired_freq}S")
            activitys_timenstamp_set = set(activitys_timenstamp_sequence)
            all_activitys_timestamp_set = all_activitys_timestamp_set.union(activitys_timenstamp_set)
            
            activitys_timenstamp_set_intersect = activitys_timenstamp_set.intersection(synchronized_activitys_df_time_stamps_set)
            execution_df = synchronized_activitys_df[synchronized_activitys_df["Time"].isin(activitys_timenstamp_set_intersect)]
            
            if activitys_timenstamp_set_intersect == set():
                continue
            
            execution_activity_instance = Execution_Activity(execution_df,time_begining,time_end,activity_comment)
            self.data[activity_name].append_execution_sequence(execution_activity_instance)

        MINIMUM_EXECUTION_SIZE = 60

        all_no_activity_executions = synchronized_activitys_df[~synchronized_activitys_df["Time"].isin(all_activitys_timestamp_set)]
        randomly_selected_no_activity_executions = all_no_activity_executions[:][:number_no_activity_executions * MINIMUM_EXECUTION_SIZE]#.sample(number_no_activity_executions * MINIMUM_EXECUTION_SIZE) #here you puted the number of executions instead of the number of samples
        
        for i in range(number_no_activity_executions):
            no_activity_execution_df = synchronized_activitys_df.iloc[i*MINIMUM_EXECUTION_SIZE:(i+1)*MINIMUM_EXECUTION_SIZE,:]
            
        
            execution_activity_instance = Execution_Activity(no_activity_execution_df,time_begining,time_end,activity_comment)
            
            self.data["no_activity"].append_execution_sequence(execution_activity_instance)
         
    def prepare_data_input_classification(self, ngram_size:int = 30, output_dtype="np_array", percentage_for_train: int = 0, percentage_for_test: int = 0, days_for_train:list = [], days_for_test:list = []) -> tuple :
        train_input_ml_model_df = pd.DataFrame()
        train_input_series = pd.Series(dtype= self.numerical_type)
        test_input_ml_model_df = pd.DataFrame()
        test_input_series = pd.Series(dtype= self.numerical_type)

        
        for activity_name in self.data:
            
            activity = self.data[activity_name]
            train_activity_df, train_activity_series, test_activity_df, test_activity_series = activity.prepare_activity_for_input_classification_by_percentages(percentage_for_train, percentage_for_test, ngram_size)

            if (train_activity_df.shape[0] == 0
                or test_activity_df.shape[0] == 0
                or train_activity_series.shape == 0
                or test_activity_series.shape == 0):
                
                continue
            
            train_activity_df.drop("Time",axis=1,inplace=True)
            test_activity_df.drop("Time",axis=1,inplace=True)

            
            train_input_ml_model_df = pd.concat([train_input_ml_model_df ,train_activity_df],axis=0)
            train_input_series =  pd.concat([train_input_series, train_activity_series])
            test_input_ml_model_df = pd.concat([test_input_ml_model_df ,test_activity_df],axis=0)
            test_input_series =  pd.concat([test_input_series, test_activity_series])


        if output_dtype == "np_array":
            
            np_train_input_ml_model_df = convert_df_to_3d_np_array(train_input_ml_model_df, self.numerical_type)
            np_train_input_series = np.array(train_input_series)
            np_test_input_ml_model_df = convert_df_to_3d_np_array(test_input_ml_model_df, self.numerical_type)
            np_test_input_series = np.array(test_input_series)
        
       
        return np_train_input_ml_model_df, np_train_input_series, np_test_input_ml_model_df, np_test_input_series
    
    def save_data_to_csv(self, path: str):
        for activity_name in self.data:
            activity = self.data[activity_name]
            csv_data = activity.join_activity_executions()
            csv_data.to_csv(f"{path}/{activity_name}_samples.csv",index=False)

