import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Code.sequence_activity_execution import Sequence_Activity_Execution
from Code.execution_activity import Execution_Activity
from Code.ml_support_lib import convert_df_to_3d_np_array
from tqdm import tqdm

class House_Activitys_Learning_Base: # take about when to convert time str to datetime; make sure that columns are named eigenvalue1, eigenvalue2 ... ; 
    def __init__(self, desired_freq: int, synchronized_activitys_df: pd.DataFrame ,time_schedule_df: pd.DataFrame, number_dimensions: int, numerical_type: np.object = np.single) -> None:
        self.data = {}
        self.numerical_type = numerical_type
        for activity in time_schedule_df["activity"].unique():
            self.data[activity] = Sequence_Activity_Execution(activity,numerical_type= self.numerical_type, number_dimensions=number_dimensions)

        self.data["no_activity"] = Sequence_Activity_Execution("no_activity",numerical_type= self.numerical_type, number_dimensions=number_dimensions)
        synchronized_activitys_df_time_stamps_set = set(synchronized_activitys_df["Time"][:])
        #check_number_of_activitys = 0        

        # 1st idea for creating "no activitys" class: append all sets of activitys and then remove the intersection of all activitys
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
            #check_number_of_activitys += execution_df.shape[0] 
            if activitys_timenstamp_set_intersect == set():
                #print("data is empty for activity", activity_name, time_begining, time_end)
                continue
            
            execution_activity_instance = Execution_Activity(execution_df,time_begining,time_end,activity_comment)
            self.data[activity_name].append_execution_sequence(execution_activity_instance)

        MINIMUM_EXECUTION_SIZE = 60

        all_no_activity_executions = synchronized_activitys_df[~synchronized_activitys_df["Time"].isin(all_activitys_timestamp_set)]
        randomly_selected_no_activity_executions = all_no_activity_executions[:][:number_no_activity_executions * MINIMUM_EXECUTION_SIZE]#.sample(number_no_activity_executions * MINIMUM_EXECUTION_SIZE) #here you puted the number of executions instead of the number of samples
        
         #MINIMUM_EXECUTION_SIZE is the number of executions in the shortest activity that lasts 10 minutes
        for i in range(number_no_activity_executions):
            no_activity_execution_df = synchronized_activitys_df.iloc[i*MINIMUM_EXECUTION_SIZE:(i+1)*MINIMUM_EXECUTION_SIZE,:]
            
        
            execution_activity_instance = Execution_Activity(no_activity_execution_df,time_begining,time_end,activity_comment)
            
            self.data["no_activity"].append_execution_sequence(execution_activity_instance)
         
    # fix so that the function works also with list of days
    # the way it is now, it only works with percentages
    def prepare_data_input_classification(self, ngram_size:int = 30, output_dtype="np_array", percentage_for_train: int = 0, percentage_for_test: int = 0, days_for_train:list = [], days_for_test:list = []) -> tuple :
        train_input_ml_model_df = pd.DataFrame()
        train_input_series = pd.Series(dtype= self.numerical_type)
        test_input_ml_model_df = pd.DataFrame()
        test_input_series = pd.Series(dtype= self.numerical_type)
        negative_centroid = np.zeros(ngram_size)

        list_test_activity_data = []

        # train_activity_df, train_activity_series, test_activity_df, test_activity_series = self.data["AS1"].prepare_activity_for_input_classification_by_percentages(percentage_for_train, percentage_for_test, ngram_size)
        # does_not_matter_1, does_not_matter_2, negative_centroid = detect_and_remove_outliers_positive_centroid(train_activity_df, train_activity_series, "train_AS1")
        total_number_of_samples_train = 0


        # for activity_name in self.data:    
        #     activity = self.data[activity_name]
        #     train_activity_df, train_activity_series, test_activity_df, test_activity_series = activity.prepare_activity_for_input_classification_by_percentages(percentage_for_train, percentage_for_test, ngram_size)
        #     total_number_of_samples_train += train_activity_df.shape[0]

        # average_number_of_samples_train = total_number_of_samples_train // len(self.data)

        
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

            
            # train_activity_df, train_activity_series, negative_centroid_part = detect_and_remove_outliers_positive_centroid(train_activity_df, train_activity_series, f"train_{activity_name}")
            # negative_centroid += negative_centroid_part
            # list_test_activity_data.append([test_activity_df, test_activity_series] )

            # train_activity_df, train_activity_series = detect_and_remove_outliers_negative_centroid(train_activity_df, train_activity_series, negative_centroid, "train")
            # test_activity_df, test_activity_series = detect_and_remove_outliers_negative_centroid(test_activity_df, test_activity_series, negative_centroid, "test")

            # if train_activity_df.shape[0] > average_number_of_samples_train:
            #     train_activity_df["classes"] = train_activity_series
            #     train_activity_df = train_activity_df.sample(n = average_number_of_samples_train)
            #     train_activity_series = train_activity_df["classes"]
            #     train_activity_df.drop("classes",axis=1,inplace=True)
            
            train_input_ml_model_df = pd.concat([train_input_ml_model_df ,train_activity_df],axis=0)
            train_input_series =  pd.concat([train_input_series, train_activity_series])
            test_input_ml_model_df = pd.concat([test_input_ml_model_df ,test_activity_df],axis=0)
            test_input_series =  pd.concat([test_input_series, test_activity_series])

        
        # for test_activity_data in list_test_activity_data:
        #     test_activity_df, test_activity_series = detect_and_remove_outliers_negative_centroid(test_activity_data[0], test_activity_data[1], negative_centroid, "test")
        #     test_input_ml_model_df = pd.concat([test_input_ml_model_df ,test_activity_df],axis=0)
        #     test_input_series =  pd.concat([test_input_series, test_activity_series])


        #test_input_ml_model_df, test_input_series = detect_and_remove_outliers_test(test_input_ml_model_df, test_input_series, negative_centroid, "test")
        
        #DEBUGGING
        print("test_activity_df",test_activity_df)

        print("train_input_ml_model_df",train_input_ml_model_df.shape)
        print("train_input_series",train_input_series.shape)
        print("test_input_ml_model_df",test_input_ml_model_df.shape)
        print("test_input_series",test_input_series.shape)



        #shuffled_train_input_ml_model_df, shuffled_train_input_series = shuffle_data_classification(train_input_ml_model_df, train_input_series)    
        #shuffled_test_input_ml_model_df, shuffled_test_input_series = shuffle_data_classification(test_input_ml_model_df, test_input_series)

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

#'''

