import pandas as pd
import numpy as np
from tqdm import tqdm
from Code.execution_activity import Execution_Activity
from Code.ngram_generator import NGram_Generator
from Code.sync_lib import generate_eigenvalue_named_columns

class Sequence_Activity_Execution:

    def __init__(self,name: str, numerical_type: type, number_dimensions: int) -> None:
        self.assert_argument_types(name, numerical_type)

        self.number_dimensions = number_dimensions
        self.numerical_type = numerical_type
        self.data = []
        self.name = name
        self.ngram_generator = NGram_Generator(numerical_type)

    def assert_argument_types(self, name: str, numerical_type: np.floating) -> None:
        assert isinstance(name, str), "name must be a string"
        assert isinstance(numerical_type, type), "numerical_type must be a type"

    def append_execution_sequence(self, execution: Execution_Activity, ):
        assert isinstance(execution, Execution_Activity)
        self.data.append(execution)

    def create_concataneted_ngram_df_from_executions(self,list_of_executions: list, ngram_size:int = 30) -> pd.DataFrame:
        concated_samples_df = pd.DataFrame()
        for execution in list_of_executions:
            execution_df = execution.get_data()
            execution_df = self.ngram_generator.create_3D_ngrams_df_from_2D_df(execution_df,ngram_size)
            #I'm not sure if this is the best way to do it, but I'm trying to avoid the case where the execution_df is empty
            if execution_df.shape[0] == 0:
                continue
            concated_samples_df = pd.concat([concated_samples_df ,execution_df],axis=0)
        return concated_samples_df

    def prepare_activity_for_input_classification_by_percentages(self, percentage_for_train: int, percentage_for_test: int, ngram_size:int = 30) -> tuple :
        
        train_data_from_activity, test_data_from_activity = self.generate_train_test_percentages(percentage_for_train, percentage_for_test)
        return self.creates_input_for_ml_model(train_data_from_activity, test_data_from_activity, ngram_size)

    def generate_train_test_percentages(self, percentage_for_train: int, percentage_for_test: int) -> tuple:
        assert percentage_for_train + percentage_for_test <= 100, "percentage_for_train + percentage_for_test must be less than 100"

        n_activitys = len(self.data)
        n_activitys_train = int((n_activitys* percentage_for_train) / 100)

        if percentage_for_test > 0:
            n_activitys_test = int((n_activitys* percentage_for_test) / 100)
            train_data_from_activity = self.data[:n_activitys_train]
            test_data_from_activity = self.data[n_activitys_train + n_activitys_test:]
        else:
            train_data_from_activity = self.data[:n_activitys_train]
            test_data_from_activity = self.data[n_activitys_train:]

        return train_data_from_activity, test_data_from_activity

    def prepare_activity_for_input_classification_by_days(self, days_for_train:list = [], days_for_test:list = [], ngram_size:int = 30) -> tuple :

        train_data_from_activity, test_data_from_activity = self.generate_train_test_days_lists(days_for_train, days_for_test)
        return self.creates_input_for_ml_model(train_data_from_activity, test_data_from_activity, ngram_size)

    def generate_train_test_days_lists(self, days_for_train:list = [], days_for_test:list = []) -> tuple:
        train_data_from_activity = []
        test_data_from_activity = []
        for execution in self.data:
            if str(execution.begining_time).split(" ")[0] in days_for_train:
                train_data_from_activity.append(execution)
            elif str(execution.begining_time).split(" ")[0] in days_for_test:
                test_data_from_activity.append(execution)
            else:
                continue
        
        return train_data_from_activity, test_data_from_activity

    def creates_input_for_ml_model(self, train_data_from_activity: list, test_data_from_activity: list, ngram_size:int = 30) -> tuple:

        ml_model_df_columns = generate_eigenvalue_named_columns(self.number_dimensions)
        ml_model_df_columns.append("Time") 

        train_input_ml_model_df = pd.DataFrame(columns=ml_model_df_columns)
        test_input_ml_model_df = pd.DataFrame(columns=ml_model_df_columns)
        train_input_series = pd.Series(dtype=self.numerical_type)
        test_input_series = pd.Series(dtype=self.numerical_type)

        if len(train_data_from_activity) != 0:
            train_input_ml_model_df = self.create_concataneted_ngram_df_from_executions(train_data_from_activity,ngram_size)
            #train_input_ml_model_df = self.ngram_generator.(train_input_ml_model_df, ngram_size)        
            train_input_series = pd.Series([self.name] * train_input_ml_model_df.shape[0] )

        if len(test_data_from_activity) != 0:
            test_input_ml_model_df = self.create_concataneted_ngram_df_from_executions(test_data_from_activity,ngram_size)
            #test_input_ml_model_df = self.ngram_generator.(test_input_ml_model_df, ngram_size)
            test_input_series = pd.Series([self.name] * test_input_ml_model_df.shape[0] )

        return train_input_ml_model_df, train_input_series, test_input_ml_model_df, test_input_series
    
    def join_activity_executions(self) -> pd.DataFrame:
        concated_samples_df = pd.DataFrame()
        for execution in self.data:
            concated_samples_df = pd.concat([concated_samples_df ,execution.get_data()],axis=0)
            
        return concated_samples_df
    
    def get_data(self) -> list:
        return self.data