import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from nltk.util import ngrams
import seaborn as sns
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.distances import dtw_distance
import matplotlib.pyplot as plt


def substitutes_empty_values_by_ones_behind(df):
    for col in df.columns:
        for row in df.index[1:]: #what if the first row is empty? maybe we should remove it
            if pd.isna(df[col][row]):
                value_to_update = df.at[row - 1,col]
                df.at[row,col] = value_to_update
    return df

#changes the frequency of a dataframe by turning every row that is in the desired freq interval to one single row corresponding to the moving averadge
def ajust_df_frequency_by_mv_avg(df_wrong_freq, desired_freq):
    
    initial_time_value = pd.to_datetime(df_wrong_freq["Time"].iloc[0])
    initial_time_value = initial_time_value.replace(second = (initial_time_value.second // desired_freq) * desired_freq)
    
    time_interval_col = pd.date_range(start=initial_time_value, periods= df_wrong_freq.shape[0] , freq= f"{desired_freq}S")
    
    list_columns = list(df_wrong_freq.columns)
    list_columns.remove("Time")

    values_mv_avg = np.zeros((df_wrong_freq.shape[0],df_wrong_freq.shape[1] - 1))
    df_mv_avg = pd.DataFrame(data= values_mv_avg,columns=list_columns)
    df_mv_avg["Time_Interval"] = time_interval_col
    columns_without_time = list_columns
    iterator_mv_avg = 0
    num_cols_interval = 0
    timestamp_mv_avg = initial_time_value

    #fix date format, pandas is changing the order of month and day for some reason
    num_col_output = 0
    for row in tqdm(df_wrong_freq.index):
        #print(df_wrong_freq.at[row,"Time"] , timestamp_mv_avg + pd.Timedelta(seconds=desired_freq))
        if df_wrong_freq.at[row,"Time"] > timestamp_mv_avg + pd.Timedelta(seconds=desired_freq): #df_mv_avg.at[iterator_mv_avg + 1,"Time_Interval"]
            for col in columns_without_time:
                df_mv_avg.at[iterator_mv_avg, col] /= num_cols_interval  
            df_mv_avg.at[iterator_mv_avg, "Time_Interval"] = timestamp_mv_avg
            
            num_cols_interval = 0
            iterator_mv_avg += 1
            measure_time = df_wrong_freq.at[row,"Time"]
            timestamp_mv_avg = measure_time.replace(second = (measure_time.second // desired_freq) * desired_freq)
            num_col_output += 1
            

        num_cols_interval += 1 

        for col in columns_without_time:
            df_mv_avg.at[iterator_mv_avg, col] += df_wrong_freq.at[row,col]
        

    for col in columns_without_time:
        df_mv_avg.at[iterator_mv_avg, col] /= num_cols_interval  
    df_mv_avg.at[iterator_mv_avg, "Time_Interval"] = timestamp_mv_avg
    df_mv_avg = df_mv_avg.iloc[:iterator_mv_avg+1,:]

    

    return df_mv_avg
    #'''
        
def add_prefixes_in_columns_names(df,df_prefix,col_in_common = "Time_Interval"):
    columns_df_renamed = []
    list_columns_df = list(df.columns)
    list_columns_df.remove(col_in_common)
    for col in list_columns_df:
        columns_df_renamed.append(df_prefix + col)
    columns_df_renamed.append(col_in_common)  
    df.columns = columns_df_renamed
    return df
    
#Renames the two input dfs according to the above function and merges by concatenating all rows in wich the value of the column col_in_common belongs to both dfs 
def merge_ajusting_col_names(df1,df2,df1_prefix="",df2_prefix="",col_in_common = "Time_Interval"):
    df1_renamed = add_prefixes_in_columns_names(df1,df1_prefix,col_in_common)
    df2_renamed = add_prefixes_in_columns_names(df2,df2_prefix,col_in_common)
    return df1_renamed.merge(df2_renamed,how="inner",on=col_in_common)


#'''  returns two sets of rows df1[intersection_column] - df2[intersection_column] and df1[intersection_column] - df2[intersection_column], this are set operations

def select_intersecting_rows(df1, df2, intersection_column): # returns empty df, find why
    intersection_set = set(df1[intersection_column]).intersection(set(df2[intersection_column]))
    df1_intersect = df1.loc[(df1[intersection_column].isin(intersection_set))]
    df2_intersect = df2.loc[(df2[intersection_column].isin(intersection_set))]
    return df1_intersect, df2_intersect

#removes nan values or puts the previous column value according to the rule you sent me on whtasapp 
def remove_nan_dfs(df1_intersect,df2_intersect,window_size= 5,max_phase_dif = 8,time_col = "Time"):
    # Loop through the DataFrame with a sliding window
    indexes_to_drop_df1 = []
    indexes_to_drop_df2 = []
    i1 = 0
    i2 = 0
    while (i1 < ( len(df1_intersect) - window_size + 1)) and i2 < (( len(df2_intersect) - window_size + 1)):
        df1_time = df1_intersect[time_col][i1]
        df2_time = df2_intersect[time_col][i2] 

        window_1 = df1_intersect[i1:i1 + window_size]
        window_2 = df2_intersect[i2:i2 + window_size]

        indexes_to_drop_df1 = []
        indexes_to_drop_df2 = []

        if np.abs((df1_time - df2_time).total_seconds()) > max_phase_dif:
            print(df2_intersect.loc[i2] ,df2_intersect.loc[i2].isnull().any())

            if df1_time > df2_time:
                if df2_intersect.loc[i2].isnull().any():
                    indexes_to_drop_df2.append(i2)
                i2 += 1
            else:
                if df1_intersect.loc[i1].isnull().any():
                    indexes_to_drop_df1.append(i1)
                i1 += 1
            continue
        window_1 = df1_intersect[i1:i1 + window_size]
        window_2 = df2_intersect[i2:i2 + window_size]
        # Check if all rows in the window have at least one NaN value in both dfs
        if window_1.isnull().any(axis=1).all() and window_2.isnull().any(axis=1).all():
            print("dropping")
            indexes_to_drop_df1 += list(window_1.index)
            indexes_to_drop_df2 += list(window_2.index)
            
        i1 += 1
        i2 += 1
    
    print(indexes_to_drop_df1, indexes_to_drop_df2)
    df1_intersect.drop(axis=0, index= indexes_to_drop_df1 ,inplace = True)
    df2_intersect.drop(axis=0, index= indexes_to_drop_df2 ,inplace = True)
        
    df1_without_nan = df1_intersect.reset_index(drop=True)
    df2_without_nan = df2_intersect.reset_index(drop=True)
    return df1_without_nan,df2_without_nan

# test if values make sense after normalyzing
def separates_non_numerical_columns(df: pd.DataFrame):
    non_numerical_columns = dict()

    for col_name, col_data in df.items():
        try: 
            df[col_name] = df[col_name].astype(float)
        except:
            non_numerical_columns[col_name] = col_data
            df.drop([col_name], axis=1, inplace=True)
    return df, non_numerical_columns

def normalyzing_df(df: pd.DataFrame):
    numerical_df, non_numerical_columns = separates_non_numerical_columns(df)
    values_df = numerical_df.values
    columns_df = numerical_df.columns

    normalized_values = StandardScaler().fit_transform(values_df)
    normalized_df = pd.DataFrame(data = normalized_values, columns=columns_df)
    
    for key, value in non_numerical_columns.items():
        normalized_df[key] = value
    return normalized_df
#test it also

def pca_dimension_reduction(normalized_df: pd.DataFrame, num_dim_reductiopn: int):
    normalized_num_df, non_numerical_columns = separates_non_numerical_columns(normalized_df)
    values_df = normalized_num_df.values
    pca_instance = PCA(n_components=num_dim_reductiopn)
    dimension_reducted_values = pca_instance.fit_transform(values_df)
    columns_name_after_reduction = [f"eigenvalue_{i+1}"  for i in range(num_dim_reductiopn)]
    dimension_reducted_df = pd.DataFrame(data = dimension_reducted_values, columns=columns_name_after_reduction)
    print('Explained variation per principal component: {}'.format(pca_instance.explained_variance_ratio_))


    for key, value in non_numerical_columns.items():
        dimension_reducted_df[key] = value
    
    return dimension_reducted_df

def drift_compensation_PCA(normalized_df: pd.DataFrame):
    print(normalized_df.columns)
    time_column = normalized_df['Time']
    dimension_reducted_df = pca_dimension_reduction(normalized_df, 2)
    most_significant_eigenvector = dimension_reducted_df[dimension_reducted_df.columns[0]]
    
    # check if this transpose is correct, maybe the shapes are should be different
    tst = np.matmul(normalized_df.T, most_significant_eigenvector)
    print("tst.shape, most_significant_eigenvector.shape, normalized_df.shape", tst.shape, most_significant_eigenvector.shape, normalized_df.shape)
    # tst_matrix = np.reshape(tst, (-1, 1))
    # most_significant_eigenvector_matrix = np.reshape(most_significant_eigenvector, (1, -1))
    
   
    corrected = normalized_df - np.matmul(most_significant_eigenvector[:, np.newaxis], tst.T[np.newaxis,:])
    drift_conpensated_df = np.abs(corrected)
    drift_conpensated_df["Time"] = time_column

    return drift_conpensated_df

class Execution_Activity():
    def __init__(self, execution_df: pd.DataFrame, begining_time: pd.DatetimeTZDtype, end_time: pd.DatetimeTZDtype, comment: str):
        self.data = execution_df
        self.begining_time = begining_time
        self.end_time = end_time
        self.comment = comment

    def get_data(self):
        return self.data
        

class Sequence_Activity_Execution:
    def __init__(self,name: str, numerical_type: np.object) -> None:
        self.data = []
        self.name = name
        self.numerical_type = numerical_type

    def append_execution_sequence(self, execution: Execution_Activity, ):
        self.data.append(execution)
    
    def create_ngrams_df(self, original_df ,ngram_size):
       
        # define a function to generate n-grams for each column

        def to_ngrams(col:pd.Series, ngram_size:int = 30) -> pd.Series:
            result = []
            #print("col",col.size)
            for i in range(col.size - ngram_size + 1):#(col.size // ngram_size ):#
                ngram = pd.Series(col[i:i+ngram_size],dtype= self.numerical_type)#(col[i*ngram_size:(i+1)*ngram_size],dtype= self.numerical_type)#
                ngram_np_array = np.array(ngram,dtype= self.numerical_type)
                result.append(ngram_np_array)

            result = pd.Series(result)
            return result
        
        ngram_df = original_df.apply(to_ngrams, axis=0, args=(ngram_size,))
        percentil_95 = int((ngram_df.shape[0] * 95) / 100)
        #DEBUGGING
        if ngram_df.shape[0] == 0:
            print(original_df.shape, ngram_df.shape, ngram_size)
            print("ngram_df.iloc[0]")
        
        # print(" underestanding ngram  " ,ngram_df.shape, type(ngram_df.iloc[0,:]), ngram_df.iloc[0,:].shape, ngram_df.iloc[0,:])
        # this is a hack to return a dataframe with only one row and should be changed by being ajusted by a parameter
        return ngram_df#.iloc[percentil_95:percentil_95+1][:]#.iloc[percentil_95:percentil_95+1][:]

    def extract_samples_from_execution(self,list_of_executions: list, ngram_size:int = 30) -> pd.DataFrame:
        concated_samples_df = pd.DataFrame()
        for execution in list_of_executions:
            execution_df = execution.get_data()
            execution_df = self.create_ngrams_df(execution_df,ngram_size)
            #I'm not sure if this is the best way to do it, but I'm trying to avoid the case where the execution_df is empty
            if execution_df.shape[0] == 0:
                continue
            concated_samples_df = pd.concat([concated_samples_df ,execution_df],axis=0)
        return concated_samples_df

    def prepare_activity_input_classification(self, percentage_for_train: int, percentage_for_test: int , days_for_train:list = [], days_for_test:list = [], ngram_size:int = 30) -> tuple :


        '''
         Add argument percentage_for_test
         Add argument number_days_to_test and number_days_to_train
         so that they are the a string with the desired day, and the its necessary to iterate over the list of executions and find activitys that have the desired day
         the respective percentage argument will be ignored unless the number_days_to_test and number_days_to_train are 0

        '''


        if percentage_for_train > 0:
            n_activitys = len(self.data)
            n_activitys_train = int((n_activitys* percentage_for_train) / 100)

            if percentage_for_test > 0:
                n_activitys_test = int((n_activitys* percentage_for_test) / 100)
                train_data_from_activity = self.data[:n_activitys_train]
                test_data_from_activity = self.data[n_activitys_train + n_activitys_test:]

            else:
                
                train_data_from_activity = self.data[:n_activitys_train]
                test_data_from_activity = self.data[n_activitys_train:]


            print(self.name)
            print("train_data_from_activity",len(train_data_from_activity))
            print("test_data_from_activity",len(test_data_from_activity))
            print("\n")

        elif len(days_for_train) > 0 and len(days_for_test) > 0:
            train_data_from_activity = []
            test_data_from_activity = []
            for execution in self.data:
                if str(execution.begining_time).split(" ")[0] in days_for_train:
                    train_data_from_activity.append(execution)
                elif str(execution.begining_time).split(" ")[0] in days_for_test:
                    test_data_from_activity.append(execution)
                else:
                    pass
        else:
            raise Exception("Invalid arguments: percentage_for_train or days_for_train and days_for_test must be provided")

        train_input_ml_model_df = pd.DataFrame(columns=['eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3', 'eigenvalue_4', 'Time'])
        test_input_ml_model_df = pd.DataFrame(columns=['eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3', 'eigenvalue_4', 'Time'])
        train_input_series = pd.Series(dtype="float32")
        test_input_series = pd.Series(dtype="float32")

        if len(train_data_from_activity) != 0:
            train_input_ml_model_df = self.extract_samples_from_execution(train_data_from_activity,ngram_size)
            #train_input_ml_model_df = self.create_ngrams_df(train_input_ml_model_df, ngram_size)        
            train_input_series = pd.Series([self.name] * train_input_ml_model_df.shape[0] )

        if len(test_data_from_activity) != 0:
            test_input_ml_model_df = self.extract_samples_from_execution(test_data_from_activity,ngram_size)
            #test_input_ml_model_df = self.create_ngrams_df(test_input_ml_model_df, ngram_size)
            test_input_series = pd.Series([self.name] * test_input_ml_model_df.shape[0] )

        
        return train_input_ml_model_df, train_input_series, test_input_ml_model_df, test_input_series
    
    def join_activity_executions(self) -> pd.DataFrame:
        concated_samples_df = pd.DataFrame()
        for execution in self.data:
            concated_samples_df = pd.concat([concated_samples_df ,execution.get_data()],axis=0)
            
        return concated_samples_df

class House_Activitys_Learning_Base: # take about when to convert time str to datetime
    def __init__(self, desired_freq: int, synchronized_activitys_df: pd.DataFrame ,time_schedule_df: pd.DataFrame, numerical_type: np.object = np.single) -> None:
        self.data = {}
        self.numerical_type = numerical_type
        for activity in time_schedule_df["activity"].unique():
            self.data[activity] = Sequence_Activity_Execution(activity,numerical_type= self.numerical_type)

        
        self.data["no_activity"] = Sequence_Activity_Execution("no_activity",numerical_type= self.numerical_type)
        synchronized_activitys_df_time_stamps_set = set(synchronized_activitys_df["Time"][:])
        print("synchronized_activitys_df_time_stamps_set",len(synchronized_activitys_df_time_stamps_set))
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
        # DEBUGGING 
        print("all_no_activity_executions",all_no_activity_executions.shape)
        print("randomly_selected_no_activity_executions", randomly_selected_no_activity_executions.shape)
         #MINIMUM_EXECUTION_SIZE is the number of executions in the shortest activity that lasts 10 minutes
        for i in range(number_no_activity_executions):
            no_activity_execution_df = synchronized_activitys_df.iloc[i*MINIMUM_EXECUTION_SIZE:(i+1)*MINIMUM_EXECUTION_SIZE,:]
            
        
            execution_activity_instance = Execution_Activity(no_activity_execution_df,time_begining,time_end,activity_comment)
            
            self.data["no_activity"].append_execution_sequence(execution_activity_instance)
         

    def convert_df_to_3d_np_array(self, df: pd.DataFrame) -> np.array:
        # df_np_array = np.array(df, dtype= self.numerical_type)
        # df_np_array = df_np_array.reshape(df_np_array.shape[0],df_np_array.shape[1],1)
        df_np_array = np.zeros((df.shape[0],df.shape[1],df.iloc[0,0].shape[0]),dtype= self.numerical_type)
        for i in tqdm(range(df_np_array.shape[0])):
            for j in range(df_np_array.shape[1]):                    
                    df_np_array[i,j] = df.iloc[i,j]
                    
        return df_np_array

    def shuffle_data_classification(self,input_ml_model_df: pd.DataFrame, input_series: pd.Series) -> tuple:
        joined_df = pd.concat([input_ml_model_df,input_series],axis=1)
        joined_df = joined_df.sample(frac=1).reset_index(drop=True)
        return joined_df.iloc[:,:-1], joined_df.iloc[:,-1]

    def calculate_dtw_centroid(self, arrays: np.array):
        model = TimeSeriesKMedoids(n_clusters=1, metric="dtw")
        model.fit(arrays)
        return model.cluster_centers_[0][0]

    def dtw_distance_to_centroid(self, arrays: np.array, centroid: np.array):
        distances = [dtw_distance(centroid, array) for array in arrays]
        return distances

    def detect_and_remove_outliers_positive_centroid(self, df_activity: pd.DataFrame, series_classes: pd.Series, acivity_name: str) -> tuple:#(pd.DataFrame, pd.Series, np.array):
        np_array_activity = self.convert_df_to_3d_np_array( df_activity)
    
        
        centroid = self.calculate_dtw_centroid(np_array_activity)
        distances_list = self.dtw_distance_to_centroid(np_array_activity, centroid)

        auxilary_df = df_activity.copy()

        auxilary_df["distances_to_centroid"] = distances_list
        auxilary_df["classes"] = series_classes

        print("auxilary_df",auxilary_df)

        percentile_dist = np.percentile(distances_list, 75)
        
        df_activity_outlier_cleaned = auxilary_df[ auxilary_df["distances_to_centroid"] < percentile_dist]
        series_classes_outlier_cleaned = df_activity_outlier_cleaned["classes"]
        df_activity_outlier_cleaned.drop("classes",axis=1,inplace=True)
        df_activity_outlier_cleaned.drop("distances_to_centroid",axis=1,inplace=True)

        df_outliers = auxilary_df[ auxilary_df["distances_to_centroid"] >= percentile_dist]
        df_outliers.drop("classes",axis=1,inplace=True)
        df_outliers.drop("distances_to_centroid",axis=1,inplace=True)


        np_array_outliers = self.convert_df_to_3d_np_array( df_outliers)
        negative_centroid = self.calculate_dtw_centroid(np_array_outliers)
        negative_centroid_normalized = negative_centroid / auxilary_df.shape[0]

        print("negative_centroid_normalized", negative_centroid_normalized)
        # np_array_activity_outlier_cleaned = self.convert_df_to_3d_np_array( df_activity_outlier_cleaned)
        # distances_list_outlier_cleaned = dtw_distance_to_centroid(np_array_activity_outlier_cleaned, centroid)

        # sns.set_theme()
        # sns.displot(data=distances_list_outlier_cleaned, kde = True, rug = True).set(title=f"Distribution of Scores {len(distances_list)}", xlabel="Scores")
        # print("dist fst to centroid", dtw_distance(calculate_dtw_centroid(np_array_activity), np_array_activity[0]), distances_list[0])
        # plt.savefig(f"distplot_{acivity_name}.png")
        # plt.clf()
        #DEBUGGING
        return df_activity_outlier_cleaned , series_classes_outlier_cleaned , centroid #negative_centroid_normalized

    def detect_and_remove_outliers_negative_centroid(self, df_activity: pd.DataFrame, series_classes: pd.Series, negative_centroid : np.array, acivity_name: str) -> tuple:#(pd.DataFrame, pd.Series):
        np_array_activity = self.convert_df_to_3d_np_array( df_activity)

        
        distances_list = self.dtw_distance_to_centroid(np_array_activity, negative_centroid)

        auxilary_df = df_activity.copy()

        print("auxilary_df", auxilary_df)
        print("df_activity", df_activity)

        auxilary_df["distances_to_centroid"] = distances_list
        auxilary_df["classes"] = series_classes

        print("auxilary_df", auxilary_df)
        print("df_activity", df_activity)

        percentile_dist = np.percentile(distances_list, 50)

        df_activity_outlier_cleaned = auxilary_df[ auxilary_df["distances_to_centroid"] > percentile_dist]
        series_classes_outlier_cleaned = df_activity_outlier_cleaned["classes"]
        df_activity_outlier_cleaned.drop("classes", axis=1, inplace=True)
        df_activity_outlier_cleaned.drop("distances_to_centroid", axis=1,inplace=True)
        
        return df_activity_outlier_cleaned , series_classes_outlier_cleaned

    def prepare_data_input_classification(self, ngram_size:int = 30, output_dtype="np_array", percentage_for_train: int = 0, percentage_for_test: int = 0, days_for_train:list = [], days_for_test:list = []) -> tuple :
        train_input_ml_model_df = pd.DataFrame()
        train_input_series = pd.Series(dtype= self.numerical_type)
        test_input_ml_model_df = pd.DataFrame()
        test_input_series = pd.Series(dtype= self.numerical_type)
        negative_centroid = np.zeros(ngram_size)

        list_test_activity_data = []

        # train_activity_df, train_activity_series, test_activity_df, test_activity_series = self.data["AS1"].prepare_activity_input_classification(percentage_for_train, percentage_for_test,days_for_train, days_for_test, ngram_size)
        # does_not_matter_1, does_not_matter_2, negative_centroid = self.detect_and_remove_outliers_positive_centroid(train_activity_df, train_activity_series, "train_AS1")
        total_number_of_samples_train = 0


        for activity_name in self.data:    
            activity = self.data[activity_name]
            train_activity_df, train_activity_series, test_activity_df, test_activity_series = activity.prepare_activity_input_classification(percentage_for_train, percentage_for_test,days_for_train, days_for_test, ngram_size)
            total_number_of_samples_train += train_activity_df.shape[0]

        # average_number_of_samples_train = total_number_of_samples_train // len(self.data)

        
        for activity_name in self.data:
            
            activity = self.data[activity_name]
            train_activity_df, train_activity_series, test_activity_df, test_activity_series = activity.prepare_activity_input_classification(percentage_for_train, percentage_for_test,days_for_train, days_for_test, ngram_size)
            
            train_activity_df.drop("Time",axis=1,inplace=True)
            test_activity_df.drop("Time",axis=1,inplace=True)

            
            # train_activity_df, train_activity_series, negative_centroid_part = self.detect_and_remove_outliers_positive_centroid(train_activity_df, train_activity_series, f"train_{activity_name}")
            # negative_centroid += negative_centroid_part
            # list_test_activity_data.append([test_activity_df, test_activity_series] )

            # train_activity_df, train_activity_series = self.detect_and_remove_outliers_negative_centroid(train_activity_df, train_activity_series, negative_centroid, "train")
            # test_activity_df, test_activity_series = self.detect_and_remove_outliers_negative_centroid(test_activity_df, test_activity_series, negative_centroid, "test")

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
        #     test_activity_df, test_activity_series = self.detect_and_remove_outliers_negative_centroid(test_activity_data[0], test_activity_data[1], negative_centroid, "test")
        #     test_input_ml_model_df = pd.concat([test_input_ml_model_df ,test_activity_df],axis=0)
        #     test_input_series =  pd.concat([test_input_series, test_activity_series])


        #DEBUGGING
        print("test_activity_df",test_activity_df)
        print("test_input_ml_model_df",test_input_ml_model_df)
        #test_input_ml_model_df, test_input_series = self.detect_and_remove_outliers_test(test_input_ml_model_df, test_input_series, negative_centroid, "test")
        
        #DEBUGGING
        print("test_activity_df",test_activity_df)

        print("train_input_ml_model_df",train_input_ml_model_df.shape)
        print("train_input_series",train_input_series.shape)
        print("test_input_ml_model_df",test_input_ml_model_df.shape)
        print("test_input_series",test_input_series.shape)

        print(train_input_ml_model_df.columns)
        print(test_activity_df.columns)


        #shuffled_train_input_ml_model_df, shuffled_train_input_series = self.shuffle_data_classification(train_input_ml_model_df, train_input_series)    
        #shuffled_test_input_ml_model_df, shuffled_test_input_series = self.shuffle_data_classification(test_input_ml_model_df, test_input_series)

        if output_dtype == "np_array":
            np_train_input_ml_model_df = self.convert_df_to_3d_np_array(train_input_ml_model_df)
            np_train_input_series = np.array(train_input_series)
            np_test_input_ml_model_df = self.convert_df_to_3d_np_array(test_input_ml_model_df)
            np_test_input_series = np.array(test_input_series)
        
       
        return np_train_input_ml_model_df, np_train_input_series, np_test_input_ml_model_df, np_test_input_series
    
    def save_data_to_csv(self, path: str):
        for activity_name in self.data:
            activity = self.data[activity_name]
            csv_data = activity.join_activity_executions()
            csv_data.to_csv(f"{path}/{activity_name}_samples.csv",index=False)

#'''

