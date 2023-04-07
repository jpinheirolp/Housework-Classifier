import pandas as pd
import numpy as np
from tqdm import tqdm

#Gets an dataframe with some empty values and fills with the previous values from the same column
def fills_empty_values(df):
    for col in df.columns:
        for row in df.index[1:]:
            if pd.isna(df[col][row]):
                value_to_update = df.at[row - 1,col]
                df.at[row,col] = value_to_update
    return df

#changes the frequency of a dataframe by turning every row that is in the esireq freq interval to one single row corresponding to the moving averadge
def create_mv_avg_df(df_wrong_freq, desired_freq):
    
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

    print(timestamp_mv_avg)
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
        
# Renames all columns form the dataframe by putting in df_prefix string into all column names
def rename_columns_for_merge(df,df_prefix,col_in_common = "Time_Interval"):
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
    df1_renamed = rename_columns_for_merge(df1,df1_prefix,col_in_common)
    df2_renamed = rename_columns_for_merge(df2,df2_prefix,col_in_common)
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


class Execution_Activity():
    def __init__(self, execution_df: pd.DataFrame):
        self.data = execution_df
class Sequence_Activity_Execution:
    def __init__(self) -> None:
        self.data = []
    def append_execution_sequence(self, execution: Execution_Activity):
        self.data.append(execution)
class House_Activitys_Learning_Base:
    def __init__(self, desired_freq: int, synchronized_activitys_df: pd.DataFrame ,time_schedule_df: pd.DataFrame):
        self.data = {}
        for activity in time_schedule_df["activity"].unique():
            self.data[activity] = Sequence_Activity_Execution()
        synchronized_activitys_df_time_stamps_set = set(synchronized_activitys_df["Time"])

        for row in time_schedule_df.iterrows:
            time_begining = time_schedule_df[row]["Started"]
            time_end = time_schedule_df[row]["Ended"]
            activity_name = time_schedule_df[row]["activity"]
            activitys_timenstamp_sequence = pd.date_range(start= time_begining, end= time_end, freq= f"{desired_freq}S")
            activitys_timenstamp_set = set(activitys_timenstamp_sequence)
            activitys_timenstamp_set_dif = activitys_timenstamp_set - synchronized_activitys_df_time_stamps_set
            
            execution_df = synchronized_activitys_df[synchronized_activitys_df["Time"] in activitys_timenstamp_set_dif]
            execution_activity_instance = Execution_Activity(execution_df)
            self.data[activity_name].append_execution_sequence(execution_activity_instance)
    
    

#''' thats just a brain storm for the next function

    
    # next step: separate the time sequences for each activity
    # class House_Activitys : dict {
    #   init(df_data,df_time,desired_freq):      
    #       df_data_time_stamps_set = set(df_data["Time"])
    #       for row in df_time.iterrows:
    #           execution_activity_df = pd.DataFrame()
    #           time_beggining = df_time[row]["Started"]
    #           time_end = df_time[row]["end"]
    #           activity_name = df_time[row]["activity"]
    #           activitys_timenstamp_sequence = pd.date_range(start= time_beggining, end= time_end, freq= f"{desired_freq}S")
    #           activitys_timenstamp_set = set(activitys_timenstamp_sequence)
    #           activitys_timenstamp_set_dif = activitys_timenstamp_set - df_data_time_stamps_set
    #           if len(activitys_timenstamp_set - activitys_timenstamp_set_dif) = 0:
    #              execution_activity_df = df_data[activitys_timenstamp_set_dif]["Time"]
    #              continue
    #           for time_stamp in activitys_timenstamp_set:
    #               execution_activity_df.append(time_stamp)
    #           Activitys_Dict[activity_name].append(execution_activity_df)
    #           
    #   class Activity: list [
    #       class Activity_Execution: df
    # ]
    # }
    #
    
    