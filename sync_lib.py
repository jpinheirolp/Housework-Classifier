import pandas as pd
import numpy as np
from tqdm import tqdm

def fills_empty_values(df):
    for col in df.columns:
        for row in df.index[1:]:
            if pd.isna(df[col][row]):
                value_to_update = df.at[row - 1,col]
                df.at[row,col] = value_to_update
    return df


def create_mv_avg_df(df_without_nan, desired_freq):
    df_without_nan["Time"] = pd.to_datetime(df_without_nan["Time"]) 

    
    initial_time_value = pd.to_datetime(df_without_nan["Time"].iloc[0])
    initial_time_value = initial_time_value.replace(second = (initial_time_value.second // desired_freq) * desired_freq)
    
    time_interval_col = pd.date_range(start=initial_time_value, periods= df_without_nan.shape[0] , freq= f"{desired_freq}S")
    
    list_columns = list(df_without_nan.columns)
    list_columns.remove("Time")

    values_mv_avg = np.zeros((df_without_nan.shape[0],df_without_nan.shape[1] - 1))
    df_mv_avg = pd.DataFrame(data= values_mv_avg,columns=list_columns)
    
    df_mv_avg["Time_Interval"] = time_interval_col
    
    columns_without_time = list_columns
    iterator_mv_avg = 0
    num_cols_interval = 0
    timestamp_mv_avg = initial_time_value
    for row in tqdm(df_without_nan.index):
        if df_without_nan.at[row,"Time"] > timestamp_mv_avg + pd.Timedelta(seconds=desired_freq): #df_mv_avg.at[iterator_mv_avg + 1,"Time_Interval"]
            for col in columns_without_time:
                df_mv_avg.at[iterator_mv_avg, col] /= num_cols_interval  
            df_mv_avg.at[iterator_mv_avg, "Time_Interval"] = timestamp_mv_avg
            num_cols_interval = 0
            iterator_mv_avg += 1
            measure_time = df_without_nan.at[row,"Time"]
            timestamp_mv_avg = measure_time.replace(second = (measure_time.second // desired_freq) * desired_freq)
            
        num_cols_interval += 1 

        for col in columns_without_time:
            df_mv_avg.at[iterator_mv_avg, col] += df_without_nan.at[row,col]
        

    for col in columns_without_time:
        df_mv_avg.at[iterator_mv_avg, col] /= num_cols_interval  
    df_mv_avg.at[iterator_mv_avg, "Time_Interval"] = timestamp_mv_avg
    df_mv_avg = df_mv_avg.iloc[:iterator_mv_avg+1,:]
    return df_mv_avg
    #'''
        
def rename_columns_for_merge(df,df_prefix,col_in_common = "Time_Interval"):
    columns_df_renamed = []
    list_columns_df = list(df.columns)
    list_columns_df.remove(col_in_common)
    for col in list_columns_df:
        columns_df_renamed.append(df_prefix + col)
    columns_df_renamed.append(col_in_common)  
    df.columns = columns_df_renamed
    return df
    
      
def merge_ajusting_col_names(df1,df2,df1_prefix,df2_prefix,col_in_common = "Time_Interval"):
    df1_renamed = rename_columns_for_merge(df1,df1_prefix,col_in_common)
    df2_renamed = rename_columns_for_merge(df2,df2_prefix,col_in_common)
    return df1_renamed.merge(df2_renamed,how="inner",on=col_in_common)


#'''

def select_intersecting_rows(df1, df2, intersection_column): # returns empty df, find why
    intersection_set = set(df1[intersection_column]).intersection(set(df2[intersection_column]))
    df1_intersect = df1.loc[(df1[intersection_column].isin(intersection_set))]
    df2_intersect = df2.loc[(df2[intersection_column].isin(intersection_set))]
    return df1_intersect, df2_intersect

def remove_nan_dfs(df1_intersect,df2_intersect,window_size= 5):
    # Loop through the DataFrame with a sliding window
    for i in range(0, len(df1_intersect) - window_size + 1):
        window_1 = df1_intersect[i:i + window_size]
        window_2 = df1_intersect[i:i + window_size]
        # Check if all rows in the window have at least one NaN value in both dfs
        if window_1.isnull().any(axis=1).all() and window_2.isnull().any(axis=1).all():
            
            df1_intersect.drop(axis=0, index= window_1.index,inplace = True)
            df2_intersect.drop(axis=0, index= window_2.index,inplace = True)

    df1_without_nan = df1_intersect.reset_index(drop=True)
    df2_without_nan = df2_intersect.reset_index(drop=True)
    return df1_without_nan,df2_without_nan

    
            
#'''

    
    # next step: think about how to do mv avg in all full tables and merge them, taking in acount the time it might take